package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/psyduck-etl/llm/provider"
	"github.com/psyduck-etl/sdk"
	"github.com/psyduck-etl/sdk/data"
)

// transformSystem frames the model as an extraction stage. The %s slot
// carries the pipeline author's prompt; keep/data mirror envelopeSchema so
// the instructions and the schema tell one story.
const transformSystem = `You are a data-extraction stage in an ETL pipeline. Extract data from the message according to these instructions:

%s

Respond with:
- keep: your 0-1 rating of how much data relevant to the instructions the message contains (0 = nothing relevant)
- data: the extracted result`

type transformConfig struct {
	provider.Config
	data.InputCodec
	data.OutputCodec

	// Prompt is the pipeline author's extraction instructions.
	Prompt string `psy:"prompt"`

	// Schema is the JSON schema (type object) the extracted data must
	// conform to.
	Schema string `psy:"schema"`

	// Threshold drops records whose keep score falls below it. The zero
	// default keeps everything; raising it filters out records with no or
	// thin extractable data.
	Threshold float64 `psy:"threshold"`

	// IncludeTransformed carries the input forward: when true, each
	// emitted record gains an "_input" field holding the text the model
	// extracted from. A post-extraction step in code — the model never
	// sees the field, so it can't garble it.
	IncludeTransformed bool `psy:"include-transformed"`

	// VerifyGrounded lists dot-paths into the extracted data — "*" wildcards
	// an array, e.g. "roles.*.apply" — whose string values must appear
	// verbatim in the source text. A value that doesn't is treated exactly
	// like invalid JSON: the ask-attempts loop re-asks, since the model is
	// sampled and a fresh attempt may ground where the last one didn't.
	// Empty string values are exempt — they mean "not found", not
	// fabrication — and non-string values are never checked.
	VerifyGrounded []string `psy:"verify-grounded"`
}

// transformResult is the envelope the model fills: the extraction plus a
// relevance score gating it.
type transformResult struct {
	Keep float64         `json:"keep"`
	Data json.RawMessage `json:"data"`
}

// envelopeSchema wraps the author's schema so the model scores relevance
// alongside extraction, giving llm-transform the same keep/threshold gate
// as llm-filter.
func envelopeSchema(user []byte) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(
		`{"type":"object","properties":{"keep":{"type":"number","minimum":0,"maximum":1,"description":"how much data relevant to the instructions the message contains, 0-1"},"data":%s},"required":["keep","data"]}`,
		user,
	))
}

// provideTransform builds the llm-transform stage: decode a record, hand
// its text to the model with the enveloped schema, and emit the extracted
// data — or drop the record when the model scores it below threshold.
func provideTransform(_ context.Context, parse sdk.Parser) (sdk.Transformer, error) {
	config := &transformConfig{
		InputCodec:  data.InputCodec{Accept: "string"},
		OutputCodec: data.OutputCodec{Emit: "json"},
	}
	if err := parse(config); err != nil {
		return nil, err
	}
	if config.Prompt == "" {
		return nil, fmt.Errorf("llm-transform: prompt is required")
	}
	if config.Schema == "" {
		return nil, fmt.Errorf("llm-transform: schema is required")
	}
	if !json.Valid([]byte(config.Schema)) {
		return nil, fmt.Errorf("llm-transform: schema is not a valid JSON document")
	}
	if config.OutputCodec.Sparse() {
		return nil, fmt.Errorf("llm-transform: extracted data is structured; emit must be a structured codec like json, not %q", config.OutputCodec.Emit)
	}
	if err := config.InputCodec.Bind(); err != nil {
		return nil, fmt.Errorf("llm-transform: %w", err)
	}
	if err := config.OutputCodec.Bind(); err != nil {
		return nil, fmt.Errorf("llm-transform: %w", err)
	}

	client, err := config.Config.Bind()
	if err != nil {
		return nil, fmt.Errorf("llm-transform: %w", err)
	}
	return makeTransform(client, config), nil
}

func makeTransform(client provider.Client, config *transformConfig) sdk.Transformer {
	system := fmt.Sprintf(transformSystem, config.Prompt)
	schema := envelopeSchema([]byte(config.Schema))

	return func(ctx context.Context, in <-chan []byte, out chan<- []byte, errs chan<- error) {
		defer close(out)
		for msg := range in {
			text, err := renderText(&config.InputCodec, msg)
			if err != nil {
				if !sendErr(ctx, errs, fmt.Errorf("llm-transform: %w", err)) {
					return
				}
				continue
			}

			result := transformResult{}
			validate := func(v any) error {
				if len(config.VerifyGrounded) == 0 {
					return nil
				}
				r, ok := v.(*transformResult)
				if !ok {
					return nil
				}
				return verifyGrounded(config.VerifyGrounded, r.Data, text)
			}
			if err := provider.CompleteJSON(ctx, client, config.AskAttempts, system, text, schema, &result, validate); err != nil {
				if ctx.Err() != nil {
					return
				}
				if !sendErr(ctx, errs, fmt.Errorf("llm-transform: %w", err)) {
					return
				}
				continue
			}
			if result.Keep < config.Threshold {
				logger.Debug("dropping record below threshold",
					"resource", "llm-transform", "keep", result.Keep, "threshold", config.Threshold)
				continue
			}

			var native any
			if err := json.Unmarshal(result.Data, &native); err != nil {
				if !sendErr(ctx, errs, fmt.Errorf("llm-transform: decode extracted data: %w", err)) {
					return
				}
				continue
			}
			if config.IncludeTransformed {
				obj, ok := native.(map[string]any)
				if !ok {
					if !sendErr(ctx, errs, fmt.Errorf("llm-transform: include-transformed needs the extracted data to be a JSON object, got %T", native)) {
						return
					}
					continue
				}
				obj["_input"] = text
			}
			encoded, err := config.OutputCodec.Encode(native)
			if err != nil {
				if !sendErr(ctx, errs, fmt.Errorf("llm-transform: encode record: %w", err)) {
					return
				}
				continue
			}

			select {
			case out <- encoded:
			case <-ctx.Done():
				return
			}
		}
	}
}

// verifyGrounded decodes raw and checks each configured path's string
// value(s) against source. A path is dot-separated, e.g. "roles.*.apply";
// "*" descends into every element of an array. Values that aren't strings,
// or paths that don't resolve, are skipped — this check only guards
// against a string field holding text absent from the source, since that's
// the shape a fabricated contact/apply value takes.
func verifyGrounded(paths []string, raw json.RawMessage, source string) error {
	var native any
	if err := json.Unmarshal(raw, &native); err != nil {
		return fmt.Errorf("decode extracted data: %w", err)
	}
	for _, path := range paths {
		if err := checkGrounded(native, strings.Split(path, "."), path, source); err != nil {
			return err
		}
	}
	return nil
}

func checkGrounded(v any, segs []string, path, source string) error {
	if len(segs) == 0 {
		s, ok := v.(string)
		if !ok || s == "" {
			return nil
		}
		if !strings.Contains(source, s) {
			return fmt.Errorf("field %q: value %q is not grounded in the source text", path, s)
		}
		return nil
	}

	seg, rest := segs[0], segs[1:]
	if seg == "*" {
		arr, ok := v.([]any)
		if !ok {
			return nil
		}
		for _, elem := range arr {
			if err := checkGrounded(elem, rest, path, source); err != nil {
				return err
			}
		}
		return nil
	}

	obj, ok := v.(map[string]any)
	if !ok {
		return nil
	}
	next, ok := obj[seg]
	if !ok {
		return nil
	}
	return checkGrounded(next, rest, path, source)
}
