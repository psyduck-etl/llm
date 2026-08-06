package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/psyduck-etl/llm/provider"
	"github.com/psyduck-etl/sdk"
	"github.com/psyduck-etl/sdk/data"
)

// filterSystem frames the model as a yes/no judge. The %s slot carries the
// enumerated asks (name: question), one per criterion. The model answers each
// as a boolean in the response schema; it never sees the weights — scoring is
// the stage's job.
const filterSystem = `You are a filtering stage in an ETL pipeline. Answer each of the following yes/no questions about the message truthfully, based only on what the message states. Each question is a boolean property in the response schema, keyed by the name shown here:

%s`

type filterConfig struct {
	provider.Config
	data.InputCodec

	// Criteria is the set of yes/no questions the model answers about each
	// record, keyed by name. Each answered-true criterion adds its weight to
	// the record's score.
	Criteria map[string]criterion `psy:"criteria"`

	// Threshold is the summed-weight score at or above which a record passes.
	Threshold float64 `psy:"threshold"`
}

// criterion is one scored question: the model is asked Ask (yes/no), and a
// true answer contributes Weight to the record's score. Weight may be
// negative — a dealbreaker drags the score below any positive threshold.
type criterion struct {
	Ask    string  `psy:"ask"`
	Weight float64 `psy:"weight"`
}

// filterSchema builds the response shape from the criteria: an object with one
// boolean property per criterion name, all required. Property descriptions
// carry the asks; json.Marshal sorts map keys so the schema is deterministic
// across runs (temperature-0 output is only reproducible if the prompt is).
func filterSchema(criteria map[string]criterion) json.RawMessage {
	type prop struct {
		Type        string `json:"type"`
		Description string `json:"description"`
	}
	props := make(map[string]prop, len(criteria))
	required := make([]string, 0, len(criteria))
	for name, c := range criteria {
		props[name] = prop{Type: "boolean", Description: c.Ask}
		required = append(required, name)
	}
	sort.Strings(required)

	doc := struct {
		Type       string          `json:"type"`
		Properties map[string]prop `json:"properties"`
		Required   []string        `json:"required"`
	}{Type: "object", Properties: props, Required: required}
	out, _ := json.Marshal(doc)
	return out
}

// filterQuestions renders the criteria as a "- name: ask" list for the system
// prompt, in the same sorted order as the schema.
func filterQuestions(criteria map[string]criterion) string {
	names := make([]string, 0, len(criteria))
	for name := range criteria {
		names = append(names, name)
	}
	sort.Strings(names)

	var b strings.Builder
	for _, name := range names {
		fmt.Fprintf(&b, "- %s: %s\n", name, criteria[name].Ask)
	}
	return strings.TrimRight(b.String(), "\n")
}

// provideFilter builds the llm-filter stage: a pure gate that asks the model
// every criterion's yes/no question, scores the record by summing the weights
// of the true answers, and passes the original bytes through untouched when
// the score clears the threshold. There is no emit codec — kept records are
// exactly the bytes that arrived.
func provideFilter(_ context.Context, parse sdk.Parser) (sdk.Transformer, error) {
	config := &filterConfig{
		InputCodec: data.InputCodec{Accept: "string"},
	}
	if err := parse(config); err != nil {
		return nil, err
	}
	if len(config.Criteria) == 0 {
		return nil, fmt.Errorf("llm-filter: at least one criterion is required")
	}
	for name, c := range config.Criteria {
		if c.Ask == "" {
			return nil, fmt.Errorf("llm-filter: criterion %q has no ask", name)
		}
	}
	if err := config.InputCodec.Bind(); err != nil {
		return nil, fmt.Errorf("llm-filter: %w", err)
	}

	client, err := config.Config.Bind()
	if err != nil {
		return nil, fmt.Errorf("llm-filter: %w", err)
	}
	return makeFilter(client, config), nil
}

func makeFilter(client provider.Client, config *filterConfig) sdk.Transformer {
	system := fmt.Sprintf(filterSystem, filterQuestions(config.Criteria))
	schema := filterSchema(config.Criteria)

	return func(ctx context.Context, in <-chan []byte, out chan<- []byte, errs chan<- error) {
		defer close(out)
		for msg := range in {
			text, err := renderText(&config.InputCodec, msg)
			if err != nil {
				if !sendErr(ctx, errs, fmt.Errorf("llm-filter: %w", err)) {
					return
				}
				continue
			}

			answers := map[string]bool{}
			if err := provider.CompleteJSON(ctx, client, config.AskAttempts, system, text, schema, &answers, nil); err != nil {
				if ctx.Err() != nil {
					return
				}
				if !sendErr(ctx, errs, fmt.Errorf("llm-filter: %w", err)) {
					return
				}
				continue
			}

			score, matched := scoreRecord(config.Criteria, answers)
			if score < config.Threshold {
				logger.Debug("dropping record below threshold",
					"resource", "llm-filter", "score", score, "threshold", config.Threshold, "matched", matched)
				continue
			}

			select {
			case out <- msg:
			case <-ctx.Done():
				return
			}
		}
	}
}

// scoreRecord sums the weights of the criteria the model answered true and
// returns the total plus the names of those matched criteria (sorted, for
// stable logging). Answers for unknown names are ignored; a missing answer
// counts as false.
func scoreRecord(criteria map[string]criterion, answers map[string]bool) (float64, []string) {
	var score float64
	matched := make([]string, 0, len(answers))
	for name, c := range criteria {
		if answers[name] {
			score += c.Weight
			matched = append(matched, name)
		}
	}
	sort.Strings(matched)
	return score, matched
}
