package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const ollamaDefaultHost = "http://localhost:11434"

// ollamaConfig configures an Ollama server backend. The tuning fields are
// pointers so absent stays distinguishable from an explicit zero — nil
// omits the knob from the request entirely, leaving the server's own
// default in charge.
type ollamaConfig struct {
	Host  string `psy:"host"`
	Model string `psy:"model"`

	// Think toggles the model's thinking phase. Thinking models (qwen3
	// and kin) stall on structured output unless this is explicitly
	// false; the server also accepts think=false for models without a
	// thinking mode, so false is safe to set unconditionally.
	Think *bool `psy:"think"`

	// Temperature overrides the model's sampling temperature. Extraction
	// and scoring want ~0; the server default (typically 0.7) is tuned
	// for chat, not structured output.
	Temperature *float64 `psy:"temperature"`

	// NumCtx overrides the context window in tokens. The server default
	// (typically 4096) silently truncates long inputs — raise it when
	// records plus schema outgrow it.
	NumCtx *int `psy:"num-ctx"`

	// Sampling and repetition knobs, all nil-unless-set (server default in
	// charge when nil). NumPredict caps generated tokens — a hard ceiling
	// against runaway thinking loops.
	Seed          *int     `psy:"seed"`
	TopK          *int     `psy:"top-k"`
	TopP          *float64 `psy:"top-p"`
	MinP          *float64 `psy:"min-p"`
	RepeatPenalty *float64 `psy:"repeat-penalty"`
	RepeatLastN   *int     `psy:"repeat-last-n"`
	NumPredict    *int     `psy:"num-predict"`
}

// bind validates the config and returns the backend on the shared client.
// requestTimeout <= 0 leaves each Complete call unbounded.
func (c *ollamaConfig) bind(httpClient *http.Client, requestTimeout time.Duration) (Client, error) {
	if c.Model == "" {
		return nil, fmt.Errorf("llm: ollama: model is required")
	}

	o := &ollama{config: *c, http: httpClient, requestTimeout: requestTimeout}
	if o.config.Host == "" {
		o.config.Host = ollamaDefaultHost
	}
	return o, nil
}

type ollama struct {
	config         ollamaConfig
	http           *http.Client
	requestTimeout time.Duration
}

// Ollama /api/chat request/response shapes, reduced to the fields we use.
// Structured output rides on the format field: Ollama compiles the JSON
// schema into a sampling grammar, so the response content is guaranteed to
// parse against it.
type ollamaRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Format   json.RawMessage `json:"format"`
	Stream   bool            `json:"stream"`
	Think    *bool           `json:"think,omitempty"`
	Options  *ollamaOptions  `json:"options,omitempty"`
}

// ollamaOptions is the request's runtime tuning envelope; nil fields are
// omitted so the server's defaults apply to anything unset.
type ollamaOptions struct {
	Temperature   *float64 `json:"temperature,omitempty"`
	NumCtx        *int     `json:"num_ctx,omitempty"`
	Seed          *int     `json:"seed,omitempty"`
	TopK          *int     `json:"top_k,omitempty"`
	TopP          *float64 `json:"top_p,omitempty"`
	MinP          *float64 `json:"min_p,omitempty"`
	RepeatPenalty *float64 `json:"repeat_penalty,omitempty"`
	RepeatLastN   *int     `json:"repeat_last_n,omitempty"`
	NumPredict    *int     `json:"num_predict,omitempty"`
}

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaResponse struct {
	Message ollamaMessage `json:"message"`
}

func (o *ollama) Complete(ctx context.Context, system, user string, schema json.RawMessage) (json.RawMessage, error) {
	messages := make([]ollamaMessage, 0, 2)
	if system != "" {
		messages = append(messages, ollamaMessage{Role: "system", Content: system})
	}
	messages = append(messages, ollamaMessage{Role: "user", Content: user})

	opts := ollamaOptions{
		Temperature:   o.config.Temperature,
		NumCtx:        o.config.NumCtx,
		Seed:          o.config.Seed,
		TopK:          o.config.TopK,
		TopP:          o.config.TopP,
		MinP:          o.config.MinP,
		RepeatPenalty: o.config.RepeatPenalty,
		RepeatLastN:   o.config.RepeatLastN,
		NumPredict:    o.config.NumPredict,
	}
	var options *ollamaOptions
	if opts != (ollamaOptions{}) {
		options = &opts
	}

	body, err := json.Marshal(ollamaRequest{
		Model:    o.config.Model,
		Messages: messages,
		Format:   schema,
		Stream:   false,
		Think:    o.config.Think,
		Options:  options,
	})
	if err != nil {
		return nil, fmt.Errorf("llm: ollama: encode request: %w", err)
	}

	reqCtx, cancel := withRequestTimeout(ctx, o.requestTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost,
		strings.TrimSuffix(o.config.Host, "/")+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("llm: ollama: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.http.Do(req)
	if err != nil {
		if requestTimedOut(reqCtx) {
			return nil, fmt.Errorf("llm: ollama: request-timeout-ms exceeded: %w: %w", err, ErrBadContent)
		}
		// A mid-generation server EOF is a bad response, not a dead
		// server — fold it into ErrBadContent so ask-attempts re-asks.
		if isEOF(err) {
			return nil, fmt.Errorf("llm: ollama: connection closed mid-response: %w: %w", err, ErrBadContent)
		}
		return nil, fmt.Errorf("llm: ollama: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("llm: ollama: %s: %s", resp.Status, errBodySnippet(resp.Body))
	}

	parsed := ollamaResponse{}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		if isEOF(err) {
			return nil, fmt.Errorf("llm: ollama: truncated response body: %w: %w", err, ErrBadContent)
		}
		return nil, fmt.Errorf("llm: ollama: decode response: %w", err)
	}

	content := json.RawMessage(parsed.Message.Content)
	if !json.Valid(content) {
		return nil, fmt.Errorf("llm: ollama: response content is not valid JSON: %s: %w", errBodySnippet(bytes.NewReader(content)), ErrBadContent)
	}
	return content, nil
}

// isEOF reports whether err is a premature end of the response stream — a
// server that closed the connection mid-generation rather than a clean send.
func isEOF(err error) bool {
	return errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF)
}
