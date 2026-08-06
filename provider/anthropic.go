package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	anthropicDefaultBaseURL = "https://api.anthropic.com"

	// anthropicVersion pins the messages API revision we speak.
	anthropicVersion = "2023-06-01"

	// anthropicToolName names the single tool we force the model to call.
	// The tool is a fiction — its input schema is the caller's schema, so
	// forcing the call is what makes the model emit conforming JSON.
	anthropicToolName = "emit"
)

// anthropicConfig configures the Anthropic messages API backend.
type anthropicConfig struct {
	APIKey    string `psy:"api-key"`
	Model     string `psy:"model"`
	MaxTokens uint   `psy:"max-tokens"`
	BaseURL   string `psy:"base-url"`
}

// bind validates the config and returns the backend on the shared client.
// requestTimeout <= 0 leaves each Complete call unbounded.
func (c *anthropicConfig) bind(httpClient *http.Client, requestTimeout time.Duration) (Client, error) {
	if c.APIKey == "" {
		return nil, fmt.Errorf("llm: anthropic: api-key is required")
	}
	if c.Model == "" {
		return nil, fmt.Errorf("llm: anthropic: model is required")
	}

	a := &anthropic{config: *c, http: httpClient, requestTimeout: requestTimeout}
	if a.config.BaseURL == "" {
		a.config.BaseURL = anthropicDefaultBaseURL
	}
	if a.config.MaxTokens == 0 {
		a.config.MaxTokens = 1024
	}
	return a, nil
}

type anthropic struct {
	config         anthropicConfig
	http           *http.Client
	requestTimeout time.Duration
}

// Anthropic messages API request/response shapes, reduced to the fields we
// use. Structured output rides on forced tool use: one tool whose
// input_schema is the caller's schema, tool_choice pinned to it, so the
// only thing the model can do is emit a conforming input object.
type anthropicRequest struct {
	Model      string              `json:"model"`
	MaxTokens  uint                `json:"max_tokens"`
	System     string              `json:"system,omitempty"`
	Messages   []anthropicMessage  `json:"messages"`
	Tools      []anthropicTool     `json:"tools"`
	ToolChoice anthropicToolChoice `json:"tool_choice"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anthropicToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type anthropicResponse struct {
	Content []struct {
		Type  string          `json:"type"`
		Input json.RawMessage `json:"input"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
}

func (a *anthropic) Complete(ctx context.Context, system, user string, schema json.RawMessage) (json.RawMessage, error) {
	body, err := json.Marshal(anthropicRequest{
		Model:     a.config.Model,
		MaxTokens: a.config.MaxTokens,
		System:    system,
		Messages:  []anthropicMessage{{Role: "user", Content: user}},
		Tools: []anthropicTool{{
			Name:        anthropicToolName,
			Description: "Emit the structured result.",
			InputSchema: schema,
		}},
		ToolChoice: anthropicToolChoice{Type: "tool", Name: anthropicToolName},
	})
	if err != nil {
		return nil, fmt.Errorf("llm: anthropic: encode request: %w", err)
	}

	reqCtx, cancel := withRequestTimeout(ctx, a.requestTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost,
		strings.TrimSuffix(a.config.BaseURL, "/")+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("llm: anthropic: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Api-Key", a.config.APIKey)
	req.Header.Set("Anthropic-Version", anthropicVersion)

	resp, err := a.http.Do(req)
	if err != nil {
		if requestTimedOut(reqCtx) {
			return nil, fmt.Errorf("llm: anthropic: request-timeout-ms exceeded: %w: %w", err, ErrBadContent)
		}
		return nil, fmt.Errorf("llm: anthropic: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("llm: anthropic: %s: %s", resp.Status, errBodySnippet(resp.Body))
	}

	parsed := anthropicResponse{}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, fmt.Errorf("llm: anthropic: decode response: %w", err)
	}
	for _, block := range parsed.Content {
		if block.Type == "tool_use" {
			return block.Input, nil
		}
	}
	return nil, fmt.Errorf("llm: anthropic: response carried no tool_use block (stop_reason %q): %w", parsed.StopReason, ErrBadContent)
}

// errBodySnippet reads a bounded prefix of an error response body for
// inclusion in the error message, so upstream diagnostics survive without
// risking an unbounded read.
func errBodySnippet(r io.Reader) string {
	b, _ := io.ReadAll(io.LimitReader(r, 512))
	if len(b) == 0 {
		return "(empty body)"
	}
	return string(bytes.TrimSpace(b))
}
