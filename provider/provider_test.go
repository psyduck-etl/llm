package provider

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

var testSchema = json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`)

func TestConfigBindExactlyOne(t *testing.T) {
	cases := map[string]Config{
		"none": {},
		"both": {
			Anthropic: &anthropicConfig{APIKey: "k", Model: "m"},
			Ollama:    &ollamaConfig{Model: "m"},
		},
	}
	for name, config := range cases {
		if _, err := config.Bind(); err == nil {
			t.Errorf("%s: expected exactly-one-of error, got nil", name)
		}
	}
}

func TestConfigBindSelects(t *testing.T) {
	anthro := Config{Anthropic: &anthropicConfig{APIKey: "k", Model: "m"}}
	client, err := anthro.Bind()
	if err != nil {
		t.Fatalf("bind anthropic: %v", err)
	}
	if _, ok := client.(*anthropic); !ok {
		t.Errorf("expected *anthropic, got %T", client)
	}

	oll := Config{Ollama: &ollamaConfig{Model: "m"}}
	client, err = oll.Bind()
	if err != nil {
		t.Fatalf("bind ollama: %v", err)
	}
	if _, ok := client.(*ollama); !ok {
		t.Errorf("expected *ollama, got %T", client)
	}
}

func TestConfigBindValidates(t *testing.T) {
	cases := map[string]Config{
		"anthropic missing api-key": {Anthropic: &anthropicConfig{Model: "m"}},
		"anthropic missing model":   {Anthropic: &anthropicConfig{APIKey: "k"}},
		"ollama missing model":      {Ollama: &ollamaConfig{}},
	}
	for name, config := range cases {
		if _, err := config.Bind(); err == nil {
			t.Errorf("%s: expected validation error, got nil", name)
		}
	}
}

func TestAnthropicComplete(t *testing.T) {
	got := anthropicRequest{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path = %q, want /v1/messages", r.URL.Path)
		}
		if key := r.Header.Get("X-Api-Key"); key != "test-key" {
			t.Errorf("x-api-key = %q", key)
		}
		if version := r.Header.Get("Anthropic-Version"); version != anthropicVersion {
			t.Errorf("anthropic-version = %q", version)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode request: %v", err)
		}
		_, _ = w.Write([]byte(`{
			"content": [{"type": "tool_use", "name": "emit", "input": {"name": "psyduck"}}],
			"stop_reason": "tool_use"
		}`))
	}))
	defer server.Close()

	config := Config{Anthropic: &anthropicConfig{APIKey: "test-key", Model: "test-model", BaseURL: server.URL}}
	client, err := config.Bind()
	if err != nil {
		t.Fatalf("bind: %v", err)
	}

	result, err := client.Complete(context.Background(), "system prompt", "user text", testSchema)
	if err != nil {
		t.Fatalf("complete: %v", err)
	}

	if got.Model != "test-model" {
		t.Errorf("request model = %q", got.Model)
	}
	if got.System != "system prompt" {
		t.Errorf("request system = %q", got.System)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" || got.Messages[0].Content != "user text" {
		t.Errorf("request messages = %+v", got.Messages)
	}
	if len(got.Tools) != 1 || string(got.Tools[0].InputSchema) != string(testSchema) {
		t.Errorf("request tools = %+v", got.Tools)
	}
	if got.ToolChoice.Type != "tool" || got.ToolChoice.Name != anthropicToolName {
		t.Errorf("request tool_choice = %+v", got.ToolChoice)
	}
	if string(result) != `{"name": "psyduck"}` {
		t.Errorf("result = %s", result)
	}
}

func TestAnthropicCompleteNoToolUse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"content": [{"type": "text", "text": "sorry"}], "stop_reason": "end_turn"}`))
	}))
	defer server.Close()

	config := Config{Anthropic: &anthropicConfig{APIKey: "k", Model: "m", BaseURL: server.URL}}
	client, err := config.Bind()
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	if _, err := client.Complete(context.Background(), "s", "u", testSchema); err == nil {
		t.Error("expected no-tool-use error, got nil")
	}
}

func TestAnthropicCompleteUpstreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"message": "bad schema"}}`))
	}))
	defer server.Close()

	config := Config{Anthropic: &anthropicConfig{APIKey: "k", Model: "m", BaseURL: server.URL}}
	client, err := config.Bind()
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	_, err = client.Complete(context.Background(), "s", "u", testSchema)
	if err == nil {
		t.Fatal("expected upstream error, got nil")
	}
	if !strings.Contains(err.Error(), "bad schema") {
		t.Errorf("error should carry the body snippet, got %v", err)
	}
}

// ollamaCapture serves one canned ollama chat response, capturing each
// request body decoded as both the request struct and a raw key set (for
// asserting which keys were present at all).
func ollamaCapture(t *testing.T, got *ollamaRequest, keys *map[string]json.RawMessage) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("path = %q, want /api/chat", r.URL.Path)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
		}
		if err := json.Unmarshal(body, got); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if err := json.Unmarshal(body, keys); err != nil {
			t.Errorf("decode request keys: %v", err)
		}
		_, _ = w.Write([]byte(`{"message": {"role": "assistant", "content": "{\"name\": \"psyduck\"}"}, "done": true}`))
	}))
}

func TestOllamaComplete(t *testing.T) {
	got := ollamaRequest{}
	keys := map[string]json.RawMessage{}
	server := ollamaCapture(t, &got, &keys)
	defer server.Close()

	config := Config{Ollama: &ollamaConfig{Host: server.URL, Model: "test-model"}}
	client, err := config.Bind()
	if err != nil {
		t.Fatalf("bind: %v", err)
	}

	result, err := client.Complete(context.Background(), "system prompt", "user text", testSchema)
	if err != nil {
		t.Fatalf("complete: %v", err)
	}

	if got.Model != "test-model" {
		t.Errorf("request model = %q", got.Model)
	}
	if got.Stream {
		t.Error("request stream should be false")
	}
	if string(got.Format) != string(testSchema) {
		t.Errorf("request format = %s", got.Format)
	}
	want := []ollamaMessage{{Role: "system", Content: "system prompt"}, {Role: "user", Content: "user text"}}
	if len(got.Messages) != 2 || got.Messages[0] != want[0] || got.Messages[1] != want[1] {
		t.Errorf("request messages = %+v", got.Messages)
	}
	if string(result) != `{"name": "psyduck"}` {
		t.Errorf("result = %s", result)
	}
	for _, key := range []string{"think", "options"} {
		if _, ok := keys[key]; ok {
			t.Errorf("unset %s should be omitted from the request, got %s", key, keys[key])
		}
	}
}

func TestOllamaCompleteTuning(t *testing.T) {
	got := ollamaRequest{}
	keys := map[string]json.RawMessage{}
	server := ollamaCapture(t, &got, &keys)
	defer server.Close()

	think, temperature, numCtx := false, 0.0, 8192
	seed, topK, topP, minP := 42, 40, 0.9, 0.05
	repeatPenalty, repeatLastN, numPredict := 1.1, 64, 512
	config := Config{Ollama: &ollamaConfig{
		Host:          server.URL,
		Model:         "m",
		Think:         &think,
		Temperature:   &temperature,
		NumCtx:        &numCtx,
		Seed:          &seed,
		TopK:          &topK,
		TopP:          &topP,
		MinP:          &minP,
		RepeatPenalty: &repeatPenalty,
		RepeatLastN:   &repeatLastN,
		NumPredict:    &numPredict,
	}}
	client, err := config.Bind()
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	if _, err := client.Complete(context.Background(), "s", "u", testSchema); err != nil {
		t.Fatalf("complete: %v", err)
	}

	if got.Think == nil || *got.Think {
		t.Errorf("request think = %v, want explicit false", got.Think)
	}
	if got.Options == nil || got.Options.Temperature == nil || *got.Options.Temperature != 0 {
		t.Errorf("request options.temperature = %+v, want explicit 0", got.Options)
	}
	if got.Options == nil || got.Options.NumCtx == nil || *got.Options.NumCtx != 8192 {
		t.Errorf("request options.num_ctx = %+v, want 8192", got.Options)
	}
	// The wire shape matters: think rides at the top level, the sampling
	// knobs inside options — misplacing them would be silently ignored.
	if want := `{"temperature":0,"num_ctx":8192,"seed":42,"top_k":40,"top_p":0.9,"min_p":0.05,"repeat_penalty":1.1,"repeat_last_n":64,"num_predict":512}`; string(keys["options"]) != want {
		t.Errorf("options = %s, want %s", keys["options"], want)
	}
	if string(keys["think"]) != "false" {
		t.Errorf("think = %s, want false", keys["think"])
	}
}

func TestOllamaCompleteInvalidContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"message": {"role": "assistant", "content": "not json"}}`))
	}))
	defer server.Close()

	config := Config{Ollama: &ollamaConfig{Host: server.URL, Model: "m"}}
	client, err := config.Bind()
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	if _, err := client.Complete(context.Background(), "s", "u", testSchema); err == nil {
		t.Error("expected invalid-content error, got nil")
	}
}

// TestOllamaCompleteTimeoutIsRetryable simulates a wedged generation: the
// server never responds inside request-timeout-ms. Complete must wrap the
// resulting deadline in ErrBadContent (not surface it as a bare transport
// error) so CompleteJSON's ask-attempts loop retries the record instead of
// failing it outright — a fresh ask can beat a runaway generation where the
// last one hung.
func TestOllamaCompleteTimeoutIsRetryable(t *testing.T) {
	unblock := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-unblock // outlives the test's short request-timeout-ms
	}))
	// close(unblock) must run before server.Close() (which waits for the
	// handler to return), so it's deferred second — defers unwind LIFO.
	defer server.Close()
	defer close(unblock)

	config := Config{
		Ollama:           &ollamaConfig{Host: server.URL, Model: "m"},
		RequestTimeoutMS: 50,
	}
	client, err := config.Bind()
	if err != nil {
		t.Fatalf("bind: %v", err)
	}

	_, err = client.Complete(context.Background(), "s", "u", testSchema)
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
	if !errors.Is(err, ErrBadContent) {
		t.Fatalf("want timeout wrapped in ErrBadContent (retryable), got: %v", err)
	}
}

func TestRetryTransportBacksOff(t *testing.T) {
	slept := []time.Duration{}
	restore := backoffSleep
	backoffSleep = func(ctx context.Context, d time.Duration) error {
		slept = append(slept, d)
		return nil
	}
	defer func() { backoffSleep = restore }()

	failures := 2
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if requests <= failures {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write([]byte(`{"message": {"role": "assistant", "content": "{}"}}`))
	}))
	defer server.Close()

	config := Config{Ollama: &ollamaConfig{Host: server.URL, Model: "m"}}
	client, err := config.Bind()
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	if _, err := client.Complete(context.Background(), "s", "u", testSchema); err != nil {
		t.Fatalf("complete should succeed after retries: %v", err)
	}
	if requests != failures+1 {
		t.Errorf("requests = %d, want %d", requests, failures+1)
	}
	if len(slept) != failures {
		t.Errorf("sleeps = %d, want %d", len(slept), failures)
	}
}

func TestRetryTransportGivesUp(t *testing.T) {
	restore := backoffSleep
	backoffSleep = func(ctx context.Context, d time.Duration) error { return nil }
	defer func() { backoffSleep = restore }()

	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer server.Close()

	config := Config{
		Ollama:           &ollamaConfig{Host: server.URL, Model: "m"},
		RetryGiveupAfter: 3,
	}
	client, err := config.Bind()
	if err != nil {
		t.Fatalf("bind: %v", err)
	}
	if _, err := client.Complete(context.Background(), "s", "u", testSchema); err == nil {
		t.Error("expected upstream error after give-up, got nil")
	}
	if requests != 3 {
		t.Errorf("requests = %d, want 3", requests)
	}
}
