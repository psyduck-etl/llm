// Package provider selects and drives the LLM backend behind the llm
// plugin's resources. Exactly one backend block (anthropic or ollama) is
// configured per resource; Config.Bind enforces that and returns the
// backend as a Client — the single schema-constrained completion surface
// the resources program against. Both backends speak plain HTTP through a
// shared retrying client (backoff.go), so rate-limit and transient-error
// behavior is uniform regardless of which backend is configured.
package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/psyduck-etl/sdk"
)

// Client is the common surface every backend implements: one
// schema-constrained completion. system carries the resource's role
// instructions, user the rendered record text, and schema a JSON schema
// (type object) the response must conform to. The returned bytes are the
// conforming JSON object.
type Client interface {
	Complete(ctx context.Context, system, user string, schema json.RawMessage) (json.RawMessage, error)
}

// Config is the shared backend selection embedded in every llm resource
// config. Exactly one backend field may be set; Bind rejects none or both.
// The remaining fields tune the shared HTTP client all backends run on.
type Config struct {
	Anthropic *anthropicConfig `psy:"anthropic"`
	Ollama    *ollamaConfig    `psy:"ollama"`

	// RequestTimeoutMS bounds one whole Complete call, retries included,
	// so a wedged upstream can't stall the stage forever. 0 leaves the
	// call unbounded (the stage context still cancels it on shutdown).
	// Expiry is folded into the ask-attempts retry loop (ErrBadContent),
	// not surfaced as an immediate error — a fresh ask can beat a runaway
	// generation that never emitted a stop token where the last one
	// couldn't.
	RequestTimeoutMS uint `psy:"request-timeout-ms"`

	// RetryGiveupAfter is the number of consecutive retryable responses
	// (429 or 5xx) after which the shared client stops backing off and
	// surfaces the error. 0 retries indefinitely.
	RetryGiveupAfter uint `psy:"retry-giveup-after"`

	// AskAttempts bounds how many times one record is asked of the model
	// before its content failure (invalid/undecodable JSON, or a
	// request-timeout-ms expiry) is surfaced. 0 re-asks forever until the
	// response is usable, matching RetryGiveupAfter's 0-means-unbounded.
	// Beware that a deterministic content failure (e.g. a schema that
	// can't fit the token budget) or a persistently wedged generation
	// replays forever at 0; set a bound when that risk matters.
	AskAttempts uint `psy:"ask-attempts"`
}

// Bind validates the exactly-one-of backend constraint and returns the
// configured backend wired to the shared retrying HTTP client.
func (c *Config) Bind() (Client, error) {
	if (c.Anthropic == nil) == (c.Ollama == nil) {
		return nil, fmt.Errorf("llm: exactly one of anthropic or ollama must be configured")
	}

	httpClient := retryingHTTPClient(c.RetryGiveupAfter)
	requestTimeout := time.Duration(c.RequestTimeoutMS) * time.Millisecond
	if c.Anthropic != nil {
		return c.Anthropic.bind(httpClient, requestTimeout)
	}
	return c.Ollama.bind(httpClient, requestTimeout)
}

// Specs returns the provider spec fragment, spliced into every llm
// resource's spec slice (the same convention as data.InputCodec.Spec).
func Specs() []*sdk.Spec {
	return []*sdk.Spec{
		{
			Name:        "anthropic",
			Description: "use the Anthropic messages API as the backend; mutually exclusive with ollama",
			Type:        sdk.TypeObject,
			Fields: []*sdk.Spec{
				{
					Name:        "api-key",
					Description: "Anthropic API key",
					Type:        sdk.TypeString,
					Required:    true,
				},
				{
					Name:        "model",
					Description: "model id, e.g. claude-haiku-4-5",
					Type:        sdk.TypeString,
					Required:    true,
				},
				{
					Name:        "max-tokens",
					Description: "response token budget; the default fits typical extraction and scoring outputs",
					Type:        sdk.TypeInt,
					Default:     1024,
				},
				{
					Name:        "base-url",
					Description: "API base URL override; empty uses https://api.anthropic.com",
					Type:        sdk.TypeString,
					Default:     "",
				},
			},
		},
		{
			Name:        "ollama",
			Description: "use an Ollama server as the backend; mutually exclusive with anthropic",
			Type:        sdk.TypeObject,
			Fields: []*sdk.Spec{
				{
					Name:        "host",
					Description: "Ollama server base URL; empty uses http://localhost:11434",
					Type:        sdk.TypeString,
					Default:     "",
				},
				{
					Name:        "model",
					Description: "model name as known to the server, e.g. llama3.2",
					Type:        sdk.TypeString,
					Required:    true,
				},
				{
					Name:        "think",
					Description: "toggle the model's thinking phase; set false for thinking models (qwen3 etc), which otherwise stall on structured output — unset leaves the model's default",
					Type:        sdk.TypeBool,
				},
				{
					Name:        "temperature",
					Description: "sampling temperature; extraction and scoring want ~0 — unset leaves the server default (typically 0.7)",
					Type:        sdk.TypeFloat,
				},
				{
					Name:        "num-ctx",
					Description: "context window in tokens; raise when records plus schema outgrow the server default (typically 4096, which silently truncates)",
					Type:        sdk.TypeInt,
				},
				{
					Name:        "num-predict",
					Description: "hard cap on generated tokens; a ceiling against runaway thinking loops — unset leaves the server default",
					Type:        sdk.TypeInt,
				},
				{
					Name:        "seed",
					Description: "RNG seed for reproducible sampling; fix it to make an arm's output deterministic across runs — unset leaves the server default",
					Type:        sdk.TypeInt,
				},
				{
					Name:        "top-k",
					Description: "sampling restricted to the k most likely tokens — unset leaves the server default",
					Type:        sdk.TypeInt,
				},
				{
					Name:        "top-p",
					Description: "nucleus sampling cumulative-probability cutoff — unset leaves the server default",
					Type:        sdk.TypeFloat,
				},
				{
					Name:        "min-p",
					Description: "minimum token probability relative to the top token — unset leaves the server default",
					Type:        sdk.TypeFloat,
				},
				{
					Name:        "repeat-penalty",
					Description: "penalty applied to recently seen tokens to discourage repetition — unset leaves the server default",
					Type:        sdk.TypeFloat,
				},
				{
					Name:        "repeat-last-n",
					Description: "how many recent tokens the repeat penalty looks back over — unset leaves the server default",
					Type:        sdk.TypeInt,
				},
			},
		},
		{
			Name:        "request-timeout-ms",
			Description: "bound on one whole LLM call, retries included; 0 (default) leaves the call unbounded. Expiry counts as an unusable response against ask-attempts rather than failing the record immediately",
			Type:        sdk.TypeInt,
			Default:     0,
		},
		{
			Name:        "retry-giveup-after",
			Description: "number of consecutive retryable upstream responses (429/5xx) after which to give up and propagate the error; 0 (default) retries with backoff indefinitely",
			Type:        sdk.TypeInt,
			Default:     0,
		},
		{
			Name:        "ask-attempts",
			Description: "total times one record is asked of the model before an unusable response (invalid/undecodable JSON, or a request-timeout-ms expiry) is surfaced as an error; 0 (default) re-asks forever",
			Type:        sdk.TypeInt,
			Default:     0,
		},
	}
}
