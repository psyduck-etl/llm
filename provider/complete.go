package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"time"
)

// ErrBadContent marks a Complete response whose content is unusable — the
// call itself succeeded but the model's output can't be consumed (invalid
// JSON from ollama, a truncated/tool-less anthropic response) — or never
// arrived at all because the request-timeout-ms budget expired, most often
// a runaway generation that never emits a stop token. Errors wrapping it
// are worth re-asking: generation is sampled, so a fresh ask can succeed —
// or at least not hang — where the last one didn't. Other transport and
// HTTP-status errors are deliberately not marked: the shared retrying
// client already owns retryable statuses, and failures like a refused
// connection are unlikely to be fixed by asking again.
var ErrBadContent = errors.New("unusable response content")

// withRequestTimeout derives a context bounded by timeout, a request's own
// deadline racing the parent ctx exactly like any other context.WithTimeout
// child. timeout <= 0 (request-timeout-ms unset) returns ctx unmodified —
// a no-op cancel — so the call stays unbounded. Backends use the returned
// context to build their HTTP request, then call requestTimedOut on it
// after a failure to tell "our deadline won the race" apart from any other
// reason the call didn't come back.
func withRequestTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, timeout)
}

// requestTimedOut reports whether reqCtx (as returned by withRequestTimeout)
// is done because its own deadline elapsed, as opposed to the parent ctx
// ending some other way (pipeline shutdown cancels the parent, which
// propagates the same Done signal but reports context.Canceled instead).
// Backends fold a true result into ErrBadContent so CompleteJSON's
// ask-attempts loop retries the record instead of failing it outright.
func requestTimedOut(reqCtx context.Context) bool {
	return errors.Is(reqCtx.Err(), context.DeadlineExceeded)
}

// CompleteJSON drives client.Complete and decodes the response into out,
// re-asking while the response is unusable: a content failure from the
// backend (ErrBadContent), JSON that doesn't decode into out, or — when
// validate is non-nil — a decoded response validate rejects. validate is
// folded into ErrBadContent exactly like a decode failure: the JSON was
// well-formed but failed a check the schema can't express (e.g. a
// fabricated contact value not present in the source text), and re-asking
// is worth it for the same sampling reason a decode retry is. validate
// receives a pointer of out's type holding the freshly decoded attempt, and
// may be nil to skip this check. attempts bounds the total number of asks;
// 0 re-asks forever, until the response is usable or ctx ends — mirroring
// retry-giveup-after's 0-means-unbounded. Any other error — transport, HTTP
// status, cancellation — returns immediately, since the shared HTTP client
// has already retried what's retryable. On giving up it returns the last
// content error wrapped with the attempt count.
//
// out must be a non-nil pointer. It is written only on success: each
// attempt decodes into a fresh value of out's type and commits only when
// the decode (and validate) succeed, so a half-decoded failed attempt
// never leaks fields into the result.
func CompleteJSON(ctx context.Context, client Client, attempts uint, system, user string, schema json.RawMessage, out any, validate func(any) error) error {
	var last error
	for i := uint(0); attempts == 0 || i < attempts; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}

		raw, err := client.Complete(ctx, system, user, schema)
		if err != nil {
			if !errors.Is(err, ErrBadContent) {
				return err
			}
			last = err
			continue
		}

		fresh := reflect.New(reflect.TypeOf(out).Elem())
		if err := json.Unmarshal(raw, fresh.Interface()); err != nil {
			last = fmt.Errorf("decode model response: %w: %w", err, ErrBadContent)
			continue
		}
		if validate != nil {
			if err := validate(fresh.Interface()); err != nil {
				last = fmt.Errorf("validate model response: %w: %w", err, ErrBadContent)
				continue
			}
		}
		reflect.ValueOf(out).Elem().Set(fresh.Elem())
		return nil
	}
	return fmt.Errorf("llm: no usable response after %d attempts: %w", attempts, last)
}
