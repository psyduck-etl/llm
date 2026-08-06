package provider

import (
	"context"
	"io"
	"math/rand"
	"net/http"
	"time"
)

const (
	// backoffBase is the first backoff interval after a retryable
	// response; each subsequent consecutive one doubles it.
	backoffBase = 2 * time.Second

	// backoffMaxShift caps the doubling. The ceiling is
	// backoffBase << backoffMaxShift = 2s << 6 = 128s; past that the
	// interval holds at the ceiling. LLM rate limits and overload
	// windows clear in seconds-to-minutes, so a ~2 minute ceiling keeps
	// retries responsive without hammering a struggling upstream.
	backoffMaxShift = 6

	// backoffJitter scales each sleep by a uniform random factor in
	// [1-backoffJitter, 1+backoffJitter] — i.e. ±10% — so concurrent
	// stages that hit a limit in lockstep drift apart instead of
	// re-colliding on every retry.
	backoffJitter = 0.10
)

// retryTransport wraps an http.RoundTripper, retrying retryable responses
// (429 Too Many Requests, and 5xx — Anthropic signals overload with 529,
// Ollama surfaces transient failures as 500s) with jittered exponential
// backoff. Success, other status codes, and the transport's own errors pass
// straight through untouched.
//
// giveUpAfter bounds the retry: after that many consecutive retryable
// tries the transport stops and returns the last response so the caller
// sees the upstream error. 0 keeps the retry unbounded.
//
// RoundTrip holds no mutable state, so a single retryTransport is safe to
// share across goroutines.
type retryTransport struct {
	base        http.RoundTripper
	giveUpAfter uint
}

// backoffSleep is the sleep the retry loop uses between attempts. It is a
// package var so tests can swap in a no-op that records the requested
// intervals instead of blocking on real backoffs.
var backoffSleep = sleepCtx

// retryable reports whether an upstream status is worth backing off and
// retrying: rate limiting or a server-side failure.
func retryable(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500
}

func (t *retryTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	for attempt := 0; ; attempt++ {
		// Rewind the body on every retry. http.NewRequestWithContext
		// populates GetBody for the in-memory bodies our JSON POSTs use,
		// so a consumed body can be replayed.
		if attempt > 0 && req.GetBody != nil {
			body, err := req.GetBody()
			if err != nil {
				return nil, err
			}
			req.Body = body
		}

		resp, err := t.base.RoundTrip(req)
		if err != nil {
			return nil, err
		}
		if !retryable(resp.StatusCode) {
			return resp, nil
		}

		// attempt+1 tries have now come back retryable. If a give-up
		// bound is set and we've reached it, hand the response back to
		// the caller (body intact) instead of retrying further.
		if t.giveUpAfter != 0 && uint(attempt+1) >= t.giveUpAfter {
			return resp, nil
		}

		// Drain and close so the connection returns to the pool before
		// we sleep, then back off and retry the same request.
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()

		if err := backoffSleep(req.Context(), backoffFor(attempt)); err != nil {
			return nil, err
		}
	}
}

// backoffFor returns the (jittered) sleep before the attempt-th retry.
// attempt 0 yields backoffBase, doubling each step until the shift
// saturates at backoffMaxShift, after which every attempt sits at the
// ceiling.
func backoffFor(attempt int) time.Duration {
	shift := min(attempt, backoffMaxShift)
	d := backoffBase << shift
	factor := 1 + (rand.Float64()*2-1)*backoffJitter
	return time.Duration(float64(d) * factor)
}

// sleepCtx sleeps for d, returning early with ctx.Err() if ctx is
// cancelled first so a cancelled stage aborts an in-flight backoff instead
// of waiting out the full interval.
func sleepCtx(ctx context.Context, d time.Duration) error {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// retryingHTTPClient builds the shared backend HTTP client: the default
// transport wrapped in a retryTransport. It carries no timeout of its own —
// request-timeout-ms is enforced per call by each backend deriving a
// context.WithTimeout from the request's ctx (see ollama.go / anthropic.go),
// not by the client, so the deadline can be told apart from other
// cancellation and folded into the ask-attempts retry loop.
func retryingHTTPClient(giveUpAfter uint) *http.Client {
	return &http.Client{
		Transport: &retryTransport{base: http.DefaultTransport, giveUpAfter: giveUpAfter},
	}
}
