package provider

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

// funcClient adapts a function to the Client interface for driving
// CompleteJSON without a real backend.
type funcClient func(ctx context.Context, system, user string, schema json.RawMessage) (json.RawMessage, error)

func (f funcClient) Complete(ctx context.Context, system, user string, schema json.RawMessage) (json.RawMessage, error) {
	return f(ctx, system, user, schema)
}

type completePayload struct {
	Keep float64 `json:"keep"`
}

func TestRequestTimedOutDistinguishesFromParentCancel(t *testing.T) {
	// Our own deadline elapsing: requestTimedOut must report true.
	reqCtx, cancel := withRequestTimeout(context.Background(), time.Millisecond)
	defer cancel()
	<-reqCtx.Done()
	if !requestTimedOut(reqCtx) {
		t.Error("want true when the derived deadline itself elapsed")
	}

	// The parent ending first (pipeline shutdown) propagates the same Done
	// signal but for a different reason — must report false so CompleteJSON
	// doesn't waste a retry on a call that's shutting down anyway.
	parent, cancelParent := context.WithCancel(context.Background())
	reqCtx2, cancel2 := withRequestTimeout(parent, time.Hour)
	defer cancel2()
	cancelParent()
	<-reqCtx2.Done()
	if requestTimedOut(reqCtx2) {
		t.Error("want false when the parent was cancelled, not our deadline")
	}

	// request-timeout-ms unset (timeout <= 0): withRequestTimeout must be a
	// no-op, returning ctx itself rather than a derived deadline.
	unbounded, cancel3 := withRequestTimeout(context.Background(), 0)
	defer cancel3()
	if unbounded.Done() != nil {
		t.Error("want ctx returned unmodified when timeout <= 0")
	}
}

func TestCompleteJSONFirstTry(t *testing.T) {
	calls := 0
	client := funcClient(func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
		calls++
		return json.RawMessage(`{"keep": 0.7}`), nil
	})

	out := completePayload{}
	if err := CompleteJSON(context.Background(), client, 3, "s", "u", nil, &out, nil); err != nil {
		t.Fatal(err)
	}
	if calls != 1 || out.Keep != 0.7 {
		t.Fatalf("want 1 call keep 0.7, got %d calls keep %f", calls, out.Keep)
	}
}

func TestCompleteJSONReasksBadContent(t *testing.T) {
	calls := 0
	client := funcClient(func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
		calls++
		if calls < 3 {
			return nil, fmt.Errorf("truncated: %w", ErrBadContent)
		}
		return json.RawMessage(`{"keep": 1}`), nil
	})

	out := completePayload{}
	if err := CompleteJSON(context.Background(), client, 3, "s", "u", nil, &out, nil); err != nil {
		t.Fatal(err)
	}
	if calls != 3 || out.Keep != 1 {
		t.Fatalf("want 3 calls keep 1, got %d calls keep %f", calls, out.Keep)
	}
}

func TestCompleteJSONReasksUndecodable(t *testing.T) {
	calls := 0
	client := funcClient(func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
		calls++
		if calls == 1 {
			// valid JSON, wrong shape for the payload type
			return json.RawMessage(`{"keep": "very"}`), nil
		}
		return json.RawMessage(`{"keep": 0.4}`), nil
	})

	out := completePayload{}
	if err := CompleteJSON(context.Background(), client, 2, "s", "u", nil, &out, nil); err != nil {
		t.Fatal(err)
	}
	if calls != 2 || out.Keep != 0.4 {
		t.Fatalf("want 2 calls keep 0.4, got %d calls keep %f", calls, out.Keep)
	}
}

func TestCompleteJSONGivesUp(t *testing.T) {
	calls := 0
	client := funcClient(func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
		calls++
		return nil, fmt.Errorf("always truncated: %w", ErrBadContent)
	})

	err := CompleteJSON(context.Background(), client, 2, "s", "u", nil, &completePayload{}, nil)
	if err == nil {
		t.Fatal("want error after exhausting attempts")
	}
	if calls != 2 {
		t.Fatalf("want 2 calls, got %d", calls)
	}
	if !errors.Is(err, ErrBadContent) || !strings.Contains(err.Error(), "2 attempts") {
		t.Fatalf("want attempt-counted ErrBadContent, got: %v", err)
	}
}

func TestCompleteJSONOtherErrorImmediate(t *testing.T) {
	calls := 0
	boom := errors.New("connection refused")
	client := funcClient(func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
		calls++
		return nil, boom
	})

	err := CompleteJSON(context.Background(), client, 5, "s", "u", nil, &completePayload{}, nil)
	if !errors.Is(err, boom) {
		t.Fatalf("want transport error surfaced, got: %v", err)
	}
	if calls != 1 {
		t.Fatalf("want 1 call, got %d", calls)
	}
}

func TestCompleteJSONUnboundedStopsOnCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	client := funcClient(func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
		calls++
		if calls == 10 {
			cancel()
		}
		return nil, fmt.Errorf("bad: %w", ErrBadContent)
	})

	err := CompleteJSON(ctx, client, 0, "s", "u", nil, &completePayload{}, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got: %v", err)
	}
	if calls != 10 {
		t.Fatalf("want 10 calls before cancel observed, got %d", calls)
	}
}

func TestCompleteJSONReasksOnValidateFailure(t *testing.T) {
	calls := 0
	client := funcClient(func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
		calls++
		if calls == 1 {
			return json.RawMessage(`{"keep": 0.5}`), nil
		}
		return json.RawMessage(`{"keep": 0.9}`), nil
	})

	out := completePayload{}
	validate := func(v any) error {
		p := v.(*completePayload)
		if p.Keep < 0.7 {
			return fmt.Errorf("keep %f too low", p.Keep)
		}
		return nil
	}
	if err := CompleteJSON(context.Background(), client, 2, "s", "u", nil, &out, validate); err != nil {
		t.Fatal(err)
	}
	if calls != 2 || out.Keep != 0.9 {
		t.Fatalf("want 2 calls keep 0.9, got %d calls keep %f", calls, out.Keep)
	}
}

func TestCompleteJSONGivesUpOnValidateFailure(t *testing.T) {
	calls := 0
	client := funcClient(func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
		calls++
		return json.RawMessage(`{"keep": 0.1}`), nil
	})

	rejectAlways := func(any) error { return errors.New("never grounded") }
	err := CompleteJSON(context.Background(), client, 2, "s", "u", nil, &completePayload{}, rejectAlways)
	if err == nil {
		t.Fatal("want error after exhausting attempts")
	}
	if calls != 2 {
		t.Fatalf("want 2 calls, got %d", calls)
	}
	if !errors.Is(err, ErrBadContent) {
		t.Fatalf("want ErrBadContent, got: %v", err)
	}
}

func TestCompleteJSONFailedAttemptDoesNotLeak(t *testing.T) {
	calls := 0
	client := funcClient(func(context.Context, string, string, json.RawMessage) (json.RawMessage, error) {
		calls++
		if calls == 1 {
			// decodes reason, then fails on keep: a partial fill that must
			// not survive into the committed result
			return json.RawMessage(`{"reason": "stale", "keep": "very"}`), nil
		}
		return json.RawMessage(`{"keep": 0.9}`), nil
	})

	out := struct {
		Reason string  `json:"reason"`
		Keep   float64 `json:"keep"`
	}{}
	if err := CompleteJSON(context.Background(), client, 2, "s", "u", nil, &out, nil); err != nil {
		t.Fatal(err)
	}
	if out.Reason != "" || out.Keep != 0.9 {
		t.Fatalf("stale fields leaked from failed attempt: %+v", out)
	}
}
