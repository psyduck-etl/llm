package main

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/psyduck-etl/sdk"
	"github.com/psyduck-etl/sdk/data"
)

func testFilterConfig() *filterConfig {
	config := &filterConfig{
		InputCodec: data.InputCodec{Accept: "string"},
		Criteria: map[string]criterion{
			"ducks": {Ask: "is the post about ducks?", Weight: 30},
			"geese": {Ask: "is the post about geese?", Weight: 10},
			"arms":  {Ask: "is the post about weapons?", Weight: -100},
		},
		Threshold: 20,
	}
	// One ask per record: the default (0) re-asks unusable content forever,
	// which would hang tests that simulate persistent content failures.
	config.AskAttempts = 1
	if err := config.InputCodec.Bind(); err != nil {
		panic(err)
	}
	return config
}

func TestFilterGate(t *testing.T) {
	for name, test := range map[string]struct {
		answers string
		want    int
	}{
		"clears threshold":       {answers: `{"ducks": true, "geese": false, "arms": false}`, want: 1},
		"at threshold passes":    {answers: `{"ducks": false, "geese": true, "arms": false}`, want: 0}, // 10 < 20
		"exactly threshold":      {answers: `{"ducks": true, "geese": false, "arms": false}`, want: 1}, // 30 >= 20
		"dealbreaker drags down": {answers: `{"ducks": true, "geese": true, "arms": true}`, want: 0},   // 30+10-100
		"nothing matches":        {answers: `{"ducks": false, "geese": false, "arms": false}`, want: 0},
	} {
		t.Run(name, func(t *testing.T) {
			client := fakeClient(func(_, _ string, _ json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(test.answers), nil
			})

			msg := []byte("a post about ducks")
			out, errs := runStage(t, makeFilter(client, testFilterConfig()), msg)
			if len(errs) != 0 {
				t.Fatalf("errs = %v", errs)
			}
			if len(out) != test.want {
				t.Fatalf("emitted %d records, want %d", len(out), test.want)
			}
			if test.want == 1 && string(out[0]) != string(msg) {
				t.Errorf("kept record should be the original bytes untouched, got %q", out[0])
			}
		})
	}
}

func TestFilterPrompt(t *testing.T) {
	var gotSystem, gotUser string
	var gotSchema json.RawMessage
	client := fakeClient(func(system, user string, schema json.RawMessage) (json.RawMessage, error) {
		gotSystem, gotUser, gotSchema = system, user, schema
		return json.RawMessage(`{"ducks": true, "geese": false, "arms": false}`), nil
	})

	_, errs := runStage(t, makeFilter(client, testFilterConfig()), []byte("quack"))
	if len(errs) != 0 {
		t.Fatalf("errs = %v", errs)
	}
	// The system prompt enumerates every criterion's ask, but never a weight.
	if !strings.Contains(gotSystem, "is the post about ducks?") {
		t.Errorf("system prompt should carry the asks, got %q", gotSystem)
	}
	if strings.Contains(gotSystem, "-100") || strings.Contains(gotSystem, "30") {
		t.Errorf("system prompt must not leak weights, got %q", gotSystem)
	}
	if gotUser != "quack" {
		t.Errorf("user text = %q", gotUser)
	}
	// The schema is a boolean property per criterion, keyed by name.
	if !strings.Contains(string(gotSchema), `"ducks"`) || !strings.Contains(string(gotSchema), `"boolean"`) {
		t.Errorf("schema should hold a boolean property per criterion, got %s", gotSchema)
	}
}

func TestFilterClientError(t *testing.T) {
	client := fakeClient(func(_, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return nil, errors.New("upstream exploded")
	})

	out, errs := runStage(t, makeFilter(client, testFilterConfig()), []byte("one"), []byte("two"))
	if len(out) != 0 {
		t.Errorf("emitted %d records, want 0", len(out))
	}
	if len(errs) != 2 {
		t.Fatalf("errs = %v, want 2", errs)
	}
}

func TestProvideFilterValidates(t *testing.T) {
	crit := `"criteria": {"ducks": {"ask": "about ducks?", "weight": 1}}`
	for name, block := range map[string]string{
		"missing criteria": `{"ollama": {"model": "m"}}`,
		"empty criteria":   `{"criteria": {}, "ollama": {"model": "m"}}`,
		"criterion no ask": `{"criteria": {"ducks": {"weight": 1}}, "ollama": {"model": "m"}}`,
		"no backend":       `{` + crit + `}`,
		"both backends":    `{` + crit + `, "ollama": {"model": "m"}, "anthropic": {"api-key": "k", "model": "m"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			parse := func(dst any) error { return sdk.DecodeJSONTagged([]byte(block), dst) }
			if _, err := provideFilter(t.Context(), parse); err == nil {
				t.Error("expected config error, got nil")
			}
		})
	}
}

func TestProvideFilterBinds(t *testing.T) {
	block := `{"criteria": {"ducks": {"ask": "about ducks?", "weight": 30}}, "ollama": {"model": "m"}}`
	parse := func(dst any) error { return sdk.DecodeJSONTagged([]byte(block), dst) }
	if _, err := provideFilter(t.Context(), parse); err != nil {
		t.Fatalf("provide: %v", err)
	}
}
