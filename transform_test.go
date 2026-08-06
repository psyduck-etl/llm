package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/psyduck-etl/sdk"
	"github.com/psyduck-etl/sdk/data"
)

const testUserSchema = `{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`

func testTransformConfig() *transformConfig {
	config := &transformConfig{
		InputCodec:  data.InputCodec{Accept: "string"},
		OutputCodec: data.OutputCodec{Emit: "json"},
		Prompt:      "pull the name out",
		Schema:      testUserSchema,
	}
	// One ask per record: the default (0) re-asks unusable content forever,
	// which would hang tests that simulate persistent content failures.
	config.AskAttempts = 1
	if err := config.InputCodec.Bind(); err != nil {
		panic(err)
	}
	if err := config.OutputCodec.Bind(); err != nil {
		panic(err)
	}
	return config
}

func TestTransformExtracts(t *testing.T) {
	var gotSystem, gotUser string
	var gotSchema json.RawMessage
	client := fakeClient(func(system, user string, schema json.RawMessage) (json.RawMessage, error) {
		gotSystem, gotUser, gotSchema = system, user, schema
		return json.RawMessage(`{"keep": 0.9, "data": {"name": "psyduck"}}`), nil
	})

	out, errs := runStage(t, makeTransform(client, testTransformConfig()), []byte("the name is psyduck"))
	if len(errs) != 0 {
		t.Fatalf("errs = %v", errs)
	}
	if len(out) != 1 {
		t.Fatalf("emitted %d records, want 1", len(out))
	}

	record := map[string]any{}
	if err := json.Unmarshal(out[0], &record); err != nil {
		t.Fatalf("emitted record is not json: %v", err)
	}
	if record["name"] != "psyduck" {
		t.Errorf("record = %v", record)
	}

	if !strings.Contains(gotSystem, "pull the name out") {
		t.Errorf("system prompt should carry the config prompt, got %q", gotSystem)
	}
	if gotUser != "the name is psyduck" {
		t.Errorf("user text = %q", gotUser)
	}
	if !json.Valid(gotSchema) {
		t.Fatalf("envelope schema is not valid json: %s", gotSchema)
	}
	if !strings.Contains(string(gotSchema), `"keep"`) || !strings.Contains(string(gotSchema), testUserSchema) {
		t.Errorf("envelope schema should wrap the user schema with keep, got %s", gotSchema)
	}
}

func TestTransformThreshold(t *testing.T) {
	for _, test := range []struct {
		keep      float64
		threshold float64
		want      int
	}{
		{keep: 0.2, threshold: 0.5, want: 0},
		{keep: 0.5, threshold: 0.5, want: 1},
		{keep: 0.0, threshold: 0.0, want: 1}, // default threshold keeps everything
	} {
		t.Run(fmt.Sprintf("keep %.1f threshold %.1f", test.keep, test.threshold), func(t *testing.T) {
			client := fakeClient(func(_, _ string, _ json.RawMessage) (json.RawMessage, error) {
				return json.RawMessage(fmt.Sprintf(`{"keep": %f, "data": {"name": "x"}}`, test.keep)), nil
			})
			config := testTransformConfig()
			config.Threshold = test.threshold

			out, errs := runStage(t, makeTransform(client, config), []byte("msg"))
			if len(errs) != 0 {
				t.Fatalf("errs = %v", errs)
			}
			if len(out) != test.want {
				t.Errorf("emitted %d records, want %d", len(out), test.want)
			}
		})
	}
}

func TestTransformIncludeTransformed(t *testing.T) {
	client := fakeClient(func(_, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"keep": 1, "data": {"name": "psyduck"}}`), nil
	})
	config := testTransformConfig()
	config.IncludeTransformed = true

	out, errs := runStage(t, makeTransform(client, config), []byte("the name is psyduck"))
	if len(errs) != 0 {
		t.Fatalf("errs = %v", errs)
	}
	if len(out) != 1 {
		t.Fatalf("emitted %d records, want 1", len(out))
	}

	record := map[string]any{}
	if err := json.Unmarshal(out[0], &record); err != nil {
		t.Fatalf("emitted record is not json: %v", err)
	}
	if record["name"] != "psyduck" {
		t.Errorf("extraction should survive alongside _input, got %v", record)
	}
	if record["_input"] != "the name is psyduck" {
		t.Errorf("_input = %v, want the model's input text", record["_input"])
	}
}

func TestTransformIncludeTransformedOffByDefault(t *testing.T) {
	client := fakeClient(func(_, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"keep": 1, "data": {"name": "psyduck"}}`), nil
	})

	out, errs := runStage(t, makeTransform(client, testTransformConfig()), []byte("msg"))
	if len(errs) != 0 {
		t.Fatalf("errs = %v", errs)
	}
	record := map[string]any{}
	if err := json.Unmarshal(out[0], &record); err != nil {
		t.Fatalf("emitted record is not json: %v", err)
	}
	if _, ok := record["_input"]; ok {
		t.Errorf("_input should be absent by default, got %v", record["_input"])
	}
}

func TestTransformIncludeTransformedNonObject(t *testing.T) {
	// The schema is documented as type object, but nothing stops a model
	// from being asked for a bare array — there is nowhere to hang _input
	// on one, so the record surfaces as an error instead.
	client := fakeClient(func(_, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"keep": 1, "data": ["not", "an", "object"]}`), nil
	})
	config := testTransformConfig()
	config.IncludeTransformed = true

	out, errs := runStage(t, makeTransform(client, config), []byte("msg"))
	if len(out) != 0 {
		t.Errorf("emitted %d records, want 0", len(out))
	}
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "include-transformed") {
		t.Fatalf("errs = %v, want one include-transformed error", errs)
	}
}

func TestTransformClientError(t *testing.T) {
	client := fakeClient(func(_, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return nil, errors.New("upstream exploded")
	})

	out, errs := runStage(t, makeTransform(client, testTransformConfig()), []byte("one"), []byte("two"))
	if len(out) != 0 {
		t.Errorf("emitted %d records, want 0", len(out))
	}
	// The stage reports each failure and keeps consuming.
	if len(errs) != 2 {
		t.Fatalf("errs = %v, want 2", errs)
	}
}

func TestTransformMalformedResponse(t *testing.T) {
	client := fakeClient(func(_, _ string, _ json.RawMessage) (json.RawMessage, error) {
		return json.RawMessage(`{"keep": "not a number"}`), nil
	})

	out, errs := runStage(t, makeTransform(client, testTransformConfig()), []byte("msg"))
	if len(out) != 0 {
		t.Errorf("emitted %d records, want 0", len(out))
	}
	if len(errs) != 1 {
		t.Errorf("errs = %v, want 1", errs)
	}
}

func TestTransformAcceptJSON(t *testing.T) {
	var gotUser string
	client := fakeClient(func(_, user string, _ json.RawMessage) (json.RawMessage, error) {
		gotUser = user
		return json.RawMessage(`{"keep": 1, "data": {"name": "x"}}`), nil
	})
	config := testTransformConfig()
	config.InputCodec = data.InputCodec{Accept: "json"}
	if err := config.InputCodec.Bind(); err != nil {
		t.Fatal(err)
	}

	_, errs := runStage(t, makeTransform(client, config), []byte(`{"body": "text", "id": 7}`))
	if len(errs) != 0 {
		t.Fatalf("errs = %v", errs)
	}

	// Structured records render back to JSON text for the model.
	rendered := map[string]any{}
	if err := json.Unmarshal([]byte(gotUser), &rendered); err != nil {
		t.Fatalf("user text should be json, got %q: %v", gotUser, err)
	}
	if rendered["body"] != "text" {
		t.Errorf("rendered = %v", rendered)
	}
}

func TestProvideTransformValidates(t *testing.T) {
	for name, block := range map[string]string{
		"missing prompt":  `{"schema": "{}", "ollama": {"model": "m"}}`,
		"missing schema":  `{"prompt": "p", "ollama": {"model": "m"}}`,
		"invalid schema":  `{"prompt": "p", "schema": "{nope", "ollama": {"model": "m"}}`,
		"no backend":      `{"prompt": "p", "schema": "{}"}`,
		"both backends":   `{"prompt": "p", "schema": "{}", "ollama": {"model": "m"}, "anthropic": {"api-key": "k", "model": "m"}}`,
		"backend invalid": `{"prompt": "p", "schema": "{}", "ollama": {}}`,
	} {
		t.Run(name, func(t *testing.T) {
			parse := func(dst any) error { return sdk.DecodeJSONTagged([]byte(block), dst) }
			if _, err := provideTransform(t.Context(), parse); err == nil {
				t.Error("expected config error, got nil")
			}
		})
	}
}

func TestProvideTransformBinds(t *testing.T) {
	block := fmt.Sprintf(`{"prompt": "p", "schema": %q, "ollama": {"model": "m"}}`, testUserSchema)
	parse := func(dst any) error { return sdk.DecodeJSONTagged([]byte(block), dst) }
	if _, err := provideTransform(t.Context(), parse); err != nil {
		t.Fatalf("provide: %v", err)
	}
}
