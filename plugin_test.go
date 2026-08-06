package main

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/psyduck-etl/llm/provider"
	"github.com/psyduck-etl/sdk"
	"github.com/psyduck-etl/sdk/data"
)

// TestMain installs the stdlib codec chain, mirroring what rpc.Serve does
// in the real plugin process, so accept/emit specs resolve in tests.
func TestMain(m *testing.M) {
	sdk.RegisterCodecs(data.Codec)
	os.Exit(m.Run())
}

// fakeClient adapts a function into a provider.Client so transformer tests
// can script model behavior and capture what the stage sent.
type fakeClient func(system, user string, schema json.RawMessage) (json.RawMessage, error)

func (f fakeClient) Complete(_ context.Context, system, user string, schema json.RawMessage) (json.RawMessage, error) {
	return f(system, user, schema)
}

var _ provider.Client = fakeClient(nil)

// runStage drives a transformer over msgs and collects everything it emits
// on out and errs, returning once the stage closes out.
func runStage(t *testing.T, stage sdk.Transformer, msgs ...[]byte) (got [][]byte, errors []error) {
	t.Helper()

	in := make(chan []byte, len(msgs))
	for _, msg := range msgs {
		in <- msg
	}
	close(in)

	out := make(chan []byte)
	errs := make(chan error, len(msgs)*2)
	done := make(chan struct{})
	go func() {
		defer close(done)
		stage(context.Background(), in, out, errs)
	}()

	for msg := range out {
		got = append(got, msg)
	}
	<-done
	close(errs)
	for err := range errs {
		errors = append(errors, err)
	}
	return got, errors
}

func TestPluginResources(t *testing.T) {
	plugin := Plugin()
	if plugin.Name() != "llm" {
		t.Errorf("plugin name = %q", plugin.Name())
	}

	byName := map[string]sdk.ResourceDescriptor{}
	for _, resource := range plugin.Resources() {
		byName[resource.Name] = resource
	}
	for _, name := range []string{"llm-transform", "llm-filter"} {
		resource, ok := byName[name]
		if !ok {
			t.Errorf("missing resource %q", name)
			continue
		}
		if resource.Kinds != sdk.TRANSFORMER {
			t.Errorf("%s kinds = %d, want TRANSFORMER", name, resource.Kinds)
		}
	}
}
