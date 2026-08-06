package main

import (
	"github.com/psyduck-etl/llm/provider"
	"github.com/psyduck-etl/sdk"
	"github.com/psyduck-etl/sdk/rpc"
)

// main serves the plugin over gRPC to the psyduck host that launched this
// binary as a subprocess.
func main() { rpc.Serve(Plugin()) }

// Plugin returns the llm transformers assembled as an in-process plugin,
// which main serves to the host. Both resources plug records into an LLM
// backend — exactly one of anthropic or ollama per resource — and lean on
// schema-constrained output: llm-transform extracts structured data from a
// record, llm-filter scores a record against criteria and gates on the
// score.
func Plugin() sdk.Plugin {
	return sdk.NewInProc("llm",
		&sdk.Resource{
			Name:               "llm-transform",
			Kinds:              sdk.TRANSFORMER,
			ProvideTransformer: provideTransform,
			Spec: specs(
				&sdk.Spec{
					Name:        "prompt",
					Description: "extraction instructions for the model, e.g. what to pull out of each record",
					Type:        sdk.TypeString,
					Required:    true,
				},
				&sdk.Spec{
					Name:        "schema",
					Description: "JSON schema (type object) the extracted data must conform to, as a JSON document string",
					Type:        sdk.TypeString,
					Required:    true,
				},
				&sdk.Spec{
					Name:        "threshold",
					Description: "drop records the model scores below this 0-1 relevance; 0 (default) keeps everything",
					Type:        sdk.TypeFloat,
					Default:     0,
				},
				&sdk.Spec{
					Name:        "include-transformed",
					Description: "carry the input forward: each emitted record gains an _input field holding the text the model extracted from (added in code after extraction, never seen by the model)",
					Type:        sdk.TypeBool,
					Default:     false,
				},
				acceptSpec(),
				emitSpec(),
			),
		},
		&sdk.Resource{
			Name:               "llm-filter",
			Kinds:              sdk.TRANSFORMER,
			ProvideTransformer: provideFilter,
			Spec: specs(
				&sdk.Spec{
					Name:        "criteria",
					Description: "named yes/no questions the model answers about each record; a true answer adds the criterion's weight to the record's score",
					Type:        sdk.TypeMap,
					Required:    true,
					ElemType: &sdk.Spec{
						Type: sdk.TypeObject,
						Fields: []*sdk.Spec{
							{Name: "ask", Description: "the yes/no question posed to the model", Type: sdk.TypeString, Required: true},
							{Name: "weight", Description: "score contribution when the answer is true; may be negative for a dealbreaker", Type: sdk.TypeFloat, Required: true},
						},
					},
				},
				&sdk.Spec{
					Name:        "threshold",
					Description: "summed-weight score at or above which a record passes; 0 (default) keeps every record not dragged negative",
					Type:        sdk.TypeFloat,
					Default:     0,
				},
				acceptSpec(),
			),
		},
	)
}

// specs splices the shared provider spec fragment into a resource's own
// specs, so every llm resource carries the same backend surface.
func specs(extra ...*sdk.Spec) []*sdk.Spec {
	return append(extra, provider.Specs()...)
}

// acceptSpec is a resource's input encoding. Records decode through this
// codec and render to the text the model sees; the default "string" treats
// each record as a bare blob of text, while chains like gzip|json make
// compressed or structured records readable.
func acceptSpec() *sdk.Spec {
	return &sdk.Spec{
		Name:        "accept",
		Description: "encoding of records the resource receives, e.g. string (a bare blob of text, the default) or json (a structured object rendered to text for the model)",
		Type:        sdk.TypeString,
		Default:     "string",
	}
}

// emitSpec is llm-transform's output encoding. Extracted records are
// native maps encoded via this codec; the default "json" emits a
// structured object.
func emitSpec() *sdk.Spec {
	return &sdk.Spec{
		Name:        "emit",
		Description: "encoding of records the resource emits, e.g. json (a structured object)",
		Type:        sdk.TypeString,
		Default:     "json",
	}
}
