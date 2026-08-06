package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/psyduck-etl/sdk/data"
)

// renderText renders a record into the text shown to the model. The sparse
// "string" accept treats the record bytes as a bare blob of text and skips
// the codec entirely (the same short-circuit ifunny's transformers use —
// "string" is a terminal-ref marker, not a chain codec). Anything else
// decodes through the codec: decoded strings pass verbatim, structured
// values marshal back to JSON text. The codec earns its keep on chained
// specs like gzip|json, where decoding is what produces readable text in
// the first place.
func renderText(codec *data.InputCodec, msg []byte) (string, error) {
	if codec.Sparse() {
		return string(msg), nil
	}
	decoded, err := codec.Decode(msg)
	if err != nil {
		return "", fmt.Errorf("decode record: %w", err)
	}
	if s, ok := decoded.(string); ok {
		return s, nil
	}
	b, err := json.Marshal(decoded)
	if err != nil {
		return "", fmt.Errorf("render record for prompt: %w", err)
	}
	return string(b), nil
}

// sendErr reports err on errs, honouring ctx cancellation. It returns
// false when the context died first, signalling the caller to stop.
func sendErr(ctx context.Context, errs chan<- error, err error) bool {
	select {
	case errs <- err:
		return true
	case <-ctx.Done():
		return false
	}
}
