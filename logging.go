package main

import (
	"os"

	"github.com/hashicorp/go-hclog"
)

// logger is the plugin-wide logger. Its lines are written to stderr as hclog
// JSON, which the psyduck host captures and re-emits through its own
// plugin.<name> logger at the embedded level. The host forwarder drops
// anything below its configured level (default Warn), so to see Info lines in
// the host logs the job must run with PSYDUCK_LOG_LEVEL=debug (or trace).
var logger = hclog.New(&hclog.LoggerOptions{
	Name:       "llm",
	Level:      logLevel(),
	Output:     os.Stderr,
	JSONFormat: true,
})

// logLevel resolves the plugin log level from PSYDUCK_LOG_LEVEL, mirroring the
// host's own mapping so plugin and host agree on what to keep. Default is Info.
func logLevel() hclog.Level {
	switch os.Getenv("PSYDUCK_LOG_LEVEL") {
	case "trace":
		return hclog.Trace
	case "debug":
		return hclog.Debug
	case "warn":
		return hclog.Warn
	case "error", "fatal", "panic":
		return hclog.Error
	default:
		return hclog.Info
	}
}
