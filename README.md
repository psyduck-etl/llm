# psyduck llm plugin

Plugs psyduck records into LLM backends. Two transformer resources, both
built on schema-constrained ("structured") output:

- **`llm-transform`** extracts structured JSON from a record according to a
  prompt and a JSON schema you supply — e.g. turn a blob of free text into
  a typed object;
- **`llm-filter`** asks the model a set of weighted yes/no questions about
  each record, sums the weights of the true answers into a score, and gates
  on a threshold — passing the original bytes through untouched.

Exactly one backend is configured per resource: the Anthropic messages API
or an Ollama server. Both are driven through the same retrying HTTP client,
so rate-limit backoff behaves the same regardless of backend.

Built against `github.com/psyduck-etl/sdk` (see `go.mod` for the exact
version). Runs as a gRPC subprocess (`sdk/rpc`) launched by the host.

```sh
go build -o llm .
```

---

## Backend configuration (shared)

Every llm resource takes exactly one of these blocks:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `anthropic` | object | — | Use the Anthropic messages API |
| `ollama` | object | — | Use an Ollama server |

### `anthropic`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `api-key` | string | *(required)* | Anthropic API key — pass via `env.ANTHROPIC_API_KEY` |
| `model` | string | *(required)* | Model id, e.g. `claude-haiku-4-5` |
| `max-tokens` | int | `1024` | Response token budget |
| `base-url` | string | `https://api.anthropic.com` | API base URL override |

Structured output rides on forced tool use: the resource declares one tool
whose `input_schema` is the schema and pins `tool_choice` to it, so the
model can only respond with a conforming object.

### `ollama`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `http://localhost:11434` | Server base URL |
| `model` | string | *(required)* | Model name as known to the server, e.g. `llama3.2` |
| `think` | bool | *(unset)* | Toggle the model's thinking phase; set `false` for thinking models (qwen3 etc), which otherwise stall on structured output. Unset leaves the model's default |
| `temperature` | float | *(unset)* | Sampling temperature; extraction and scoring want ~0. Unset leaves the server default (typically 0.7) |
| `num-ctx` | int | *(unset)* | Context window in tokens; raise when records plus schema outgrow the server default (typically 4096, which silently truncates input) |

Structured output rides on the `format` field of `/api/chat`: the server
compiles the schema into a sampling grammar, so responses are guaranteed to
parse against it. The tuning fields are only sent when set (`think` at the
request top level, the others inside `options`), so unset knobs keep the
server's own defaults.

### Shared HTTP knobs

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `request-timeout-ms` | int | `0` | Bound on one whole LLM call, retries included; 0 = unbounded |
| `retry-giveup-after` | int | `0` | Consecutive retryable responses (429/5xx) after which to give up; 0 retries with backoff indefinitely |
| `ask-attempts` | int | `0` | Total asks per record before an unusable response (invalid or undecodable JSON) surfaces as an error; 0 re-asks forever |

---

## Resource: `llm-transform`

Kind: **transformer**. One record in, at most one record out.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | *(required)* | Extraction instructions for the model |
| `schema` | string | *(required)* | JSON schema (type `object`) the extracted data must conform to, as a JSON document string — inline heredoc or `file()` |
| `threshold` | float | `0` | Drop records the model scores below this 0-1 relevance; 0 keeps everything |
| `include-transformed` | bool | `false` | Carry the input forward: each emitted record gains an `_input` field holding the text the model extracted from |
| `accept` | string | `string` | Input encoding (see below) |
| `emit` | string | `json` | Output encoding of the extracted record; must be structured |

The model fills an envelope `{keep: 0-1, data: <your schema>}`: `keep` is
its rating of how much relevant data the record contained, `data` the
extraction. Records scoring under `threshold` are dropped, so a raised
threshold turns the transform into an extract-and-filter in one pass.

`include-transformed` is a post-extraction step in code — the model never
sees the `_input` field, so it arrives verbatim, useful for eyeballing
extraction quality or keeping provenance. It requires the extracted data to
be a JSON object (the normal case; the schema is type `object`), and it
overwrites any `_input` property the schema itself declared.

## Resource: `llm-filter`

Kind: **transformer**. A pure gate — kept records are exactly the bytes
that arrived, there is no output encoding.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `criteria` | map of `{ask, weight}` | *(required)* | Named yes/no questions the model answers about each record; a true answer adds `weight` to the score |
| `threshold` | float | `0` | Summed-weight score at or above which a record passes |
| `accept` | string | `string` | Input encoding (see below) |

Each criterion is a named object with an `ask` (the yes/no question) and a
`weight` (its score contribution when answered true; negative weights make
dealbreakers). The model answers every question as a boolean — it never
sees the weights — and the stage sums the weights of the true answers. A
record passes when that score is at or above `threshold`. The score and the
matched criteria are surfaced in debug logs (`PSYDUCK_LOG_LEVEL=debug`) so
you can see why records were dropped.

### The `accept` encoding

`accept` controls how record bytes become the text the model sees. The
default `string` treats each record as a bare blob of text. `json` decodes
and re-renders structured records; chains like `gzip|json` make compressed
records readable in the first place.

---

## Examples

### Extract structured jobs from free text (Ollama, local)

See [`extract-jobs.psy`](extract-jobs.psy) for the full runnable pipeline:
POST free-text job posts to an endpoint, extract typed fields, keep only
remote roles.

### Filter with Anthropic

```hcl
transform "llm-filter" "on-topic" {
  anthropic = {
    api-key = env.ANTHROPIC_API_KEY
    model   = "claude-haiku-4-5"
  }
  criteria = {
    on-topic = { ask = "Is the post about programming, infrastructure, or computer hardware?", weight = 10 }
    spam     = { ask = "Is the post spam or an advertisement?", weight = -100 }
  }
  threshold = 10
}
```

### Extract-and-filter in one pass

```hcl
transform "llm-transform" "products" {
  anthropic = {
    api-key = env.ANTHROPIC_API_KEY
    model   = "claude-haiku-4-5"
  }
  prompt    = "Extract every product mentioned in the post with its sentiment"
  schema    = file("product-schema.json")
  threshold = 0.4   # drop posts with no product mentions at all
}
```
