# Transcribe and Summarize

Local Python + Azure OpenAI pipeline that transcribes the audio from a Twitch
stream and summarizes **what was said** — singing and instruments are detected,
marked in the transcript, and excluded from the summary.

## How it works

```
audio.m4a
   │  ffmpeg: decode to 16 kHz mono
   ▼
Silero VAD (local)  ──► keep only voiced regions; drop purely-instrumental stretches
   │  (concatenate voiced audio into <=25 MB mp3 chunks + a timestamp-mapping table)
   ▼
Azure OpenAI Whisper  ──► transcribe chunks in parallel; map timestamps to absolute time
   ▼
Azure OpenAI gpt-4.1-mini  ──► tags each segment TALKING / SINGING, writes summary of talking only
   ▼
output/<name>.segments.json   raw timestamped segments
output/<name>.marked.md       full transcript, every line tagged [TALKING] / [SINGING]
output/<name>.summary.md      summary of the talking (singing excluded)
output/<name>.stats.md        audio stats + per-run cost breakdown
```

### Why the VAD step matters

Whisper hallucinates badly on music with no speech: fed instrumental audio it gets
stuck emitting a looping phrase (in testing, "What do I want to do?" appeared **37×*
times over a keytar solo) and — worse — it *drops the real speech* buried in those
windows. Silero VAD runs locally, finds the regions that actually contain a human
voice (speech or singing), and only those are sent to the cloud. This removes the
hallucination loops and recovers speech that was previously lost. It also cuts
cloud cost, since silent/instrumental audio is never uploaded or billed.

VAD detects *voice*, so it keeps both talking and singing and discards pure
instrumental. Separating singing from talking then happens in the LLM step below.

VAD runs on the **ONNX** build of Silero (faster on CPU than the PyTorch path) and is
**parallelized across CPU cores**: the decoded audio is split into `--vad-jobs` contiguous
slices (with a 1 s overlap so boundary speech isn't lost) processed in separate processes,
then the regions are merged back. The model itself is sequential, but independent slices
aren't, so this is near-linear up to ~8 workers — on the test machine it cut VAD for an
8.8-hour stream from ~3 min to well under a minute. A GPU does *not* help here: Silero is a
tiny model run on ~1M sequential 32 ms frames, so per-frame transfer overhead would dominate
— ONNX-on-CPU plus multiprocessing is the right lever. `--vad-jobs 1` disables parallelism.

### Why this design for singing vs. talking

No Azure (or any mainstream) speech service natively labels "this audio is being
*sung*." So the pipeline transcribes the voiced audio with timestamps, then asks the
LLM to classify each segment using linguistic cues (lyrics rhyme/repeat and aren't
directed at chat; talking is conversational and references the stream/chat). The
transcript keeps singing but marks it `[SINGING]`; the summary ignores it. The
classifier is biased to keep a segment as TALKING whenever it contains genuine
conversational speech, so real talk is never lost to a mixed talk-while-humming line.

**Classification is batched** (`--llm-classify-batch`, default 200 segments/call) and run in
parallel. This matters: a single call over thousands of segments degrades badly and
over-tags singing (a 9-hour test mislabeled ~84% of segments as singing in one call;
batching brought it to a realistic ~18%). Summarization is a separate pass — the model
identifies spoken content itself and ignores humming/lyrics, so it doesn't depend on the
labels. Both calls are resilient to Azure's content filter (benign talk about e.g. exhaustion
can trip the "self-harm" category — and it fires intermittently on the same text). If a
classification batch is rejected, it is **bisected recursively** to isolate the offending
segment(s); only those default to TALKING, not the whole batch (never drops speech). The
summary falls back to map-reduce so one flagged span can't sink it.

This is a heuristic. It does well on clear cases, and the classifier is told that any
segment containing a real spoken aside is TALKING. Its weak spot is **blended
segments**: when the streamer hums a melody *and* drops a brief remark to chat inside
the same ~5-10 s Whisper segment (e.g. "…do you play Pokemon Go, da dum bum, okay…"),
a single label can't be fully right. The aside text is still present in the transcript,
and the summary draws on the spoken content regardless of the label, so little is
actually lost. The raw `segments.json` is preserved so you can re-classify with
`--from-segments` after tweaking the prompt.

## Setup

Requires **ffmpeg** on `PATH` (used to decode audio). `pip install` also pulls in
PyTorch (CPU build) and onnxruntime for Silero VAD — a sizeable download, but no GPU needed.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Configuration lives in `.env` and should not be committed:

```
AZURE_OPENAI_ENDPOINT=https://<account>.openai.azure.com/
AZURE_OPENAI_KEY=...
AZURE_OPENAI_API_VERSION=2024-10-21
WHISPER_DEPLOYMENT=whisper
LLM_DEPLOYMENT=gpt-4.1-mini
LLM_REASONING_EFFORT=high
# optional flag defaults (the matching CLI flag overrides; e.g. `--mentions ""` disables mentions):
VOCAB=Jon,Rachie,keytar,Phasmophobia
VOCAB_FILE=VOCAB.txt
MENTIONS=SomebodysNameHere
```

Some Azure LLM deployments/model aliases, such as `gpt-chat-latest`, only allow the
default temperature. The script tries `temperature=0` for deterministic classification and
summaries, then automatically retries without the temperature option if Azure rejects it.

## Usage

```powershell
.\.venv\Scripts\python.exe tas.py demodata\veryshorttest.m4a --outdir output
```

Options:

| flag | default | meaning |
|------|---------|---------|
| `--outdir DIR` | `output` | where to write results |
| `--from-segments FILE` | - | reuse a specific `*.segments.json`, skip transcription (re-runs only classify+summary; saves Whisper cost) |
| `--force-transcribe` | off | ignore an existing output `<name>.segments.json` and run Whisper again |
| `--whisper-chunk-seconds N` | `1200` | max seconds of *voiced* audio per Whisper upload (mp3-compressed) |
| `--whisper-concurrency N` | `3` | parallel Whisper requests (match the deployment's per-minute request limit) |
| `--llm-concurrency N` | `4` | parallel LLM requests for classification and summaries; lower to `1` when the LLM deployment returns `429` rate limits |
| `--llm-min-interval S` | `0` | minimum seconds between LLM requests across all threads; useful when Azure rate-limits on token throughput even at low request counts |
| `--llm-reasoning-effort LEVEL` | `.env` `LLM_REASONING_EFFORT`, else `high` | LLM reasoning effort for classification and summaries: `minimal`, `low`, `medium`, `high`, or `none` to omit the parameter; CLI overrides `.env` |
| `--vad-threshold F` | `0.5` | Silero speech-probability threshold (lower = catch more speech but more music; higher = stricter) |
| `--vad-jobs N` | `min(8, cpus)` | parallel processes for VAD (splits audio across CPU cores; capped at 8; `1` disables) |
| `--summary-detail` | `flow` | `flow` = prose overview + a short curated timeline of key moments; `brief` = one concise overview+bullets; `detailed` = exhaustive per-section breakdown |
| `--section-minutes F` | `30` | time-window size used when building the flow/detailed summary |
| `--context TEXT` | – | extra background on the stream (see below) |
| `--vocab "a,b,c"` | `.env` `VOCAB` | names/terms that may be misheard (see below); fixes Whisper spellings |
| `--vocab-file FILE` | `.env` `VOCAB_FILE` | read vocab terms from a file (one per line and/or comma-separated; `#` comments and blank lines ignored); merged with `--vocab` |
| `--mentions "a,b"` | `.env` `MENTIONS`, else none | names to track; when set, adds a section listing every timestamped mention with context |
| `--whisper-language CODE` | auto | ISO-639-1 hint for Whisper (e.g. `en`) |
| `--keep-temp` | off | keep the intermediate mp3 chunks |
| `--content-filter-report FILE` | - | write JSONL diagnostics for Azure content-filter rejections, including rejected ranges, isolated segment text, and category/severity details when Azure returns them |

On reruns, if `--outdir` already contains `<name>.segments.json`, the script reuses it automatically and skips Whisper. Pass `--force-transcribe` to regenerate the transcript from audio.

### Debugging Azure content-filter rejections

If Azure rejects classification or flow-summary calls for a PG-rated stream, rerun from the
saved transcript and write a diagnostics report:

```powershell
.\.venv\Scripts\python.exe tas.py demodata\fullstream.m4a `
  --from-segments output\fullstream.segments.json `
  --outdir output `
  --content-filter-report output\fullstream.content_filter.jsonl
```

The report is newline-delimited JSON. Entries with `"isolated_segment": true` are the
segments that still failed after recursive bisection; `reason` and `error.categories` show
the Azure category/severity details when the service provides them. If only larger ranges
appear, the rejection may be caused by cumulative context or by a filter that Azure reports
without per-category metadata.

### Summary detail levels

- **`flow`** (default) — readable, for a human who didn't watch. It produces, in order:
  - a **multi-paragraph overview** that tells the story of the stream — which songs were
    learned/looped and what made them interesting (e.g. unusually hard, multi-meter, a
    special dedication), the standout conversations, the mood, and notable community
    moments. It deliberately skips generic mechanics the reader already knows.
  - **Songs this stream** — the consolidated list of songs, each tagged
    livelearn / loop / concert-grand, with a note when something was notable. Not
    very accurate.
  - **How the stream flowed** — a curated, chronological **timeline of key moments**
    with rough `HH:MM:SS` timestamps. The entry count scales with length (~one per 25
    min), so it shows the shape of the stream without a minute-by-minute play-by-play.
    It always surfaces song livelearns/loops, large gift-sub bombs, level-5 hype trains,
    raids, and free-livelearn raffles.
  - **Issues & change requests** — a chronological list of real stream/Jon problems and
    concrete stream/setup change requests: gear/software/platform failures, broken equipment,
    workflow problems, health/strain limits, and planned setup improvements. Not included:
    Normal music-making decisions, song-composition problems, extra takes, transpositions,
    instrument choices, and unrelated outside anecdotes. Related notes are consolidated
    into incident summaries with a timestamp or time range, so multi-step problems don't
    sprawl.
  - **Mentions of …** — see `--mentions` below.

  Internally it's a map-reduce: it extracts structured, timestamped notes per
  `--section-minutes` window, then distills them, so it doesn't over-compress a long
  stream the way a single pass would. If a window trips Azure's content filter, it's
  bisected so only the offending segment is dropped, not the whole window. If the final
  reduce writes an overview that is too short or starts too late in a long stream, the
  overview is rewritten from the compact song list, timeline, events, and sampled notes.
- **`detailed`** — exhaustive. Every `--section-minutes` window gets its own
  `HH:MM:SS – HH:MM:SS` heading and a full set of timestamped bullets. Use this when you
  want everything that was said, not just the highlights. (No Songs/Issues/Mentions
  sections — those are `flow`-only.)
- **`brief`** — one tight overview paragraph plus a handful of bullets, no timeline.

The flow summary is **tuned for Jon's music-request streams** (bids → livelearn → loop,
raffles, concert-grand mode, gift bombs, hype trains). That knowledge lives in the
`STREAM_PRIMER` constant near the top of `tas.py`; edit it if you adapt
the tool to a different streamer or format.

### Giving the summarizer context (names, vocabulary, tracked mentions)

- `--context "…"` — extra free-form background, appended to the built-in primer. Use it
  for anything specific to a given stream (e.g. a guest, a theme).
- `--vocab "Jon,Rachie,keytar,Phasmophobia,Elden Ring"` — comma-separated names/terms
  Whisper is likely to mis-hear. Fed to Whisper's `prompt` (biases transcription toward
  those exact spellings) **and** to the summarizer, so proper nouns come out right
  instead of as phonetic guesses. Worth listing expected song titles and regular viewers.
- `--vocab-file vocab.txt` — same terms, read from a file instead of (or in addition
  to) the command line. Handy for a long, reusable list of regulars/song titles. Terms
  may be one per line and/or comma-separated; blank lines and lines starting with `#`
  are ignored. Anything passed via `--vocab` is merged in, with case-insensitive
  duplicates dropped. Note Whisper's `prompt` is token-limited (~224 tokens), so a very
  long list will crowd out earlier terms — keep it to names that actually matter.
- `--mentions "Alinsa"` — names to track (default: none; pass names to enable). The flow
  summary adds a **Mentions of …** section with every timestamped reference and its
  context. This is found in two stages: the transcript is first scanned in Python for
  candidate words — exact/fuzzy spelling **and a phonetic (Soundex-style) skeleton**, so
  speech-to-text mis-hearings like `Elinsa`, `Elince`, or `Elincevix` (all of which Whisper
  produced for `Alinsa`/`Alinsavix` across runs) are caught even when the letters differ.
  A single LLM call then verifies the candidates by **sound + context**, keeping genuine
  references (including raffle-keyword tags) and dropping coincidences (`aliens`,
  `alliance`) and unrelated sound-alikes (`Melissa`). That two-stage design is far more
  reliable than asking the model to "list every mention," which drifts into a general
  who's-who. Because the name's spelling varies run-to-run, mention coverage is best-effort;
  the surest fix for a critical name is putting it in `--vocab` so Whisper spells it right.

`--context`/`--vocab`/`--mentions` all apply with `--from-segments` too (vocab can't fix
an already-transcribed name there, but everything else still works). Example:

```powershell
.\.venv\Scripts\python.exe tas.py demodata\fullstream.m4a --outdir output `
  --vocab "Rachie,keytar,Phasmophobia,Elden Ring,Session Horns,Kopiklani,Alinsavix" `
  --mentions "Alinsa"
```

If you find real talking is being dropped, lower `--vad-threshold` (e.g. `0.3`); if
instrumental music is leaking in and causing hallucinations, raise it (e.g. `0.6`).

### Throughput / Whisper rate limit

Whisper chunks are transcribed in parallel, but the real ceiling is the deployment's
**request-rate limit**, which scales with its capacity. Whisper `Standard` is capped at
**3 requests/min** in most regions (we set capacity to 3), so `--whisper-concurrency 3` is
the practical max; the SDK retries/backs off on `429`. Two ways to go faster on long
streams: (1) larger `--whisper-chunk-seconds` (fewer requests — an mp3 chunk holds ~25 min
of voiced audio under the 25 MB cap), or (2) request a Whisper quota increase. For a much
bigger jump you'd switch to a `gpt-4o-transcribe` model (far higher throughput) at the
cost of Whisper's segment-level timestamps, which this pipeline relies on.

For LLM classification and summarization use `--llm-concurrency` instead. LLM `429 Too Many
Requests` can mean token-per-minute or dynamic capacity limits, not only raw requests per
minute. If a new or small LLM deployment returns `429`, rerun with `--llm-concurrency 1`;
if it still happens, add `--llm-min-interval 5` or higher to pace requests. The script also
prints Azure rate-limit headers when present and performs extra backoff/retry around LLM
requests after the SDK's own retries.

## Cloud resources & cost

All Azure resources are **pay-per-use** — nothing bills just for existing:

- **Whisper** deployment (`Standard` SKU) — billed per minute of audio transcribed.
- **gpt-4.1-mini** deployment (`GlobalStandard` SKU) — billed per token.
- The Cognitive Services account (`S0`) carries no standing/hourly fee.

The `<name>.stats.md` cost estimate multiplies **actual usage** (Whisper audio minutes and
real prompt/completion token counts) by list prices. The LLM rate is looked up from a
built-in `LLM_PRICING` table (Azure **Global Standard** prices, per 1M tokens) keyed off the
model the API actually reports, so swapping `LLM_DEPLOYMENT` to another model (gpt-4o, a
GPT-5.x tier, an o-series model, etc.) prices it correctly. Dated deployment names match
their base via longest-prefix lookup; an unknown model falls back to gpt-4.1-mini's rate and
the report labels it `assumed …`. Prices were captured 2026-06-03 and are estimates only —
update `LLM_PRICING` / `WHISPER_USD_PER_MIN` if Azure's rates change.
