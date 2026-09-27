# Streaming voice pipeline — what changed and why

## The problem this solves

Your old turn looked like:

```
STT → fast_reply()  [BLOCKS until full LLM response]  → play_text_to_speech_stream()  [then starts]
```

Even though `voice_service.py` already streamed *TTS synthesis* sentence-by-sentence,
it could only start once `fast_reply()` handed back the complete string. For a
40-word reply, that's commonly 1–3 seconds of silence before the user hears
anything — the single biggest lever on perceived latency, bigger than swapping
which LLM you use.

## The fix

Three new/changed files, each with one job:

| File | Job |
|---|---|
| `streaming_llm.py` | Streams tokens from the LLM, yields complete **sentences** as soon as they're ready |
| `voice_service.py` | Added `speak_sentences_live_async()` — consumes sentences as they arrive instead of a finished string. Old `play_text_to_speech_stream()` still works unchanged, for tool/crisis replies where you already have the full text. |
| `orchestrator.py` | The only file that imports from both — wires LLM streaming straight into TTS streaming, then saves the finished turn to `ThreadStore` once, at the end |

Result: sentence 1 starts synthesizing and playing while the LLM is still
generating sentence 4. Generation and speech now overlap, not just synthesis
and playback.

## How to wire it into `voice_engine.py`

Find where your main loop currently does something like:

```python
reply = fast_reply(llm, user_text, language, emotion, recent_history)
executor.submit(play_text_to_speech_stream, reply, language)
store.append_message(thread_id, "user", user_text)
store.append_message(thread_id, "assistant", reply)
```

Replace it with:

```python
from orchestrator import speak_streaming

reply = asyncio.run(
    speak_streaming(llm, user_text, language, emotion, thread_id, store)
)
```

(If your main loop is already `async def`, just `await speak_streaming(...)`
directly instead of wrapping in `asyncio.run`.)

**Keep the old path for tool/crisis replies.** `_needs_tool(user_text)` still
routes to the full `AgentExecutor` (unmodified), and its output should still
go through `play_text_to_speech_stream()`, not the new streaming path — those
replies are short and tool-driven (e.g. reading out a helpline number
correctly matters more than shaving 300ms).

## Other latency levers, roughly in order of impact

1. **This change** (LLM/TTS overlap) — biggest single win, done.
2. **STT**: if you're calling a cloud Whisper API, that's another network
   round-trip before the LLM even starts. `faster-whisper` running locally
   (int8 quantization) is fast enough on a CPU for short utterances and
   removes that hop entirely. Not included in this bundle since your
   `voice_engine.py` wasn't shared, but worth doing next.
2. **edge_tts is a cloud call per sentence.** Fine on good wifi, but every
   sentence round-trips to Microsoft's endpoint. If you want to remove that
   dependency (and get more *consistent* latency, not just lower average),
   look at swapping `_synthesize_to_file()` in `voice_service.py` for a local
   engine like [Piper](https://github.com/rhasspy/piper) — CPU-only, ONNX,
   typically 100–300ms per sentence, decent voice quality. Because the
   synthesis step is isolated in one function, this is a contained swap.
3. **Tool-trigger keywords in `generic_agent.py`** — double check these
   aren't over-firing on substrings (e.g. `"sas"` in `_TOOL_TRIGGER_KEYWORDS`
   is a loose match) and sending plain conversational turns into the slower
   ReAct loop unnecessarily.

## Running on Arch Linux

See `setup_linux.sh` — installs SDL2/SDL2_mixer (pygame's mixer needs an
MP3-capable SDL2_mixer specifically, which a minimal Arch install often
lacks), portaudio, ffmpeg, installs `uv` if it isn't already present, and
runs `uv sync` against `pyproject.toml` to create `.venv/` and a `uv.lock`.
No `pip` calls anywhere. It also runs a quick self-test that pygame's mixer
actually opens.

```bash
chmod +x setup_linux.sh
./setup_linux.sh
```

After setup, run the engine either way:

```bash
source .venv/bin/activate
python voice_engine.py --config wellness_config.json

# or, without activating (always guaranteed in sync with pyproject.toml):
uv run python voice_engine.py --config wellness_config.json
```

Managing dependencies stays inside `pyproject.toml`, not pip:

```bash
uv add some-package        # adds it, resolves, locks, installs — one step
uv remove some-package
uv sync                    # re-sync .venv after pulling a pyproject.toml change
```

Reach for `uv pip install <package>` (never bare `pip`) only in the rare
case a package can't go through `uv add` cleanly — it still targets
`.venv` instead of some other Python on the system.

### About your mic not working

The script prints a few diagnostic commands at the end (`arecord -l`,
`pactl list sources short`, checking for a stuck mute flag). Run those
first — "mic doesn't work" on Linux is very often either PipeWire/PulseAudio
not seeing the device, or the input muted at the OS mixer level, rather than
a hardware fault. If `arecord -l` shows nothing at all, that's a driver/kernel
issue; if it shows a device but `pactl list sources` doesn't, it's a
PipeWire/Pulse routing issue.

## What didn't change

`thread_store.py`, `wellness_tools.py`, `emotion_trigger_server.py`, and the
tool/crisis path through `generic_agent.build_agent()` are all untouched —
this bundle only touches the fast conversational path.