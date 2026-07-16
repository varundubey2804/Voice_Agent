"""
EMBER voice companion — Tangible Copilot Edition (optimized).
"""

import argparse
import json
import logging
import logging.handlers
import random
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from rpi_ws281x import Color, PixelStrip
    _HAS_WS281X = True
except ImportError:
    _HAS_WS281X = False

try:
    import paho.mqtt.client as mqtt
    _HAS_MQTT = True
except ImportError:
    _HAS_MQTT = False

log = logging.getLogger("ember")

# ---------------------------------------------------------------------------
# MQTT Bridge
# ---------------------------------------------------------------------------

class MqttBridge:
    def __init__(self, mqtt_config: dict):
        self._config = mqtt_config or {}
        self._client = None

        if not mqtt_config or not _HAS_MQTT:
            if mqtt_config and not _HAS_MQTT:
                log.warning("paho-mqtt not installed — running without physical motion bridge")
            return

        try:
            self._client = self._make_client()
            self._client.on_disconnect = self._on_disconnect
            self._client.reconnect_delay_set(min_delay=1, max_delay=30)
            self._client.connect(mqtt_config.get("broker_ip", "127.0.0.1"), mqtt_config.get("port", 1883), 60)
            self._client.loop_start()
            log.info("MQTT Bridge connected to %s", mqtt_config.get("broker_ip"))
        except Exception as e:
            log.warning("MQTT Bridge init failed (%s) - running without physical motion", e)
            self._client = None

    @staticmethod
    def _make_client():
        # VERSION1, not VERSION2 — keeps the plain (client, userdata, rc)
        # callback signature below instead of VERSION2's ReasonCode objects.
        try:
            return mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:
            return mqtt.Client()

    @staticmethod
    def _on_disconnect(client, userdata, rc):
        log.warning("MQTT bridge disconnected (rc=%s), will auto-reconnect", rc)

    def publish_motion(self, action_name: str):
        if not self._client:
            return
        motions = self._config.get("motions", {})
        motion_cmd = motions.get(action_name)
        if motion_cmd:
            topic = self._config.get("motion_topic", "tangible/lamp/motion")
            try:
                self._client.publish(topic, json.dumps({"motion": motion_cmd}), qos=1)
            except Exception as e:
                log.warning("failed to publish motion '%s': %s", action_name, e)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_config: dict):
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log.setLevel(level)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if log_config.get("console", True):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log_file = log_config.get("log_file")
    if log_file:
        try:
            file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=log_config.get("max_bytes", 1_000_000), backupCount=log_config.get("backup_count", 3))
            file_handler.setFormatter(fmt)
            log.addHandler(file_handler)
        except OSError as e:
            print(f"[ember] could not open log file '{log_file}' ({e}) — logging to console only", file=sys.stderr)

# ---------------------------------------------------------------------------
# Config loading + validation
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))

def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())

def validate_config(config: dict, check_paths: bool = True) -> list[str]:
    issues = []
    for key in ("sentiment_bands", "llm", "tts"):
        if key not in config:
            issues.append(f"missing required top-level config key: '{key}'")

    bands = config.get("sentiment_bands", [])
    if not bands:
        issues.append("sentiment_bands is empty — nothing would ever match a transcript")
    else:
        for i, band in enumerate(bands):
            name = band.get("name", f"#{i}")
            for req in ("name", "min_score", "max_score", "led"):
                if req not in band:
                    issues.append(f"sentiment_bands[{i}] ('{name}') missing '{req}'")
            if band.get("use_llm"):
                if not band.get("system_prompt"):
                    issues.append(f"sentiment_bands[{i}] ('{name}') has use_llm=true but no system_prompt")
            elif not band.get("replies"):
                issues.append(f"sentiment_bands[{i}] ('{name}') has use_llm=false and no replies — would produce empty output")

        sorted_bands = sorted(bands, key=lambda b: b.get("min_score", 0))
        if sorted_bands[0].get("min_score", 0) > -1.0:
            issues.append("sentiment_bands don't cover down to -1.0")
        if sorted_bands[-1].get("max_score", 0) < 1.0:
            issues.append("sentiment_bands don't cover up to 1.0")
        for a, b in zip(sorted_bands, sorted_bands[1:]):
            if abs(a.get("max_score", 0) - b.get("min_score", 0)) > 1e-6:
                issues.append(f"gap/overlap between bands '{a.get('name')}' and '{b.get('name')}'")

    if check_paths:
        for label, p in (
            ("llm.model_path", config.get("llm", {}).get("model_path")),
            ("tts.piper_bin", config.get("tts", {}).get("piper_bin")),
            ("tts.voice_path", config.get("tts", {}).get("voice_path")),
            ("stt.whisper_bin", config.get("stt", {}).get("whisper_bin")),
            ("stt.model_path", config.get("stt", {}).get("model_path")),
        ):
            if p and not Path(p).exists():
                issues.append(f"{label} points at a path that doesn't exist: {p}")
    return issues

def load_and_validate_config(path: str, check_paths: bool = True) -> dict:
    config = load_config(path)
    issues = validate_config(config, check_paths=check_paths)
    if issues:
        raise ConfigError(issues)
    return config

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

class ConversationMemory:
    def __init__(self, memory_config: Optional[dict]):
        self._config = memory_config or {}
        self._max_turns = self._config.get("max_turns", 5)
        self._persist_path = self._config.get("persist_path")
        self._turns: deque = deque(maxlen=self._max_turns)
        if self._config.get("persist") and self._persist_path:
            self._load()

    def _load(self):
        try:
            data = json.loads(Path(self._persist_path).read_text())
            for turn in data[-self._max_turns:]:
                self._turns.append(turn)
            log.info("loaded %d prior turn(s) from %s", len(self._turns), self._persist_path)
        except FileNotFoundError:
            log.info("no existing memory file at %s, starting fresh", self._persist_path)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("could not load memory file %s (%s), starting fresh", self._persist_path, e)

    def _save(self):
        if not (self._config.get("persist") and self._persist_path):
            return
        try:
            Path(self._persist_path).write_text(json.dumps(list(self._turns)))
        except OSError as e:
            log.warning("could not persist memory to %s: %s", self._persist_path, e)

    def add(self, transcript: str, reply: str, sentiment: float, band_name: str):
        self._turns.append({"transcript": transcript, "reply": reply, "sentiment": sentiment, "band": band_name, "ts": time.time()})
        self._save()

    def as_chat_messages(self) -> list[dict]:
        if not self._config.get("include_in_prompt", True):
            return []
        messages = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn["transcript"]})
            messages.append({"role": "assistant", "content": turn["reply"]})
        return messages

    def clear(self):
        self._turns.clear()
        self._save()

# ---------------------------------------------------------------------------
# Sentiment gate
# ---------------------------------------------------------------------------

class SentimentGate:
    def __init__(self, bands: list[dict]):
        self._vader = SentimentIntensityAnalyzer()
        self._bands = sorted(bands, key=lambda b: b["min_score"])

    def score(self, text: str) -> float:
        return self._vader.polarity_scores(text)["compound"]

    def band_for(self, compound_score: float) -> dict:
        for band in self._bands:
            if band["min_score"] <= compound_score <= band["max_score"]:
                return band
        return min(self._bands, key=lambda b: min(abs(compound_score - b["min_score"]), abs(compound_score - b["max_score"])))

# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

class SpeechToText:
    def __init__(self, stt_config: dict):
        self._config = stt_config

    def _record(self) -> Optional[str]:
        cfg = self._config
        wav_path = cfg.get("tmp_wav", "/tmp/ember_input.wav")
        cmd = ["arecord", "-D", cfg.get("audio_device", "default"), "-f", "S16_LE", "-r", str(cfg.get("sample_rate", 16000)), "-c", "1", "-d", str(cfg.get("record_seconds", 5)), wav_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=cfg.get("record_seconds", 5) + 5)
            return wav_path
        except subprocess.CalledProcessError as e:
            log.error("arecord failed: %s", (e.stderr or b"").decode(errors="ignore") or e)
        except subprocess.TimeoutExpired:
            log.error("arecord timed out — mic device may be unavailable or busy")
        except FileNotFoundError:
            log.error("arecord not found — is alsa-utils installed?")
        return None

    def _is_silent(self, wav_path: str) -> bool:
        threshold = self._config.get("silence_rms_threshold", 150)
        try:
            with wave.open(wav_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
        except (wave.Error, OSError, EOFError) as e:
            log.warning("could not read recorded wav (%s), treating as silence", e)
            return True
        count = len(frames) // 2
        if count == 0:
            return True
        total = sum(int.from_bytes(frames[i:i + 2], "little", signed=True) ** 2 for i in range(0, len(frames) - 1, 2))
        rms = (total / count) ** 0.5
        return rms < threshold

    def _transcribe(self, wav_path: str) -> str:
        cfg = self._config
        out_prefix = wav_path.rsplit(".", 1)[0]
        cmd = [cfg["whisper_bin"], "-m", cfg["model_path"], "-f", wav_path, "-otxt", "-of", out_prefix, "-nt"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=cfg.get("transcribe_timeout", 30))
        except subprocess.CalledProcessError as e:
            log.error("whisper.cpp failed: %s", (e.stderr or b"").decode(errors="ignore") or e)
            return ""
        except subprocess.TimeoutExpired:
            log.error("whisper.cpp timed out during transcription")
            return ""
        except FileNotFoundError:
            log.error("whisper binary not found at %s", cfg.get("whisper_bin"))
            return ""
        try:
            return Path(out_prefix + ".txt").read_text().strip()
        except FileNotFoundError:
            log.warning("whisper produced no output file for %s", wav_path)
            return ""

    def listen(self) -> Optional[str]:
        wav_path = self._record()
        if wav_path is None:
            return None
        if self._is_silent(wav_path):
            return None
        transcript = self._transcribe(wav_path)
        return transcript if transcript else None

# ---------------------------------------------------------------------------
# LLM (lazy-loaded, auto-unloaded to free RAM on a resource-constrained Pi)
# ---------------------------------------------------------------------------

class LazyLLM:
    def __init__(self, llm_config: dict):
        self._config = llm_config
        self._llm = None
        self._lock = threading.Lock()
        self._idle_unload_seconds = llm_config.get("idle_unload_seconds", 30)
        self._unload_timer: Optional[threading.Timer] = None

    def _ensure_loaded(self):
        with self._lock:
            if self._llm is None:
                log.info("loading LLM from %s", self._config.get("model_path"))
                from llama_cpp import Llama
                self._llm = Llama(model_path=self._config["model_path"], n_ctx=self._config.get("n_ctx", 512), n_threads=self._config.get("n_threads", 4), verbose=False)

    def unload(self):
        with self._lock:
            if self._llm is not None:
                log.debug("unloading LLM to free RAM")
            self._llm = None

    def _schedule_idle_unload(self):
        if self._unload_timer is not None:
            self._unload_timer.cancel()
        if self._idle_unload_seconds:
            self._unload_timer = threading.Timer(self._idle_unload_seconds, self.unload)
            self._unload_timer.daemon = True
            self._unload_timer.start()

    def generate(self, system_prompt: str, user_text: str, history: Optional[list[dict]] = None) -> str:
        self._ensure_loaded()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_text})
        result = self._llm.create_chat_completion(messages=messages, max_tokens=self._config.get("max_tokens", 60), temperature=self._config.get("temperature", 0.7))
        # Free the model after a period of inactivity instead of keeping it
        # resident in RAM indefinitely between turns.
        self._schedule_idle_unload()
        return result["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------------
# LED controller
# ---------------------------------------------------------------------------

class LedController:
    def __init__(self, led_config: dict):
        self._config = led_config
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._strip = None
        self._enabled = False
        wants_hw = led_config.get("enabled", True)
        if wants_hw and _HAS_WS281X:
            try:
                self._strip = PixelStrip(led_config["led_count"], led_config["gpio_pin"], brightness=int(led_config.get("brightness", 0.6) * 255))
                self._strip.begin()
                self._enabled = True
            except Exception as e:
                log.warning("LED hardware init failed (%s) — falling back to log-only mode", e)
        elif wants_hw and not _HAS_WS281X:
            log.info("rpi_ws281x not installed — LED running in log-only mode")
        self._effect_handlers = {"solid": self._effect_solid, "breathe": self._effect_breathe, "pulse": self._effect_pulse}

    def _set_all(self, rgb: list[int]):
        if not self._enabled:
            log.debug("LED color %s", rgb)
            return
        try:
            color = Color(*rgb)
            for i in range(self._strip.numPixels()):
                self._strip.setPixelColor(i, color)
            self._strip.show()
        except Exception as e:
            log.warning("LED write failed (%s) — disabling LED for rest of session", e)
            self._enabled = False

    def _effect_solid(self, effect_cfg: dict, stop_event: threading.Event):
        self._set_all(effect_cfg["color"])

    def _effect_breathe(self, effect_cfg: dict, stop_event: threading.Event):
        color = effect_cfg["color"]
        speed = effect_cfg.get("speed", 0.03)
        brightness, direction = 0.0, 1
        while not stop_event.is_set():
            brightness += direction * speed
            if brightness >= 1.0 or brightness <= 0.0:
                direction *= -1
                brightness = max(0.0, min(1.0, brightness))
            self._set_all([int(c * brightness) for c in color])
            time.sleep(0.02)

    def _effect_pulse(self, effect_cfg: dict, stop_event: threading.Event):
        color = effect_cfg["color"]
        for _ in range(effect_cfg.get("times", 2)):
            if stop_event.is_set():
                return
            self._set_all(color)
            time.sleep(0.25)
            self._set_all([0, 0, 0])
            time.sleep(0.15)

    def run_effect(self, effect_cfg: dict, background: bool = False):
        self.stop()
        handler = self._effect_handlers.get(effect_cfg.get("type"))
        if handler is None:
            log.warning("unknown LED effect type '%s', skipping", effect_cfg.get("type"))
            return
        self._stop_event = threading.Event()
        def safe_run():
            try:
                handler(effect_cfg, self._stop_event)
            except Exception:
                log.exception("LED effect '%s' crashed", effect_cfg.get("type"))
        if background:
            self._thread = threading.Thread(target=safe_run, daemon=True)
            self._thread.start()
        else:
            safe_run()

    def stop(self):
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=1)
        self._thread = None

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def speak(text: str, tts_config: dict) -> bool:
    output_wav = tts_config.get("output_wav", "/tmp/ember_reply.wav")
    timeout = tts_config.get("timeout", 20)
    # Use a list-form subprocess call (no shell=True) so text with quotes/
    # special characters can't break out of the command or inject shell code.
    try:
        piper = subprocess.run(
            [tts_config["piper_bin"], "--model", tts_config["voice_path"], "--output_file", output_wav],
            input=text.encode(), capture_output=True, timeout=timeout,
        )
        if piper.returncode != 0:
            log.error("Piper TTS failed: %s", piper.stderr.decode(errors="ignore"))
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log.error("Piper TTS failed: %s", e)
        return False
    try:
        subprocess.run(["aplay", output_wav], check=True, capture_output=True, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        log.error("audio playback failed: %s", e)
        return False
    return True

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    transcript: str
    sentiment: float
    band_name: str
    reply: str
    used_llm: bool

class Companion:
    def __init__(self, config: dict):
        self.config = config
        self.gate = SentimentGate(config["sentiment_bands"])
        self.llm = LazyLLM(config["llm"])
        self.leds = LedController(config.get("led", {"enabled": False}))
        self.stt = SpeechToText(config["stt"]) if "stt" in config else None
        self.memory = ConversationMemory(config.get("memory"))
        self.mqtt = MqttBridge(config.get("mqtt"))

    def respond(self, transcript: str) -> Turn:
        compound = self.gate.score(transcript)
        band = self.gate.band_for(compound)
        used_llm = False

        if band.get("use_llm"):
            thinking_effect = self.config.get("led", {}).get("thinking_effect")
            if thinking_effect:
                self.leds.run_effect(thinking_effect, background=True)

            self.mqtt.publish_motion("thinking")

            try:
                history = self.memory.as_chat_messages()
                reply = self.llm.generate(band["system_prompt"], transcript, history=history)
                used_llm = True
            except Exception:
                log.exception("LLM generation failed for band '%s' — falling back to canned reply", band.get("name"))
                fallback_pool = self.config.get("fallback_replies") or band.get("replies") or ["I'm here with you, even if I'm having trouble finding the words right now."]
                reply = random.choice(fallback_pool)
            finally:
                self.leds.stop()
        else:
            reply_pool = band.get("replies", [])
            reply = random.choice(reply_pool) if reply_pool else ""

        try:
            self.leds.run_effect(band["led"], background=(band["led"]["type"] == "breathe"))
        except Exception:
            log.exception("failed to set LED for band '%s'", band.get("name"))

        self.memory.add(transcript, reply, compound, band["name"])
        return Turn(transcript=transcript, sentiment=compound, band_name=band["name"], reply=reply, used_llm=used_llm)

    def respond_and_speak(self, transcript: str) -> Turn:
        turn = self.respond(transcript)
        if turn.reply:
            self.mqtt.publish_motion("replying")
            if not speak(turn.reply, self.config["tts"]):
                log.warning("could not speak reply aloud, continuing anyway: %r", turn.reply)
        return turn

    def listen_respond_speak(self) -> Optional[Turn]:
        if self.stt is None:
            raise RuntimeError("No 'stt' block in config — voice input not configured.")

        self.mqtt.publish_motion("listening")

        transcript = self.stt.listen()
        if not transcript:
            return None
        return self.respond_and_speak(transcript)

    def run_loop(self):
        idle_effect = self.config.get("loop", {}).get("idle_led_effect")
        pause_seconds = self.config.get("loop", {}).get("pause_between_turns", 0.5)
        max_consecutive_errors = self.config.get("loop", {}).get("max_consecutive_errors", 5)

        log.info("EMBER voice loop starting")
        consecutive_errors = 0
        try:
            while True:
                try:
                    if idle_effect:
                        self.leds.run_effect(idle_effect, background=(idle_effect["type"] == "breathe"))
                    turn = self.listen_respond_speak()
                    if turn is not None:
                        log.info("band=%s sentiment=%.2f llm=%s you=%r ember=%r", turn.band_name, turn.sentiment, turn.used_llm, turn.transcript, turn.reply)
                    consecutive_errors = 0
                except Exception:
                    consecutive_errors += 1
                    log.exception("turn failed (%d consecutive)", consecutive_errors)
                    if consecutive_errors >= max_consecutive_errors:
                        log.critical("too many consecutive failures (%d) — stopping loop", consecutive_errors)
                        break
                time.sleep(pause_seconds)
        except KeyboardInterrupt:
            log.info("interrupted by user")
        finally:
            self.leds.stop()
            self.llm.unload()
            log.info("EMBER voice loop stopped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMBER voice companion")
    parser.add_argument("--config", default=str(Path(__file__).parent / "ember_config.json"))
    parser.add_argument("--bench", action="store_true", help="text-only test, no mic/hardware required")
    parser.add_argument("--skip-path-check", action="store_true", help="skip validating model/binary paths exist")
    args = parser.parse_args()

    raw_config = load_config(args.config)
    setup_logging(raw_config.get("logging", {}))
    # FIX: was `args.skip-path-check` (invalid syntax, parsed as subtraction) —
    # argparse converts the dash to an underscore, so the real attribute is:
    issues = validate_config(raw_config, check_paths=not (args.bench or args.skip_path_check))
    if issues:
        for issue in issues:
            log.error("config issue: %s", issue)
        if args.bench:
            log.warning("continuing in --bench mode despite %d config issue(s)", len(issues))
        else:
            log.critical("refusing to start with %d config issue(s) — fix ember_config.json, or rerun with --bench / --skip-path-check for testing", len(issues))
            sys.exit(1)

    companion = Companion(raw_config)
    if args.bench:
        test_lines = ["I don't know, today's just been really hard, nothing's working out.", "I'm kind of tired but it's fine, just a long day.", "Honestly things are going pretty well right now."]
        for line in test_lines:
            turn = companion.respond(line)
            log.info("band=%s sentiment=%.2f llm=%s you=%r ember=%r", turn.band_name, turn.sentiment, turn.used_llm, turn.transcript, turn.reply)
    else:
        companion.run_loop()
