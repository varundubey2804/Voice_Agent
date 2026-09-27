#!/usr/bin/env bash
# setup_linux.sh — Set up Veena's voice pipeline on Arch Linux, using uv
# for the Python environment instead of venv/pip.
#
# WHAT THIS INSTALLS AND WHY
# ────────────────────────────
# pygame's mixer (used for TTS playback) is a thin wrapper around SDL2 +
# SDL2_mixer. A minimal Arch install often has SDL2 but NOT an MP3-capable
# SDL2_mixer, which is exactly the format edge_tts produces — so playback
# can silently fail or throw a codec error even though pygame "initialized"
# fine. We install the full SDL2_mixer stack explicitly to avoid that.
#
# We also install portaudio, since any real STT/mic-capture library
# (sounddevice, pyaudio) needs it. ffmpeg is intentionally NOT installed by
# this script — see the comment above the pacman call below for why.
#
# PYTHON ENV: uv instead of venv/pip
# ────────────────────────────────────
# uv replaces `python -m venv` + `pip install` with one faster, cached
# tool. It creates the same kind of .venv/ directory underneath, so
# nothing else about how you run the project changes — you still
# `source .venv/bin/activate` afterward (or prefix commands with `uv run`).
#
# PYTHON ENV: pure uv, no pip
# ────────────────────────────
# `uv sync` reads pyproject.toml, resolves + locks dependencies (writing
# uv.lock on first run), and installs everything into ./.venv — in one
# step, no separate `python -m venv` / `pip install` calls. Adding or
# removing a dependency later is `uv add <pkg>` / `uv remove <pkg>`, which
# updates pyproject.toml + the lockfile + .venv together. Plain pip is not
# used anywhere in this script; see the note at the bottom for the rare
# case where a package genuinely needs it.
#
# Run with: bash setup_linux.sh
set -euo pipefail

echo "── Installing system audio dependencies (pacman) ──"
# NOTE: deliberately NOT `pacman -Syu` here. -u does a FULL system upgrade
# of every installed package, which can collide with unrelated pinned
# dependencies elsewhere on your system (e.g. an AUR Electron app pinned
# to an old electron27, which in turn pins old libjpeg/libjsoncpp — this
# is a real, common Arch conflict, nothing to do with this script).
# -Sy (sync the package DB) + -S --needed (install just what we need,
# pulling only the deps those specific packages require) gets us what we
# need without touching anything unrelated already on your system.
#
# Partial upgrades like this are officially "unsupported" by Arch as a
# blanket policy, but for a handful of self-contained libraries like these
# it's the standard workaround. Run a real `sudo pacman -Syu` on your own
# schedule to keep the system properly in sync — if/when that hits the
# same electron27 conflict, the fix is to resolve which package pins it
# (`pactree -r electron27`) and update or remove that package, not to
# avoid full upgrades forever.
#
# ffmpeg is DELIBERATELY LEFT OUT of this list. Nothing in this project
# needs it: edge_tts produces mp3 directly, and pygame's mixer plays it
# via SDL2_mixer, not ffmpeg. It's only handy if you ever want to convert
# or inspect audio files by hand later. If/when you want it, install it
# as part of a real `sudo pacman -Syu` (so it upgrades together with
# whatever else on your system pins an old ffmpeg soname, e.g. chromaprint
# or freerdp) rather than on its own with --needed, which is exactly what
# produced the soname conflict you just hit.
sudo pacman -Sy
sudo pacman -S --needed \
    sdl2 sdl2_mixer sdl2_ttf sdl2_image \
    portaudio \
    base-devel

# ── On Debian/Ubuntu instead, use: ──
#   sudo apt update && sudo apt install -y \
#     libsdl2-2.0-0 libsdl2-mixer-2.0-0 \
#     portaudio19-dev build-essential
# (uv installs its own managed Python, so a system python3 package isn't
# strictly required either way — but build-essential is, for compiling
# webrtcvad's C extension. ffmpeg intentionally omitted — see note above.)

echo
echo "── Checking that PulseAudio/PipeWire is actually running ──"
if command -v pactl >/dev/null 2>&1; then
    pactl info | head -n 5 || echo "⚠️  pactl couldn't reach a running audio server."
else
    echo "⚠️  pactl not found — install 'pipewire-pulse' or 'pulseaudio' if playback fails."
fi

echo
echo "── Installing uv (if not already present) ──"
if command -v uv >/dev/null 2>&1; then
    echo "✅ uv already installed: $(uv --version)"
else
    # Arch ships uv in the 'extra' repo on most mirrors; fall back to the
    # official installer script if pacman doesn't have it for some reason.
    if sudo pacman -S --needed --noconfirm uv 2>/dev/null; then
        echo "✅ Installed uv via pacman: $(uv --version)"
    else
        echo "pacman didn't have 'uv' — falling back to the official installer"
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # The installer puts uv in ~/.local/bin — make sure this shell sees it
        export PATH="$HOME/.local/bin:$PATH"
        echo "✅ Installed uv via official script: $(uv --version)"
        echo "   (add ~/.local/bin to your PATH permanently if it isn't already —"
        echo "    the installer usually does this for you in ~/.bashrc / ~/.zshrc)"
    fi
fi

echo
echo "── Syncing the environment from pyproject.toml (creates ./.venv + uv.lock) ──"
# uv sync reads pyproject.toml, resolves + locks (uv.lock) on first run, and
# installs into .venv — this replaces `python -m venv` + `pip install -r ...`
# in one step. Re-running it later is fast and idempotent.
uv sync

echo
echo "── Quick self-test: can pygame open the mixer? ──"
uv run python - <<'EOF'
import pygame
try:
    pygame.mixer.init()
    print("✅ pygame mixer initialized OK:", pygame.mixer.get_init())
except Exception as exc:
    print("❌ pygame mixer failed:", exc)
    print("   -> check that SDL2_mixer is installed and an audio server is running")
EOF

echo
echo "── Done ──"
echo "Two ways to run things from here on:"
echo "  1) source .venv/bin/activate   # then run python scripts normally"
echo "  2) uv run python voice_engine.py ...   # no activation needed, always in sync"
echo
echo "Managing dependencies from here on (stay in pyproject.toml, not pip):"
echo "  uv add <package>            # add + resolve + lock + install, one step"
echo "  uv remove <package>         # same, in reverse"
echo "  uv sync                     # re-sync .venv after pulling pyproject.toml changes"
echo
echo "  Only reach for pip if a package genuinely can't be installed by uv"
echo "  (very rare — e.g. it needs a manual --no-build-isolation flag uv doesn't"
echo "  expose, or an editable install from a weird local path). In that case,"
echo "  use 'uv pip install <package>' rather than bare pip, so it still installs"
echo "  into .venv instead of silently going to some other Python on your system."
echo
echo "MIC TROUBLESHOOTING (since you mentioned the laptop mic may not work):"
echo "  arecord -l                 # list detected capture devices"
echo "  pactl list sources short    # list PulseAudio/PipeWire capture sources"
echo "  If a device is listed but silent, check it isn't muted:"
echo "  pactl list sources | grep -A5 'Mute:'"