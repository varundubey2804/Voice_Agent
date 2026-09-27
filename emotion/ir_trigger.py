"""
emotion/ir_trigger.py — Fires a basic curl command to trigger MacroDroid webhooks.
"""
from __future__ import annotations
import subprocess
import threading
from config import MACRODROID_DEVICE_ID
EMOTION_TO_TRIGGER = {
    "happy":     "happy",
    "surprise":  "happy",     # Map surprise to happy/bright
    "neutral":   "neutral",
    "sad":       "stressed",  # Map sad/angry/fear to stressed/calm
    "angry":     "stressed",
    "fear":      "stressed",
    "disgust":   "stressed",
    "stressed":  "stressed",  # sentiment_analysis.py emits this label directly
    # Backward compatibility aliases
    "positive":  "happy",
    "negative":  "stressed",
    "calm":      "calm",
    "confused":  "confused",
}
_last_sent: str | None = None
def _fire_curl(url: str) -> None:
    """Executes the exact curl command."""
    try:
        res = subprocess.run(
            ["curl", "-s", "--max-time", "4", url],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.stdout:
            print(f" MacroDroid response: {res.stdout.strip()}")
    except Exception as e:
        print(f"⚠️ Curl failed: {e}")
def trigger(emotion_or_color: str, force: bool = False, blocking: bool = False) -> None:
    """
    Fires: curl "https://trigger.macrodroid.com/<DEVICE_ID>/<action>"
    """
    global _last_sent
    key = emotion_or_color.lower().strip()
    action = EMOTION_TO_TRIGGER.get(key, key)
    if not force and action == _last_sent:
        return  # Avoid sending duplicate triggers
    _last_sent = action
    url = f"https://trigger.macrodroid.com/{MACRODROID_DEVICE_ID}/{action}"
    print(f"🌐 Firing: curl \"{url}\"")
    if blocking:
        _fire_curl(url)
    else:
        # Non-daemon thread ensures the request finishes even if caller continues
        threading.Thread(target=_fire_curl, args=(url,), daemon=False).start()
# Aliases for your pipeline
def send_emotion_color(emotion_name: str) -> None:
    trigger(emotion_name)
def send_color(color_name: str, force: bool = False) -> None:
    trigger(color_name, force=force)
if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "calm"
    # When testing from CLI, block so Python doesn't exit before curl finishes
    trigger(target, force=True, blocking=True)