import io
import pyaudio
from gtts import gTTS
from pydub import AudioSegment

def play_text_to_speech_stream(text):
    """Generate TTS audio in memory and stream directly to PyAudio."""
    # Generate speech in memory
    tts = gTTS(text=text, lang='en')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # Decode MP3 to raw PCM
    audio = AudioSegment.from_file(mp3_fp, format="mp3")
    raw_data = audio.raw_data

    # Setup PyAudio playback
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(audio.sample_width),
        channels=audio.channels,
        rate=audio.frame_rate,
        output=True
    )

    # Play audio directly
    stream.write(raw_data)

    stream.stop_stream()
    stream.close()
    p.terminate()
