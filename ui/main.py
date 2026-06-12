import sys
import threading
import requests
import json
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.core.window import Window

# For real audio capture, pyaudio is typical. Here we stub it out for UI demonstration.
# Assuming backend runs on localhost:8000
BACKEND_URL = "http://localhost:8000"

class AssistantUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=10, spacing=10, **kwargs)
        Window.clearcolor = (0.1, 0.1, 0.1, 1)

        # Title
        self.add_widget(Label(text="Handheld AI Assistant", font_size='24sp', size_hint_y=0.1, color=(1,1,1,1)))

        # Chat history area
        self.history_scroll = ScrollView(size_hint=(1, 0.6))
        self.history_label = Label(text="Welcome! Say something...", size_hint_y=None, text_size=(Window.width-40, None), color=(0.8,0.8,0.8,1))
        self.history_label.bind(texture_size=self.history_label.setter('size'))
        self.history_scroll.add_widget(self.history_label)
        self.add_widget(self.history_scroll)

        # Status
        self.status_label = Label(text="Status: Ready", size_hint_y=0.05, color=(0.5,0.8,0.5,1))
        self.add_widget(self.status_label)

        # Controls area
        controls = BoxLayout(orientation='horizontal', size_hint_y=0.15, spacing=10)

        self.ptt_button = Button(text="Push to Talk", background_color=(0.2, 0.6, 1, 1))
        self.ptt_button.bind(on_press=self.start_recording)
        self.ptt_button.bind(on_release=self.stop_recording)

        self.text_input = TextInput(hint_text="Or type here...", multiline=False)
        self.text_input.bind(on_text_validate=self.send_text)

        send_btn = Button(text="Send", size_hint_x=0.3, background_color=(0.2, 0.8, 0.2, 1))
        send_btn.bind(on_press=self.send_text)

        controls.add_widget(self.ptt_button)
        controls.add_widget(self.text_input)
        controls.add_widget(send_btn)

        self.add_widget(controls)

    def append_message(self, role, text):
        current_text = self.history_label.text
        new_text = f"[{role}]: {text}\n"
        self.history_label.text = current_text + "\n" + new_text
        # Scroll to bottom (basic implementation)
        self.history_scroll.scroll_y = 0

    def start_recording(self, instance):
        self.status_label.text = "Status: Listening..."
        self.ptt_button.background_color = (1, 0.2, 0.2, 1)
        self.ptt_button.text = "Recording..."
        # In a real app, start PyAudio stream here

    def stop_recording(self, instance):
        self.status_label.text = "Status: Processing Audio..."
        self.ptt_button.background_color = (0.2, 0.6, 1, 1)
        self.ptt_button.text = "Push to Talk"
        # In a real app, stop stream, save to bytes, and send to /transcribe-and-chat
        # We simulate sending a stub audio request here
        threading.Thread(target=self._mock_send_audio).start()

    def _mock_send_audio(self):
        # Stub for sending audio
        try:
            # Just mimicking a text request for demo
            self.send_request("Hello, this is a mock audio message.")
        except Exception as e:
            Clock.schedule_once(lambda dt: self.update_status(f"Error: {e}"))

    def send_text(self, instance=None):
        text = self.text_input.text.strip()
        if not text:
            return
        self.text_input.text = ""
        self.append_message("User", text)
        self.status_label.text = "Status: Thinking..."

        # Run network request in background
        threading.Thread(target=self.send_request, args=(text,)).start()

    def send_request(self, text):
        try:
            response = requests.post(f"{BACKEND_URL}/chat", json={"text": text}, timeout=30)
            if response.status_code == 200:
                data = response.json()
                agent_text = data.get("agent_text", "No response.")
                Clock.schedule_once(lambda dt: self.append_message("Agent", agent_text))
                Clock.schedule_once(lambda dt: self.update_status("Status: Ready"))
            else:
                Clock.schedule_once(lambda dt: self.update_status(f"Error: Server returned {response.status_code}"))
        except requests.exceptions.RequestException as e:
            Clock.schedule_once(lambda dt: self.update_status(f"Connection Error: Is backend running?"))

    def update_status(self, text):
        self.status_label.text = text


class HandheldAIApp(App):
    def build(self):
        return AssistantUI()

if __name__ == '__main__':
    HandheldAIApp().run()
