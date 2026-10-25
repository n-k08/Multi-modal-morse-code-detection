import asyncio
import io
import os
import re
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import AudioProcessorBase, VideoProcessorBase, webrtc_streamer


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "AI-Morse-Decoder"))

from audio import FileAudioSource
from timing_decoder import TimingDecoder
from tone_detector import ToneDetector
from morse_table import MORSE_CODE


st.set_page_config(page_title="Morse Field Desk", page_icon=".", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink: #14231f; --paper: #f3f5ee; --green: #b6dc52; --blue: #85c9d5; }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, rgba(182,220,82,.13), transparent 36%),
                    repeating-linear-gradient(0deg, rgba(20,35,31,.025) 0, rgba(20,35,31,.025) 1px, transparent 1px, transparent 28px),
                    var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #14231f; }
    [data-testid="stSidebar"] * { color: #f3f5ee; }
    h1, h2, h3, p, label, button, input { font-family: 'Space Grotesk', sans-serif; }
    h1 { font-size: 2.4rem; letter-spacing: 0; color: var(--ink); }
    code, pre, [data-testid="stMetricValue"] { font-family: 'DM Mono', monospace; }
    [data-testid="stMetric"] { background: rgba(255,255,255,.62); border-left: 3px solid var(--green); padding: 12px 16px; }
    .stButton > button { border-radius: 4px; border: 1px solid var(--ink); background: var(--ink); color: white; }
    .stButton > button:hover { border-color: var(--ink); background: #29463d; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)


class HandGestureProcessor(VideoProcessorBase):
    def __init__(self, dash_fingers: int = 3, letter_pause: float = 1.4):
        self.hands_api = mp.solutions.hands
        self.hands = self.hands_api.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.drawer = mp.solutions.drawing_utils
        self.dash_fingers = dash_fingers
        self.letter_pause = letter_pause
        self.lock = threading.Lock()
        self.pattern = ""
        self.text = ""
        self.last_gesture_at = 0.0
        self.last_gesture = None
        self.current_gesture = "No hand"

    def _classify_hand(self, landmarks) -> str:
        points = landmarks.landmark
        wrist = np.array([points[self.hands_api.HandLandmark.WRIST].x,
                          points[self.hands_api.HandLandmark.WRIST].y])
        fingers = (
            (self.hands_api.HandLandmark.INDEX_FINGER_TIP, self.hands_api.HandLandmark.INDEX_FINGER_PIP),
            (self.hands_api.HandLandmark.MIDDLE_FINGER_TIP, self.hands_api.HandLandmark.MIDDLE_FINGER_PIP),
            (self.hands_api.HandLandmark.RING_FINGER_TIP, self.hands_api.HandLandmark.RING_FINGER_PIP),
            (self.hands_api.HandLandmark.PINKY_TIP, self.hands_api.HandLandmark.PINKY_PIP),
            (self.hands_api.HandLandmark.THUMB_TIP, self.hands_api.HandLandmark.THUMB_IP),
        )
        extended = 0
        for tip_id, joint_id in fingers:
            tip = np.array([points[tip_id].x, points[tip_id].y])
            joint = np.array([points[joint_id].x, points[joint_id].y])
            if np.linalg.norm(tip - wrist) > np.linalg.norm(joint - wrist) * 1.12:
                extended += 1
        return "dash" if extended >= self.dash_fingers else "dot"

    def _finish_letter(self) -> None:
        if self.pattern:
            self.text += MORSE_CODE.get(self.pattern, "?")
            self.pattern = ""

    def _snapshot(self):
        with self.lock:
            return self.pattern, self.text, self.current_gesture

    def clear(self) -> None:
        with self.lock:
            self.pattern = ""
            self.text = ""
            self.last_gesture_at = 0.0
            self.last_gesture = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        detected = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(output, hand, self.hands_api.HAND_CONNECTIONS)
            detected = self._classify_hand(hand)

        now = time.monotonic()
        with self.lock:
            self.current_gesture = detected or "No hand"
            word_pause = self.letter_pause * 2.3
            if self.pattern and now - self.last_gesture_at >= self.letter_pause:
                self._finish_letter()
            if not self.pattern and self.text and now - self.last_gesture_at >= word_pause:
                if not self.text.endswith(" "):
                    self.text += " "

            if detected is None:
                self.last_gesture = None
            elif detected != self.last_gesture and now - self.last_gesture_at >= 0.35:
                if self.pattern and now - self.last_gesture_at >= word_pause and not self.text.endswith(" "):
                    self._finish_letter()
                    self.text += " "
                elif self.pattern and now - self.last_gesture_at >= self.letter_pause:
                    self._finish_letter()
                self.pattern += "." if detected == "dot" else "-"
                self.last_gesture_at = now
                self.last_gesture = detected

            pattern = self.pattern or "-"
            text = self.text[-42:] or "Waiting for a gesture"
            gesture_label = detected or "No hand"

        cv2.putText(output, f"Gesture: {gesture_label}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (182, 220, 82), 2)
        cv2.putText(output, f"Morse: {pattern}", (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, f"Text: {text}", (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return av.VideoFrame.from_ndarray(output, format="bgr24")


class LiveAudioProcessor(AudioProcessorBase):
    def __init__(self, wpm=20, frequency=700, filter_q=10, threshold=0.35,
                 volume_min_db=-60, volume_max_db=0, gap_scale=1.0,
                 strict_protocol=True):
        self.lock = threading.Lock()
        self.sample_rate = 0
        self.elapsed = 0.0
        self.level_db = -120.0
        self.levels = deque(maxlen=120)
        self.detector = None
        self.decoder = None
        self.update_settings(
            wpm, frequency, filter_q, threshold, volume_min_db,
            volume_max_db, gap_scale, strict_protocol,
        )

    def update_settings(self, wpm, frequency, filter_q, threshold,
                        volume_min_db, volume_max_db, gap_scale, strict_protocol):
        settings = (int(wpm), frequency, float(filter_q), float(threshold),
                    float(volume_min_db), float(volume_max_db), float(gap_scale),
                    bool(strict_protocol))
        with self.lock:
            if settings == getattr(self, "settings", None):
                return
            self.settings = settings
            (self.wpm, self.frequency, self.filter_q, self.threshold,
             self.volume_min_db, self.volume_max_db, self.gap_scale,
             self.strict_protocol) = settings
            if self.sample_rate:
                self._reset_pipeline()

    def _reset_pipeline(self):
        self.detector = ToneDetector(
            sample_rate=self.sample_rate,
            target_freq=self.frequency,
            threshold=self.threshold,
            hold_off_ms=40,
        )
        if self.frequency is not None:
            self.detector.BAND_WIDTH = max(20.0, self.frequency / self.filter_q)
        self.decoder = TimingDecoder(
            wpm=self.wpm,
            gap_scale=self.gap_scale,
            strict_protocol=self.strict_protocol,
        )
        self.elapsed = 0.0

    def _snapshot(self):
        with self.lock:
            return {
                "text": self.decoder.raw_text if self.decoder else "",
                "wpm": self.decoder.current_wpm if self.decoder else float(self.wpm),
                "frequency": self.detector.detected_freq if self.detector else self.frequency,
                "level": self.level_db,
                "tone": bool(self.detector and self.detector._is_on),
                "levels": list(self.levels),
            }

    def clear(self):
        with self.lock:
            self.levels.clear()
            self.level_db = -120.0
            if self.sample_rate:
                self._reset_pipeline()

    def recv(self, frame):
        frame_rate = int(frame.sample_rate or 48000)
        samples = frame.to_ndarray()
        channels = max(1, len(frame.layout.channels))
        if samples.ndim > 1:
            if frame.format.is_planar:
                mono = samples.reshape(channels, -1).mean(axis=0)
            else:
                mono = samples.reshape(-1, channels).mean(axis=1)
        else:
            mono = samples
        if np.issubdtype(mono.dtype, np.integer):
            limits = np.iinfo(mono.dtype)
            mono = mono.astype(np.float32) / max(abs(limits.min), limits.max)
        else:
            mono = mono.astype(np.float32)

        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        level_db = 20.0 * np.log10(max(rms, 1e-6))
        with self.lock:
            if frame_rate != self.sample_rate:
                self.sample_rate = frame_rate
                self._reset_pipeline()
            timestamp = self.elapsed
            in_range = self.volume_min_db <= level_db <= self.volume_max_db
            signal = mono if in_range else np.zeros_like(mono)
            event = self.detector.process(signal, timestamp)
            if event:
                self.decoder.process_event(*event)
            self.elapsed += len(mono) / self.sample_rate
            self.decoder.check_timeout(self.elapsed)
            if self.detector.detected_freq is not None:
                self.detector.BAND_WIDTH = max(20.0, self.detector.detected_freq / self.filter_q)
            self.level_db = max(-120.0, min(0.0, level_db))
            self.levels.append(self.level_db)
        return frame


def _audio_level_db(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-6))


async def _decode_audio(path: str, sample_rate: int, wpm: int, frequency: float | None,
                        threshold: float, progress, filter_q: float = 10,
                        volume_min_db: float = -60, volume_max_db: float = 0,
                        gap_scale: float = 1.0, strict_protocol: bool = False):
    detector = ToneDetector(
        sample_rate=sample_rate,
        target_freq=frequency,
        threshold=threshold,
        hold_off_ms=40,
    )
    if frequency is not None:
        detector.BAND_WIDTH = max(20.0, frequency / filter_q)
    decoder = TimingDecoder(wpm=wpm, gap_scale=gap_scale, strict_protocol=strict_protocol)

    async with FileAudioSource(path, target_sample_rate=sample_rate) as source:
        total = source.total_chunks or 1
        last_update = 0.0
        chunk_duration = source.chunk_size / sample_rate
        index = 0
        async for chunk, timestamp in source.chunks():
            index += 1
            signal = chunk if volume_min_db <= _audio_level_db(chunk) <= volume_max_db else np.zeros_like(chunk)
            event = detector.process(signal, timestamp)
            if event:
                decoder.process_event(*event)
            decoder.check_timeout(timestamp)
            if detector.detected_freq is not None:
                detector.BAND_WIDTH = max(20.0, detector.detected_freq / filter_q)
            if timestamp - last_update >= 1.0:
                progress.progress(min(index / total, 1.0), text=f"Decoding audio: {index}/{total} chunks")
                last_update = timestamp

        silence = np.zeros(int(sample_rate * chunk_duration), dtype=np.float32)
        for offset in range(1, 11):
            timestamp = source.duration_seconds + offset * chunk_duration
            event = detector.process(silence, timestamp)
            if event:
                decoder.process_event(*event)
            decoder.check_timeout(timestamp)
        decoder.flush()
        progress.progress(1.0, text="Audio decode complete")
        return {
            "text": decoder.raw_text.strip(),
            "frequency": detector.detected_freq,
            "wpm": decoder.current_wpm,
            "duration": source.duration_seconds,
        }


def decode_audio(path: str, sample_rate: int, wpm: int, frequency: float | None,
                 threshold: float, progress, filter_q: float = 10,
                 volume_min_db: float = -60, volume_max_db: float = 0,
                 gap_scale: float = 1.0, strict_protocol: bool = False):
    return asyncio.run(_decode_audio(
        path, sample_rate, wpm, frequency, threshold, progress, filter_q,
        volume_min_db, volume_max_db, gap_scale, strict_protocol,
    ))


def decode_written_morse(source: str) -> str:
    normalized = source.replace("·", ".").replace("−", "-").replace("–", "-").replace("—", "-")
    decoded_lines = []
    for line in normalized.splitlines():
        if not line.strip():
            decoded_lines.append("")
            continue
        words = re.split(r"/|\s{3,}", line.strip())
        decoded_words = []
        for word in words:
            codes = word.split()
            if codes:
                decoded_words.append("".join(MORSE_CODE.get(code, f"[{code}]") for code in codes))
        decoded_lines.append(" ".join(decoded_words))
    return "\n".join(decoded_lines).rstrip()


def encode_text_to_morse(source: str) -> tuple[str, list[str]]:
    reverse_table = {
        character: pattern
        for pattern, character in MORSE_CODE.items()
        if len(character) == 1
    }
    encoded_lines = []
    unsupported = set()
    for line in source.upper().splitlines():
        if not line.strip():
            encoded_lines.append("")
            continue
        encoded_words = []
        for word in line.split():
            patterns = []
            for character in word:
                pattern = reverse_table.get(character)
                if pattern:
                    patterns.append(pattern)
                else:
                    unsupported.add(character)
            if patterns:
                encoded_words.append(" ".join(patterns))
        encoded_lines.append(" / ".join(encoded_words))
    return "\n".join(encoded_lines).rstrip(), sorted(unsupported)


def generate_morse_audio(morse: str, wpm: int, frequency: int,
                         sample_rate: int = 22050) -> tuple[bytes, float]:
    unit_seconds = 1.2 / wpm
    unit_samples = max(1, round(sample_rate * unit_seconds))
    segments = []
    sample_position = 0

    def append_silence(units: int) -> None:
        nonlocal sample_position
        count = unit_samples * units
        if count:
            segments.append(np.zeros(count, dtype=np.float32))
            sample_position += count

    def append_tone(units: int) -> None:
        nonlocal sample_position
        count = unit_samples * units
        indices = np.arange(count, dtype=np.float32)
        tone = 0.72 * np.sin(2 * np.pi * frequency * (sample_position + indices) / sample_rate)
        ramp_samples = min(round(sample_rate * 0.004), count // 2)
        if ramp_samples:
            ramp = np.linspace(0.0, 1.0, ramp_samples, endpoint=False, dtype=np.float32)
            tone[:ramp_samples] *= ramp
            tone[-ramp_samples:] *= ramp[::-1]
        segments.append(tone.astype(np.float32))
        sample_position += count

    lines = morse.splitlines()
    previous_line_index = None
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        if previous_line_index is not None:
            blank_lines = line_index - previous_line_index - 1
            append_silence(14 if blank_lines else 7)
        words = line.split(" / ")
        for word_index, word in enumerate(words):
            letters = word.split()
            for letter_index, letter in enumerate(letters):
                for symbol_index, symbol in enumerate(letter):
                    append_tone(1 if symbol == "." else 3)
                    if symbol_index < len(letter) - 1:
                        append_silence(1)
                if letter_index < len(letters) - 1:
                    append_silence(3)
            if word_index < len(words) - 1:
                append_silence(7)
        previous_line_index = line_index

    if not segments:
        return b"", 0.0
    append_silence(1)
    samples = np.concatenate(segments)
    output = io.BytesIO()
    import soundfile as sf

    sf.write(output, samples, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue(), len(samples) / sample_rate


def farnsworth_gap_scale(character_wpm: float, overall_wpm: float) -> float:
    if overall_wpm >= character_wpm:
        return 1.0
    return max(1.0, (50.0 * character_wpm / overall_wpm - 31.0) / 19.0)


@st.fragment(run_every=0.75)
def render_live_status(processor, mode: str):
    if processor is None:
        st.info("Start the camera above to begin decoding.")
        return
    if mode == "gesture":
        pattern, text, gesture = processor._snapshot()
        first, second, third = st.columns(3)
        first.metric("Gesture", gesture)
        second.metric("Current Morse", pattern or "- - -")
        third.metric("Decoded text", text.strip() or "-")
        st.caption("Closed hand = dot. Three or more extended fingers = dash. Briefly lower the hand between identical symbols.")


@st.fragment(run_every=0.75)
def render_live_audio_status(processor):
    if processor is None:
        st.info("Press Start to listen through your browser microphone.")
        return
    state = processor._snapshot()
    first, second, third, fourth = st.columns(4)
    first.metric("Words", len(state["text"].split()))
    second.metric("Characters", len(state["text"]))
    third.metric("Speed", f"{state['wpm']:.1f} WPM")
    fourth.metric("Tone", f"{state['frequency']:.0f} Hz" if state["frequency"] else "Auto")
    status = "TONE DETECTED" if state["tone"] else "LISTENING"
    st.caption(f"{status} · INPUT LEVEL {state['level']:.1f} dBFS")
    if state["levels"]:
        st.line_chart({"Input level (dBFS)": state["levels"]}, height=150)
    st.markdown("**Decoded message**")
    st.code(state["text"] or "Waiting for Morse audio...", language=None, wrap_lines=True)
    first_action, second_action = st.columns(2)
    first_action.download_button(
        "Download text",
        data=state["text"],
        file_name="morse-transcript.txt",
        mime="text/plain",
        disabled=not state["text"],
        key="live-audio-download",
    )
    if second_action.button("Clear message", key="live-audio-clear"):
        processor.clear()


def render_audio_mode():
    st.header("Morse Code Audio Decoder")
    st.caption("Decode Morse beeps from a live microphone or an audio file.")

    with st.expander("Decoder configuration", expanded=True):
        first, second, third = st.columns(3)
        wpm = first.number_input("WPM rate", min_value=5, max_value=60, value=20)
        farnsworth = second.checkbox("Farnsworth timing")
        farnsworth_wpm = third.number_input(
            "Farnsworth WPM",
            min_value=5,
            max_value=int(wpm),
            value=min(15, int(wpm)),
            disabled=not farnsworth,
        )
        fourth, fifth, sixth = st.columns(3)
        auto_frequency = fourth.checkbox("Automatic frequency tracking", value=True)
        frequency = fifth.number_input(
            "Tone frequency (Hz)", min_value=300, max_value=1500, value=700,
            disabled=auto_frequency,
        )
        filter_q = sixth.number_input("Filter Q", min_value=1, max_value=50, value=10)
        seventh, eighth, ninth = st.columns(3)
        threshold = seventh.slider("Signal threshold", 0.05, 0.8, 0.35, 0.05)
        volume_min_db = eighth.number_input("Volume dB min", min_value=-120, max_value=0, value=-60)
        volume_max_db = ninth.number_input("Volume dB max", min_value=-60, max_value=6, value=0)
        strict_protocol = st.checkbox("Strict protocol filtering", value=True)

    if volume_min_db >= volume_max_db:
        st.error("Volume dB max must be greater than Volume dB min.")
        return

    frequency_setting = None if auto_frequency else float(frequency)
    gap_scale = farnsworth_gap_scale(wpm, farnsworth_wpm) if farnsworth else 1.0
    settings = {
        "wpm": int(wpm),
        "frequency": frequency_setting,
        "filter_q": float(filter_q),
        "threshold": float(threshold),
        "volume_min_db": float(volume_min_db),
        "volume_max_db": float(volume_max_db),
        "gap_scale": gap_scale,
        "strict_protocol": strict_protocol,
    }

    live_tab, file_tab = st.tabs(("Live microphone", "Audio file"))
    with live_tab:
        st.markdown("#### Live microphone")
        context = webrtc_streamer(
            key="audio-live-morse",
            audio_processor_factory=lambda: LiveAudioProcessor(**settings),
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            audio_receiver_size=8,
            sendback_audio=False,
        )
        processor = context.audio_processor
        if processor:
            processor.update_settings(**settings)
        render_live_audio_status(processor)

    with file_tab:
        st.markdown("#### Audio file")
        audio_file = st.file_uploader(
            "Drop or choose an audio file",
            type=["wav", "flac", "ogg", "aiff", "aif", "mp3", "aac", "m4a", "mp4", "wma"],
            key="audio-file-upload",
        )
        if audio_file:
            st.audio(audio_file)
            st.caption(f"{audio_file.name} · {audio_file.size / 1024:.1f} KB")
        first, second = st.columns(2)
        sample_rate = first.selectbox("Processing sample rate", (22050, 44100), index=1)
        decode_clicked = second.button("Decode file", type="primary", disabled=audio_file is None)
        if decode_clicked:
            suffix = Path(getattr(audio_file, "name", "recording.wav")).suffix or ".wav"
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
                    temporary.write(audio_file.getvalue())
                    temporary_path = temporary.name
                progress = st.progress(0.0, text="Preparing audio")
                st.session_state["audio_result"] = decode_audio(
                    temporary_path,
                    sample_rate,
                    int(wpm),
                    frequency_setting,
                    float(threshold),
                    progress,
                    float(filter_q),
                    float(volume_min_db),
                    float(volume_max_db),
                    gap_scale,
                    strict_protocol,
                )
            except Exception as exc:
                st.error(f"Could not decode this audio: {exc}")
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.unlink(temporary_path)

        result = st.session_state.get("audio_result")
        if result:
            st.divider()
            st.markdown("#### Decoded message")
            first, second, third = st.columns(3)
            first.metric("Words", len(result["text"].split()))
            second.metric("Characters", len(result["text"]))
            third.metric("Speed", f"{result['wpm']:.1f} WPM")
            st.caption(
                f"{result['frequency']:.0f} Hz detected" if result["frequency"] else "No tone detected"
            )
            st.code(result["text"] or "No Morse signal detected.", language=None, wrap_lines=True)
            first_action, second_action = st.columns(2)
            first_action.download_button(
                "Download text", result["text"], "morse-transcript.txt", "text/plain",
                disabled=not result["text"], key="file-audio-download",
            )
            if second_action.button("Clear message", key="file-audio-clear"):
                st.session_state.pop("audio_result", None)
                st.rerun()


def render_gesture_mode():
    st.header("Hand Gesture Morse")
    dash_fingers = st.slider("Extended fingers for dash", min_value=2, max_value=5, value=3)
    letter_pause = st.slider("Pause before finishing a letter (seconds)", min_value=0.8, max_value=2.5, value=1.4, step=0.1)
    context = webrtc_streamer(
        key="hand-gesture-morse",
        video_processor_factory=lambda: HandGestureProcessor(dash_fingers, letter_pause),
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    render_live_status(context.video_processor, "gesture")
    if context.video_processor and st.button("Clear gesture output", key="clear-gesture"):
        context.video_processor.clear()


def render_written_morse_mode():
    st.header("Text and Morse")
    st.caption("Translate between written text and Morse code, then listen to the encoded message.")
    decode_tab, encode_tab = st.tabs(("Morse to text", "Text to Morse and audio"))

    with decode_tab:
        source = st.text_area(
            "Morse code",
            placeholder=".... . .-.. .-.. --- / .-- --- .-. .-.. -..\n\n... --- ...",
            height=160,
            help="Separate letters with spaces; separate words with / or three spaces. Line breaks are preserved.",
            key="written-morse-input",
        )
        if source.strip():
            decoded = decode_written_morse(source)
            letters = sum(character.isalpha() for character in decoded)
            digits = sum(character.isdigit() for character in decoded)
            first, second = st.columns(2)
            first.metric("Letters", letters)
            second.metric("Numbers", digits)
            st.text_area("Decoded text", value=decoded, height=160, disabled=True)

    with encode_tab:
        with st.form("text-to-morse-form"):
            source_text = st.text_area(
                "Text to encode",
                placeholder="Hello world!\nSOS 2026",
                max_chars=300,
                height=150,
                key="text-to-morse-input",
            )
            first, second, third = st.columns(3)
            wpm = first.number_input("Playback speed (WPM)", min_value=5, max_value=50, value=20, key="encode-wpm")
            frequency = second.number_input("Tone (Hz)", min_value=300, max_value=1200, value=700, key="encode-frequency")
            sample_rate = third.selectbox("Audio quality", (22050, 44100), index=0, key="encode-sample-rate")
            generate_clicked = st.form_submit_button("Generate Morse audio", type="primary")

        encoded, unsupported = encode_text_to_morse(source_text)
        if generate_clicked:
            if not encoded:
                st.warning("Enter at least one supported letter or number.")
            else:
                audio_bytes, duration = generate_morse_audio(encoded, int(wpm), int(frequency), int(sample_rate))
                st.session_state["generated_morse_audio"] = {
                    "source": source_text,
                    "encoded": encoded,
                    "settings": (int(wpm), int(frequency), int(sample_rate)),
                    "bytes": audio_bytes,
                    "duration": duration,
                }

        if encoded:
            st.markdown("**Morse code**")
            st.code(encoded, language=None, wrap_lines=True)
        if unsupported:
            st.warning(f"Skipped unsupported characters: {' '.join(unsupported)}")

        generated = st.session_state.get("generated_morse_audio")
        if generated and generated["source"] == source_text and generated["settings"] == (int(wpm), int(frequency), int(sample_rate)):
            st.caption(f"Audio duration: {generated['duration']:.1f} seconds")
            st.audio(generated["bytes"], format="audio/wav")
            st.download_button(
                "Download Morse audio",
                data=generated["bytes"],
                file_name="morse-message.wav",
                mime="audio/wav",
                key="download-generated-morse-audio",
            )


st.sidebar.markdown("### MORSE / FIELD DESK")
mode = st.sidebar.radio("Decoder", ("Audio", "Hand gestures", "Written Morse"))
st.sidebar.caption("One workspace for three Morse input methods.")
st.title("Morse Field Desk")

if mode == "Audio":
    render_audio_mode()
elif mode == "Hand gestures":
    render_gesture_mode()
else:
    render_written_morse_mode()