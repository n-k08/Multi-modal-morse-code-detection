# Morse Field Desk

One Streamlit app combining the workspace's audio Morse decoder, hand-gesture decoder, and written Morse text decoder.

## Run

Use Python 3.10 or newer and install into a project virtual environment:

```powershell
py -3.10 -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

The app opens in your browser. Select an input mode from the sidebar.

## Deploy to Streamlit Community Cloud

Create an app from `n-k08/Multi-modal-morse-code-detection`, branch `main`, with `streamlit_app.py` as the entrypoint. In **Advanced settings**, select **Python 3.12**; `mediapipe==0.10.21` does not publish a Python 3.14 wheel. The root `requirements.txt` installs Python dependencies and `packages.txt` installs the PortAudio runtime used by the audio decoder modules.

## Modes

- **Audio:** decode live browser-microphone input or upload and play WAV, FLAC, OGG, AIFF, MP3, AAC, M4A, MP4, or WMA. Configure WPM, optional Farnsworth spacing, tone frequency tracking, filter Q, signal threshold, and volume limits. Live and file results show word/character/speed stats and support copying or downloading the transcript. Decoding runs locally without an external AI service.
- **Hand gestures:** allow browser webcam access. A closed hand sends a dot; the configured number of extended fingers sends a dash. Briefly lower the hand between identical symbols. Letters are completed after a pause.
- **Text and Morse:** decode written Morse into text, or enter text to generate its Morse representation and playable/downloadable audio. Encoding supports letters, numbers, and punctuation; unsupported characters are reported.

Live hand-gesture webcam access requires a browser that supports camera permissions. Deployed sites must use HTTPS; restrictive networks may require a configured TURN server.

Compressed formats use the FFmpeg codecs bundled with PyAV; no separate FFmpeg installation is required for supported codecs.