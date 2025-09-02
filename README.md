# Transcrição 0800 (Streamlit + faster-whisper)

App simples para transcrever WAV estéreo (Agente L / Cliente R) gerando `.txt`, `.srt` e `.vtt`.

## Deploy no Streamlit Cloud

Inclua estes arquivos na raiz do repositório:

- `app.py`
- `transcriber.py`
- `requirements.txt`
- `packages.txt`  ← instala deps do PyAV/FFmpeg
- `runtime.txt`   ← fixa Python 3.10
- `.gitignore`

### `packages.txt`
```
ffmpeg
pkg-config
libavcodec-dev
libavdevice-dev
libavfilter-dev
libavformat-dev
libswresample-dev
libswscale-dev
```

### `runtime.txt`
```
3.10
```

## Uso local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
