import streamlit as st
from pathlib import Path
from datetime import datetime
from transcriber import Transcriber
import zipfile
import io

st.set_page_config(page_title="Transcrição 0800", page_icon="🎧", layout="wide")

st.title("🎧 Transcrição 0800 — faster-whisper")
st.caption("Envie WAV estéreo (Agente no L, Cliente no R). Gera .txt, .srt e .vtt.")

with st.sidebar:
    st.header("Configurações")
    model_size = st.selectbox("Modelo", ["large-v3", "medium", "small", "tiny"], index=0)
    lang_hint = st.text_input("Dica de idioma (vazio = auto)", value="pt")
    use_vad = st.checkbox("VAD (corte de silêncio)", value=True)
    vad_min_silence_ms = st.number_input("VAD: silêncio mínimo (ms)", 0, 5000, 600, 50)
    vad_speech_pad_ms  = st.number_input("VAD: padding (ms)", 0, 2000, 200, 50)
    normalize = st.checkbox("Normalizar loudness (loudnorm)", value=False)
    split_channels = st.checkbox("Separar L/R (estéreo)", value=True)
    label_left = st.text_input("Rótulo canal L", value="Agente")
    label_right = st.text_input("Rótulo canal R", value="Cliente")
    start_clock_iso = st.text_input("Início do relógio (ISO, opcional)", value="")
    start_clock = None
    if start_clock_iso.strip():
        try:
            start_clock = datetime.fromisoformat(start_clock_iso.strip())
        except Exception:
            st.warning("START_CLOCK inválido; ignorando.")

st.subheader("Envio de arquivos")
files = st.file_uploader("Selecione um ou mais arquivos .wav", type=["wav"], accept_multiple_files=True)

if "transcriber" not in st.session_state or st.session_state.get("last_model") != model_size:
    with st.status("Carregando modelo...", expanded=False):
        st.session_state["transcriber"] = Transcriber(model_size=model_size)
        st.session_state["last_model"] = model_size

if st.button("Processar", type="primary", disabled=not files):
    out_dir = Path("saida")
    out_dir.mkdir(parents=True, exist_ok=True)
    t = st.session_state["transcriber"]
    all_paths = []

    progress = st.progress(0.0, text="Iniciando...")
    for i, f in enumerate(files, 1):
        wav_path = out_dir / f.name
        with open(wav_path, "wb") as w:
            w.write(f.read())
        labels = (label_left, label_right)
        with st.status(f"Transcrevendo **{f.name}**...", expanded=False):
            outs = t.process_file(
                src=wav_path, out_dir=out_dir, lang_hint=(lang_hint or None),
                normalize=normalize, split_channels=split_channels, labels=labels,
                use_vad=use_vad, vad_min_silence_ms=int(vad_min_silence_ms), vad_speech_pad_ms=int(vad_speech_pad_ms),
                start_clock=start_clock
            )
        all_paths.extend([str(p) for p in outs])
        progress.progress(i / len(files), text=f"Finalizado {i}/{len(files)}")

    # Zipar resultados
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.glob("*")):
            if p.is_file():
                zf.write(p, arcname=p.name)
    mem.seek(0)
    st.success("Pronto! Baixe o pacote zipado com todos os resultados.")
    st.download_button("⬇️ Baixar resultados (.zip)", data=mem, file_name="resultados_transcricao.zip", mime="application/zip")

st.divider()
st.markdown("Dica: se o deploy no Streamlit Cloud falhar por causa do `av`/`PyAV`, verifique **packages.txt** e **runtime.txt** deste repositório.")
