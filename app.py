import os
from pathlib import Path
from datetime import datetime
import streamlit as st
from transcriber import Transcriber

st.set_page_config(page_title="Faster-Whisper Transcriber", page_icon="🎧", layout="centered")

st.title("🎧 Faster-Whisper Transcriber")
st.caption("Separe canais (Agente/Cliente), normalize, VAD e exporte .txt / .srt / .vtt")

with st.sidebar:
    st.header("⚙️ Configurações")
    model_size = st.selectbox("Modelo", ["tiny", "base", "small", "medium", "large-v3"], index=4, key="k_model")
    lang_hint = st.text_input("Idioma (vazio = autodetect)", value="pt", key="k_lang")
    normalize = st.checkbox("Normalizar (loudness)", value=False, key="k_norm")
    split_channels = st.checkbox("Separar canais estéreo (L=Agente / R=Cliente)", value=True, key="k_split")
    label_left = st.text_input("Rótulo canal L", value="Agente", key="k_labL")
    label_right = st.text_input("Rótulo canal R", value="Cliente", key="k_labR")
    use_vad = st.checkbox("Ativar VAD (corta silêncios)", value=True, key="k_vad")
    vad_min_silence_ms = st.number_input("VAD: silêncio mínimo (ms)", value=600, step=100, key="k_vad_min")
    vad_speech_pad_ms = st.number_input("VAD: padding fala (ms)", value=200, step=50, key="k_vad_pad")
    beam_size = st.number_input("Beam size", value=8, min_value=1, max_value=16, step=1, key="k_beam")
    best_of = st.number_input("Best of", value=5, min_value=1, max_value=10, step=1, key="k_bestof")
    start_clock_iso = st.text_input("Início relógio real (ISO, opcional)", value="", key="k_startiso")

    st.divider()
    st.write("💡 Dica: use modelos menores p/ ganho de velocidade.")

with st.form(key="proc_form"):
    uploaded = st.file_uploader("Envie um ou mais arquivos .wav", type=["wav"], accept_multiple_files=True, key="wav_uploader")
    submit = st.form_submit_button("🚀 Processar")
out_dir = Path("saida")
out_dir.mkdir(exist_ok=True)

if submit and uploaded:
    trans = Transcriber(model_size=model_size)
    st.info(f"Dispositivo: {trans.device} | compute_type: {trans.compute_type}")

    start_clock = None
    if start_clock_iso := start_clock_iso.strip():
        try:
            start_clock = datetime.fromisoformat(start_clock_iso)
        except Exception as e:
            st.warning(f"START_CLOCK_ISO inválido: {e}")

    for up in uploaded:
        # Salvar o arquivo temporariamente
        src_path = out_dir / up.name
        with open(src_path, "wb") as f:
            f.write(up.read())
        st.write(f"➡️ **Processando**: `{up.name}`")

        outputs = trans.process_file(
            src=src_path, out_dir=out_dir, lang_hint=(lang_hint or None),
            normalize=normalize, split_channels=split_channels,
            labels=(label_left, label_right),
            use_vad=use_vad, vad_min_silence_ms=int(vad_min_silence_ms), vad_speech_pad_ms=int(vad_speech_pad_ms),
            beam_size=int(beam_size), best_of=int(best_of),
            start_clock=start_clock,
        )

        # Mostrar botões de download
        for path in outputs:
            with open(path, "rb") as fh:
                st.download_button(
    label=f"⬇️ Baixar {path.name}",
    data=fh.read(),
    file_name=path.name,
    mime="text/plain" if path.suffix == ".txt" else "text/vtt",
    key=f"dl-{path.name}-{str(path.stat().st_mtime_ns)}"
)
        st.success(f"✅ Finalizado: {up.name}")

st.markdown("---")
st.markdown("Made with ❤️ + faster-whisper")
