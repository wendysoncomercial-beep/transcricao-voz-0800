# 🎧 Faster-Whisper Transcriber (Streamlit)

Transcrição rápida de áudios `.wav` usando **faster-whisper**, com separação de canais estéreo (Agente/Cliente),
normalização opcional e exportação em **.txt**, **.srt** e **.vtt** — tudo com interface **Streamlit**.

## 🧰 Como usar localmente

```bash
git clone https://github.com/<SEU_USUARIO>/faster-whisper-streamlit.git
cd faster-whisper-streamlit
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

> ✅ **Sem ffmpeg instalado?** O pacote `imageio-ffmpeg` baixa um binário compatível automaticamente.
> O app adiciona o binário ao `PATH` em tempo de execução.

## 🚀 Deploy no Streamlit Community Cloud

1. Suba estes arquivos no seu GitHub (repo público ou privado).
2. Em **streamlit.io → Deploy app**, aponte para `app.py`.
3. Sem necessidade de `Procfile`. O `requirements.txt` já contempla as dependências.

## 🧪 Dicas de performance

- Modelos menores (ex.: `small`, `medium`) transcrevem mais rápido.
- Em máquinas com GPU CUDA, o app detecta automaticamente e usa `float16`.
- Ative VAD para reduzir trechos de silêncio.

## 📄 Saídas geradas

Para cada arquivo:
- `NOME.txt` → linhas legíveis, com timestamp relativo e (opcional) horário real.
- `NOME.srt` e `NOME.vtt` → legendas com rótulo de falante.
