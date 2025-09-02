from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterable, Optional, Tuple, List
import ffmpeg  # ffmpeg-python
from faster_whisper import WhisperModel
import imageio_ffmpeg
import platform

# Garantir ffmpeg no PATH via imageio-ffmpeg
def _ensure_ffmpeg_on_path():
    try:
        ffbin = imageio_ffmpeg.get_ffmpeg_exe()
        bindir = str(Path(ffbin).parent)
        if bindir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass

_ensure_ffmpeg_on_path()

def fmt_rel_ts(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def fmt_wall_ts(start_clock: Optional[datetime], offset_s: float) -> Optional[str]:
    if not start_clock:
        return None
    ts = start_clock + timedelta(seconds=offset_s)
    tz = ts.strftime("%z"); tz_fmt = f"{tz[:3]}:{tz[3:]}" if tz else ""
    return f"{ts.strftime('%H:%M:%S')} {tz_fmt}".strip()

def write_outputs(base_out: Path, segments, speaker: Optional[str] = None, start_clock: Optional[datetime] = None):
    txt_lines, srt_lines, vtt_lines = [], [], ["WEBVTT\n"]
    idx = 1
    for seg in segments:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        rel_start = fmt_rel_ts(seg.start); rel_end = fmt_rel_ts(seg.end)
        wall = fmt_wall_ts(start_clock, seg.start)
        rel_hdr = rel_start[:8]; wall_hdr = f" | {wall}" if wall else ""
        spk = (speaker or "Speaker").strip()
        header_txt = f"[{rel_hdr}{wall_hdr}] {spk}: "
        full_txt = f"{header_txt}{text}"
        txt_lines.append(full_txt)
        srt_text_line = f"{spk}{' ['+wall+']' if wall else ''}: {text}"
        srt_lines += [str(idx), f"{rel_start} --> {rel_end}", srt_text_line, ""]
        vtt_lines += [f"{rel_start.replace(',', '.')} --> {rel_end.replace(',', '.')}", srt_text_line, ""]
        idx += 1
    (base_out.with_suffix(".txt")).write_text("\n".join(txt_lines), encoding="utf-8")
    (base_out.with_suffix(".srt")).write_text("\n".join(srt_lines), encoding="utf-8")
    (base_out.with_suffix(".vtt")).write_text("\n".join(vtt_lines), encoding="utf-8")

def is_stereo(wav_path: Path) -> bool:
    try:
        info = ffmpeg.probe(str(wav_path))
        for st in info.get("streams", []):
            if st.get("codec_type") == "audio":
                return int(st.get("channels", 1)) == 2
    except Exception:
        pass
    return False

def split_stereo(in_wav: Path, left_out: Path, right_out: Path):
    (
        ffmpeg
        .input(str(in_wav))
        .output(str(left_out), ac=1, map="0:a:0")
        .overwrite_output().run(quiet=True)
    )
    (
        ffmpeg
        .input(str(in_wav))
        .output(str(right_out), ac=1, map="0:a:1")
        .overwrite_output().run(quiet=True)
    )

def loudnorm(in_wav: Path, out_wav: Path, i_lufs: str = "-16", tp_db: str = "-1.5", lra: str = "11"):
    (
        ffmpeg
        .input(str(in_wav))
        .output(str(out_wav), af=f"loudnorm=I={i_lufs}:TP={tp_db}:LRA={lra}")
        .overwrite_output().run(quiet=True)
    )

class Transcriber:
    def __init__(self, model_size: str = "large-v3"):
        device = "cpu"; compute_type = "int8"
        try:
            import torch  # optional
            if torch.cuda.is_available():
                device = "cuda"; compute_type = "float16"
        except Exception:
            pass
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcrever_um(
        self,
        wav_path: Path, out_dir: Path, lang_hint: Optional[str],
        use_vad: bool, vad_min_silence_ms: int, vad_speech_pad_ms: int,
        beam_size: int, best_of: int, temperatures: List[float],
        compression_ratio_threshold: float, log_prob_threshold: float,
        no_speech_threshold: float, condition_on_previous_text: bool,
        speaker_label: Optional[str], start_clock: Optional[datetime],
    ) -> Tuple[Path, Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        base_out = out_dir / wav_path.stem
        kwargs = dict(
            language=None if not lang_hint else lang_hint,
            vad_filter=use_vad, beam_size=beam_size, best_of=best_of,
            temperature=temperatures, compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold, no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text, word_timestamps=False,
        )
        if use_vad:
            kwargs["vad_parameters"] = dict(min_silence_duration_ms=vad_min_silence_ms, speech_pad_ms=vad_speech_pad_ms)
        segments, info = self.model.transcribe(str(wav_path), **kwargs)
        write_outputs(base_out, segments, speaker=speaker_label, start_clock=start_clock)
        return base_out.with_suffix(".txt"), base_out.with_suffix(".srt"), base_out.with_suffix(".vtt")

    def process_file(
        self,
        src: Path, out_dir: Path, lang_hint: Optional[str] = "pt",
        normalize: bool = False, split_channels: bool = True,
        labels: Tuple[str, str] = ("Agente", "Cliente"),
        use_vad: bool = True, vad_min_silence_ms: int = 600, vad_speech_pad_ms: int = 200,
        beam_size: int = 8, best_of: int = 5, temperatures: List[float] = [0.0, 0.2, 0.4, 0.6],
        compression_ratio_threshold: float = 2.4, log_prob_threshold: float = -1.2,
        no_speech_threshold: float = 0.45, condition_on_previous_text: bool = False,
        start_clock: Optional[datetime] = None,
    ) -> List[Path]:
        outputs: List[Path] = []
        work = src
        tmp = []

        if normalize:
            norm = out_dir / f"{src.stem}.norm.tmp.wav"
            try:
                loudnorm(src, norm)
                work = norm; tmp.append(norm)
            except Exception:
                pass

        if split_channels and is_stereo(work):
            L = out_dir / f"{src.stem}.L.tmp.wav"
            R = out_dir / f"{src.stem}.R.tmp.wav"
            try:
                split_stereo(work, L, R)
                # Left
                outputs += list(self.transcrever_um(
                    L, out_dir, lang_hint, use_vad, vad_min_silence_ms, vad_speech_pad_ms,
                    beam_size, best_of, temperatures, compression_ratio_threshold,
                    log_prob_threshold, no_speech_threshold, condition_on_previous_text,
                    labels[0], start_clock,
                ))
                # Right
                outputs += list(self.transcrever_um(
                    R, out_dir, lang_hint, use_vad, vad_min_silence_ms, vad_speech_pad_ms,
                    beam_size, best_of, temperatures, compression_ratio_threshold,
                    log_prob_threshold, no_speech_threshold, condition_on_previous_text,
                    labels[1], start_clock,
                ))
                tmp += [L, R]
            except Exception:
                outputs += list(self.transcrever_um(
                    work, out_dir, lang_hint, use_vad, vad_min_silence_ms, vad_speech_pad_ms,
                    beam_size, best_of, temperatures, compression_ratio_threshold,
                    log_prob_threshold, no_speech_threshold, condition_on_previous_text,
                    None, start_clock,
                ))
        else:
            outputs += list(self.transcrever_um(
                work, out_dir, lang_hint, use_vad, vad_min_silence_ms, vad_speech_pad_ms,
                beam_size, best_of, temperatures, compression_ratio_threshold,
                log_prob_threshold, no_speech_threshold, condition_on_previous_text,
                None, start_clock,
            ))

        for p in tmp:
            try: p.unlink()
            except Exception: pass
        return outputs
