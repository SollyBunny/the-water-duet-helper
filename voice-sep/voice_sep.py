#!/usr/bin/env python3
"""
voice_sep.py - split a duet into two simultaneous singers + instrumental.

Why not diarization?
  Diarization can only say *when* someone sings. Two people singing at the same
  time occupy the same seconds, so time-masking cannot separate them. This uses
  a learned multi-singer separator instead.

Pipeline:
  1. Demucs (htdemucs) extracts the vocal stem and the instrumental.
  2. MedleyVox (Conv-TasNet, 24 kHz) splits the vocal stem into singer_1 and
     singer_2 spectrally. It is trained on overlapping duet/unison singing.
  3. The instrumental stays at 44.1 kHz.

The separator runs on 10 s chunks with 1 s overlap and is roughly 25x faster
than realtime on CPU, so a 4-minute song takes seconds (Demucs is the slow part).

Outputs:
    singer_1.wav             isolated voice 1        (24 kHz)
    singer_2.wav             isolated voice 2        (24 kHz)
    instrumental.wav         backing track           (44.1 kHz)
    diarization.json         levels, F0 and timings

Usage:
    .venv/bin/python voice-sep/voice_sep.py song.opus
    .venv/bin/python voice-sep/voice_sep.py song.opus -o output/song --keep-stems
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch
from asteroid_filterbanks import make_enc_dec
from asteroid.masknn import TDConvNet
from asteroid.models.base_models import (
    BaseEncoderMaskerDecoder,
    _shape_reconstructed,
    _unsqueeze_to_3d,
)
from asteroid.utils.torch_utils import jitable_shape, pad_x_to_y

SR = 24000
MODEL_URL = (
    "https://huggingface.co/Cyru5/MedleyVox/resolve/main/"
    "vocals%20238/vocals.pth"
)

ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


class Cancelled(Exception):
    """Raised when the caller asks the pipeline to stop."""


def log(msg: str) -> None:
    print(msg, flush=True)


# Overall progress weights. Demucs dominates the wall clock on a cold run.
_STAGE_WEIGHTS = {
    "prepare": (0.00, 0.02),
    "demucs": (0.02, 0.70),
    "model": (0.72, 0.03),
    "separator": (0.75, 0.22),
    "writing": (0.97, 0.03),
    "done": (1.00, 0.00),
}


def _emit(on_progress: ProgressCallback | None, stage: str, fraction: float,
          message: str, **extra: Any) -> None:
    if on_progress is None:
        return
    fraction = max(0.0, min(1.0, float(fraction)))
    start, span = _STAGE_WEIGHTS.get(stage, (0.0, 0.0))
    info = {
        "stage": stage,
        "fraction": fraction,
        "overall": min(1.0, start + span * fraction),
        "message": message,
    }
    info.update(extra)
    on_progress(info)


def _check_cancel(should_cancel: CancelCheck | None) -> None:
    if should_cancel is not None and should_cancel():
        raise Cancelled()


# --------------------------------------------------------------------------
# multi-singer model
# --------------------------------------------------------------------------

class MedleyVox(BaseEncoderMaskerDecoder):
    """MedleyVox conv_tasnet_stft wrapper with mixture-consistent output."""

    def forward(self, wav):
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)
        tf_rep = self.forward_encoder(wav)
        masks = self.forward_masker(tf_rep)
        masked = self.apply_masks(tf_rep, masks)
        decoded = self.forward_decoder(masked)
        rec = _shape_reconstructed(pad_x_to_y(decoded, wav), shape)
        rec = rec + 1.0 / rec.shape[1] * (wav - rec.sum(dim=1, keepdim=True))
        return rec


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.urlopen(url)
    total = int(req.headers.get("Content-Length") or 0)
    done = 0
    last_pct = -10
    with open(tmp, "wb") as fh:
        while True:
            block = req.read(1 << 20)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if total:
                pct = int(done * 100 / total)
                if pct >= last_pct + 10:
                    log(f"      {pct:3d}%  ({done / 2**20:.0f} MiB)")
                    last_pct = pct
    tmp.rename(dest)
    log(f"      saved {dest}")


def ensure_model(model_dir: Path) -> Path:
    ckpt = model_dir / "medleyvox_vocals238.pth"
    if ckpt.exists() and ckpt.stat().st_size > 200_000_000:
        return ckpt
    log("[2/3] Downloading MedleyVox weights (~223 MiB)...")
    _download(MODEL_URL, ckpt)
    return ckpt


def load_model(model_dir: Path, device: str) -> MedleyVox:
    ckpt = ensure_model(model_dir)
    encoder, decoder = make_enc_dec(
        "torch_stft", n_filters=2048, kernel_size=2048, stride=512,
        sample_rate=SR,
    )
    masker = TDConvNet(
        in_chan=encoder.n_feats_out, n_src=2, out_chan=None,
        n_blocks=6, n_repeats=4, bn_chan=256, hid_chan=1024,
        skip_chan=256, mask_act="linear",
    )
    model = MedleyVox(encoder, masker, decoder).to(device).eval()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    # use the EMA weights stored alongside the online weights
    state = {k[len("ema_model.module."):]: v
             for k, v in state.items() if k.startswith("ema_model.module.")}
    model.load_state_dict(state, strict=True)
    return model


# --------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------

def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T), sr


def resample(x: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return x
    import torchaudio.functional as AF
    return AF.resample(x, sr_in, sr_out)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    sf.write(str(path), audio, sr, subtype="PCM_16")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def median_f0(x: np.ndarray, sr: int, max_seconds: float = 15.0) -> float | None:
    """Rough median F0 on the loudest excerpt, used only in metadata."""
    if x.size < sr:
        return None
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if mono.size > max_seconds * sr:
        win = int(max_seconds * sr)
        step = int(5 * sr)
        energies = [np.mean(mono[i:i + win] ** 2)
                    for i in range(0, max(1, mono.size - win), step)]
        start = int(np.argmax(energies)) * step
        mono = mono[start:start + win]
    try:
        import torchaudio.functional as AF
        wav = torch.from_numpy(mono).float().unsqueeze(0)
        f0 = AF.detect_pitch_frequency(
            wav, sr, frame_time=0.02, win_length=40,
            freq_low=60, freq_high=1200,
        ).squeeze(0).numpy()
        f0 = f0[f0 > 0]
        return float(np.median(f0)) if f0.size else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Demucs stems
# --------------------------------------------------------------------------

def get_stems(input_path: Path, outdir: Path, model_name: str, device: str,
              reuse: bool,
              on_progress: ProgressCallback | None = None,
              should_cancel: CancelCheck | None = None,
              ) -> tuple[torch.Tensor, torch.Tensor, int]:
    stem_dir = outdir / "stems"
    vocals_path = stem_dir / "vocals.wav"
    instru_path = stem_dir / "instrumental.wav"

    _check_cancel(should_cancel)
    if reuse and vocals_path.exists() and instru_path.exists():
        _emit(on_progress, "demucs", 1.0, "Reusing cached Demucs stems")
        vocals, sr = load_audio(vocals_path)
        instru, _ = load_audio(instru_path)
        return vocals, instru, sr

    from demucs.api import Separator
    stem_dir.mkdir(parents=True, exist_ok=True)
    _emit(on_progress, "demucs", 0.0, f"Loading Demucs '{model_name}'")
    sep = Separator(model=model_name, device=device, shifts=1, overlap=0.25,
                    split=True, progress=False, callback=None)

    segment_seconds = getattr(sep.model, "segment", None) or 7.8
    segment_frames = max(1, int(segment_seconds * sep.samplerate))

    def demucs_progress(info: dict[str, Any]) -> None:
        _check_cancel(should_cancel)
        length = max(1, int(info.get("audio_length") or 1))
        offset = int(info.get("segment_offset") or 0)
        done = offset + (segment_frames if info.get("state") == "end"
                         else segment_frames / 2)
        _emit(on_progress, "demucs", min(1.0, done / length),
              "Separating vocals and instrumental",
              segment_offset=offset, audio_length=length)

    sep.update_parameter(callback=demucs_progress)
    _emit(on_progress, "demucs", 0.0, "Separating vocals and instrumental")
    _, stems = sep.separate_audio_file(str(input_path))
    _check_cancel(should_cancel)
    sr = sep.samplerate
    vocals = stems["vocals"].cpu()
    instru = stems.get("no_vocals")
    if instru is None:
        original, orig_sr = load_audio(input_path)
        if original.shape[0] == 1:
            original = original.repeat(2, 1)
        original = resample(original, orig_sr, sr)
        shortest = min(original.shape[-1], vocals.shape[-1])
        original = original[..., :shortest]
        vocals = vocals[..., :shortest]
        instru = original - vocals
    if vocals.shape[0] == 1:
        vocals = vocals.repeat(2, 1)
        instru = instru.repeat(2, 1)

    write_wav(vocals_path, vocals.numpy().T, sr)
    write_wav(instru_path, instru.numpy().T, sr)
    _emit(on_progress, "demucs", 1.0, "Demucs separation complete")
    return vocals, instru, sr


# --------------------------------------------------------------------------
# chunked separation
# --------------------------------------------------------------------------

def _align(est: np.ndarray, prev_tail: np.ndarray | None) -> np.ndarray:
    """Keep singer identity consistent across chunks."""
    if prev_tail is None or prev_tail.shape[1] == 0:
        return est
    ov = min(prev_tail.shape[1], est.shape[1])
    a, b = est[:, :ov], prev_tail[:, -ov:]
    if min(rms(a[0]), rms(a[1])) < 1e-4:
        return est
    ident = np.corrcoef(a[0], b[0])[0, 1] + np.corrcoef(a[1], b[1])[0, 1]
    swap = np.corrcoef(a[1], b[0])[0, 1] + np.corrcoef(a[0], b[1])[0, 1]
    if np.isfinite(swap) and swap > ident + 0.03:
        return est[::-1].copy()
    return est


def separate_singers(model: MedleyVox, wav: torch.Tensor, sr: int,
                     chunk_seconds: float, overlap_seconds: float,
                     device: str,
                     on_progress: ProgressCallback | None = None,
                     should_cancel: CancelCheck | None = None,
                     ) -> np.ndarray:
    """Return (2, N) singers from a mono 24 kHz vocal stem."""
    _check_cancel(should_cancel)
    n = wav.numel()
    if chunk_seconds <= 0 or n <= int(chunk_seconds * sr):
        _emit(on_progress, "separator", 0.0, "Separating singers")
        with torch.inference_mode():
            out = model(wav.unsqueeze(0).unsqueeze(0).to(device))[0]
        _emit(on_progress, "separator", 1.0, "Separated singers")
        return out.cpu().numpy()

    chunk = int(chunk_seconds * sr)
    overlap = int(overlap_seconds * sr)
    step = max(1, chunk - overlap)
    starts = list(range(0, n, step))
    total = len(starts)
    out = np.zeros((2, n), dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    prev_tail: np.ndarray | None = None
    t0 = time.monotonic()

    for i, start in enumerate(starts):
        _check_cancel(should_cancel)
        end = min(n, start + chunk)
        seg = wav[start:end]
        with torch.inference_mode():
            est = model(seg.unsqueeze(0).unsqueeze(0).to(device))[0].cpu().numpy()
        est = _align(est, prev_tail)

        w = np.ones(est.shape[1], dtype=np.float32)
        f = min(overlap, est.shape[1] // 2)
        if f > 0:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f, dtype=np.float32))
            if start > 0:
                w[:f] *= ramp
            if end < n:
                w[-f:] *= ramp[::-1]
        out[:, start:end] += est * w
        weight[start:end] += w
        prev_tail = est[:, -overlap:] if overlap > 0 else None

        done = i + 1
        elapsed = time.monotonic() - t0
        eta = (elapsed / done) * (total - done)
        _emit(on_progress, "separator", done / total,
              f"Separating singers (chunk {done}/{total})",
              chunks_done=done, chunks_total=total,
              elapsed_seconds=elapsed, eta_seconds=eta)
        log(f"      chunk {done}/{total}  elapsed {elapsed:4.0f}s  eta {eta:4.0f}s")

    _check_cancel(should_cancel)
    return out / np.maximum(weight, 1e-8)


# --------------------------------------------------------------------------
# full pipeline
# --------------------------------------------------------------------------

def run_pipeline(input_path: Path, outdir: Path, *,
                 demucs_model: str = "htdemucs",
                 device: str = "cpu",
                 model_dir: Path | None = None,
                 chunk_seconds: float = 10.0,
                 overlap_seconds: float = 1.0,
                 keep_stems: bool = False,
                 threads: int | None = None,
                 model: MedleyVox | None = None,
                 on_progress: ProgressCallback | None = None,
                 should_cancel: CancelCheck | None = None,
                 ) -> dict[str, Any]:
    """Run the full separation pipeline.

    ``on_progress`` is called from the worker thread with a dict containing
    ``stage``, ``fraction``, ``overall``, ``message`` and stage-specific fields.
    ``should_cancel`` is polled between stages/chunks; raise-free callers get
    :class:`Cancelled` when it returns true.
    """
    t0 = time.time()
    input_path = Path(input_path)
    outdir = Path(outdir)
    model_dir = Path(model_dir) if model_dir is not None else (
        Path(__file__).resolve().parent / "models")
    outdir.mkdir(parents=True, exist_ok=True)

    if threads:
        torch.set_num_threads(threads)
    _check_cancel(should_cancel)

    _emit(on_progress, "prepare", 0.0, "Preparing")
    vocals, instru, sr = get_stems(
        input_path.resolve(), outdir, demucs_model, device,
        reuse=True, on_progress=on_progress, should_cancel=should_cancel,
    )

    _check_cancel(should_cancel)
    _emit(on_progress, "model", 0.0, "Loading MedleyVox multi-singer model")
    if model is None:
        model = load_model(model_dir, device)
    _emit(on_progress, "model", 1.0, "MedleyVox model ready")

    _check_cancel(should_cancel)
    vocals_24k = resample(vocals.mean(dim=0, keepdim=True), sr, SR)
    singers = separate_singers(model, vocals_24k.squeeze(0), SR,
                               chunk_seconds, overlap_seconds, device,
                               on_progress=on_progress,
                               should_cancel=should_cancel)

    _check_cancel(should_cancel)
    _emit(on_progress, "writing", 0.0, "Writing tracks")

    write_wav(outdir / "instrumental.wav", instru.numpy().T, sr)

    meta: dict[str, Any] = {
        "input": str(input_path),
        "output_dir": str(outdir),
        "demucs_model": demucs_model,
        "separator": "MedleyVox Conv-TasNet (EMA weights)",
        "sample_rate": SR,
        "duration_seconds": round(singers.shape[1] / SR, 3),
        "singers": [],
        "elapsed_seconds": None,
    }

    for i in range(singers.shape[0]):
        _check_cancel(should_cancel)
        voice = singers[i:i + 1]
        base = f"singer_{i + 1}"
        write_wav(outdir / f"{base}.wav", voice, SR)
        entry = {"file": f"{base}.wav", "rms": round(rms(voice), 5)}
        f0 = median_f0(voice.T, SR)
        if f0:
            entry["median_f0_hz"] = round(f0, 1)
        meta["singers"].append(entry)
        log(f"      {base}: rms={entry['rms']:.4f}"
            + (f", median f0={f0:.1f} Hz" if f0 else ""))

    if not keep_stems:
        for p in (outdir / "stems").glob("*"):
            p.unlink()
        (outdir / "stems").rmdir()

    meta["elapsed_seconds"] = round(time.time() - t0, 1)
    (outdir / "diarization.json").write_text(json.dumps(meta, indent=2) + "\n")
    _emit(on_progress, "done", 1.0, "Done", elapsed_seconds=meta["elapsed_seconds"])
    return meta


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

class _CliProgress:
    def __init__(self) -> None:
        self.last_demucs_pct = -10
        self.model_started = False
        self.writing_started = False

    def __call__(self, info: dict[str, Any]) -> None:
        stage = info["stage"]
        if stage == "demucs":
            if info["message"].startswith("Reusing"):
                log(f"[1/3] {info['message']}")
            elif info["fraction"] >= 1.0:
                log("[1/3] Demucs separation complete")
            else:
                pct = int(info["fraction"] * 100)
                if pct >= self.last_demucs_pct + 10:
                    self.last_demucs_pct = pct
                    log(f"[1/3] {info['message']} {pct:3d}%")
        elif stage == "model" and not self.model_started:
            self.model_started = True
            log("[2/3] Loading MedleyVox multi-singer model...")
        elif stage == "separator":
            # separate_singers already logs a human-readable chunk line.
            pass
        elif stage == "writing" and not self.writing_started:
            self.writing_started = True
            log("[3/3] Writing tracks...")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split a duet into two singers plus instrumental.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="audio file (mp3/wav/flac/opus/...)")
    ap.add_argument("-o", "--outdir", type=Path, default=None,
                    help="output directory (default: output/<track>)")
    ap.add_argument("-n", "--demucs-model", default="htdemucs",
                    help="Demucs model: htdemucs, htdemucs_ft, mdx_extra_q, ...")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--model-dir", type=Path,
                    default=Path(__file__).resolve().parent / "models")
    ap.add_argument("--chunk-seconds", type=float, default=10.0,
                    help="separator chunk length (0 = one pass)")
    ap.add_argument("--overlap-seconds", type=float, default=1.0,
                    help="overlap between separator chunks")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--no-reuse", action="store_true",
                    help="re-run Demucs even if cached stems exist")
    ap.add_argument("--keep-stems", action="store_true",
                    help="keep the 44.1 kHz vocal/instrumental stems")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if args.device == "cuda" and not torch.cuda.is_available():
        log("CUDA requested but not available; using CPU.")
        args.device = "cpu"

    outdir = args.outdir or Path("output") / args.input.stem
    outdir = Path(str(outdir).replace(" ", "-"))
    outdir.mkdir(parents=True, exist_ok=True)

    log(f"Input : {args.input}")
    log(f"Output: {outdir}")

    if args.no_reuse:
        for name in ("vocals.wav", "instrumental.wav"):
            (outdir / "stems" / name).unlink(missing_ok=True)

    meta = run_pipeline(
        args.input, outdir,
        demucs_model=args.demucs_model,
        device=args.device,
        model_dir=args.model_dir,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
        keep_stems=args.keep_stems,
        threads=args.threads,
        on_progress=_CliProgress(),
    )
    log(f"\nDone in {meta['elapsed_seconds']:.0f}s -> {outdir}")


if __name__ == "__main__":
    main()
