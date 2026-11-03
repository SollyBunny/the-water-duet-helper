# voice-sep

Split a song into **two simultaneous singers + instrumental**.

Most vocal separators (Demucs, Spleeter, UVR) return one combined vocal stem.
Speaker diarization can only tell you *when* someone sings, so it cannot split
a duet where both voices overlap. This project uses a learned multi-singer
separator for the voices and Demucs for the backing track.

## Pipeline

1. **Demucs** (`htdemucs`) extracts the vocal stem and the instrumental.
2. **MedleyVox** (Conv-TasNet, 24 kHz) splits the vocal stem into `singer_1`
   and `singer_2` spectrally. It is trained on overlapping duet/unison singing.
3. The instrumental stays at 44.1 kHz.

The separator runs on 10 s chunks with 1 s overlap and is roughly 25x faster
than realtime on CPU. A 4-minute track takes about **10 seconds** once Demucs
stems are cached (the first Demucs pass is the slow part, ~2-3 min).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch torchaudio
uv pip install --python .venv/bin/python \
  demucs soundfile asteroid-filterbanks asteroid
```

Models download automatically on first run:

- Demucs `htdemucs` → PyTorch hub cache (~80 MB)
- MedleyVox `vocals 238` EMA checkpoint → `models/medleyvox_vocals238.pth`
  (~223 MB, from `Cyru5/MedleyVox` on Hugging Face)

## Usage

```bash
.venv/bin/python voice-sep/voice_sep.py "song.opus"
.venv/bin/python voice-sep/voice_sep.py "song.mp3" -o output/song
```

Useful options:

```bash
--chunk-seconds 10    # separator chunk length (0 = one pass)
--overlap-seconds 1
--demucs-model htdemucs_ft   # better vocals, slower
--keep-stems          # keep 44.1 kHz vocals/instrumental
--threads N
```

## Outputs

```
output/<track>/
  singer_1.wav             isolated voice 1        (24 kHz, mono)
  singer_2.wav             isolated voice 2        (24 kHz, mono)
  instrumental.wav         backing track           (44.1 kHz, stereo)
  diarization.json         run metadata, RMS/F0 levels
  stems/
    vocals.wav             combined vocal stem     (44.1 kHz, stereo)
    instrumental.wav
```

`diarization.json` reports the median F0 of each output, which helps identify
which file is the lower/higher voice. The two singer stems sum exactly to the
vocal stem by construction.

## Notes and limitations

- Singer stems are 24 kHz because that is MedleyVox's native rate. The
  instrumental is not downsampled.
- Truly overlapping **unison** singing is the hardest case; the model still
  returns two streams, but leakage between them is expected.
- The separator sees the Demucs vocal stem, so any Demucs bleed ends up in the
  singer stems.
- Chunk order is aligned by correlation over the overlap. If a singer is
  completely silent across a boundary, identity could in principle flip; the
  F0 values in the metadata let you check.

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) (MIT)
- [MedleyVox](https://github.com/jeonchangbin49/MedleyVox) — ICASSP 2023,
  *MedleyVox: An Evaluation Dataset for Multiple Singing Voices Separation*.
  Checkpoints trained by [Cyru5](https://huggingface.co/Cyru5/MedleyVox).
