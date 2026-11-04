# Duet Helper

Split a duet into two simultaneous singers plus an instrumental, and play the
result in the browser.

## Layout

```
client/          static browser client (served at /)
output/<name>/   generated tracks + optional lyrics.txt (served at /output/<name>/...)
voice-sep/       Demucs + MedleyVox separation pipeline and model weights
server.py        HTTP + WebSocket server
.venv/           Python environment
requirements.txt
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch torchaudio
uv pip install --python .venv/bin/python -r requirements.txt
```

Models download automatically on first run:

- Demucs `htdemucs` → PyTorch hub cache (~80 MB)
- MedleyVox `vocals 238` EMA checkpoint → `voice-sep/models/medleyvox_vocals238.pth`
  (~223 MB, from `Cyru5/MedleyVox` on Hugging Face)

## Run

```bash
.venv/bin/python server.py --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/>. The client loads `output/the-water` by default;
use `?output=<name>` to load another output. Lyrics are read from
`output/<name>/lyrics.txt` when present.

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | `client/index.html` |
| `GET` | `/<path>` | static file from `client/` |
| `GET` | `/output/<name>/<path>` | static file from `output/<name>/` |
| `GET` | `/output/<name>/index.json` | generated track manifest |
| `GET` | `/api/ws` | WebSocket state/progress stream |
| `POST`/`PUT`/`PUSH` | `/api/output/<name>` | upload source audio and queue it |
| `DELETE` | `/api/output/<name>` | stop processing and delete the output |

Upload the file as the raw request body (`Content-Type: audio/mpeg`, etc.), or
as a `multipart/form-data` part named `file`.

### WebSocket messages

```json
{"type": "state", "outputs": [...]}
{"type": "add", "output": {...}}
{"type": "update", "output": {...}}
{"type": "progress", "name": "...", "status": "...", "progress": {...}}
{"type": "delete", "name": "..."}
```

Each output object contains `name`, `status`, `progress`, `error`, `meta`,
`files`, `index`, `created`, and `updated`. `status` is one of `queued`,
`processing`, `deleting`, `complete`, `error`, or `incomplete`.

## CLI

The separation pipeline can still be run directly:

```bash
.venv/bin/python voice-sep/voice_sep.py "song.opus" -o output/song
```

## Credits

- [Demucs](https://github.com/facebookresearch/demucs) (MIT)
- [MedleyVox](https://github.com/jeonchangbin49/MedleyVox) — ICASSP 2023,
  *MedleyVox: An Evaluation Dataset for Multiple Singing Voices Separation*.
  Checkpoints trained by [Cyru5](https://huggingface.co/Cyru5/MedleyVox).
