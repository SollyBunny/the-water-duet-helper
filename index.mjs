/**
 * @typedef {{
 *   path: string,
 *   canvas: HTMLCanvasElement,
 *   context: CanvasRenderingContext2D,
 *   head: HTMLDivElement,
 *   audio: HTMLAudioElement,
 *   peaks: Float32Array | null,
 * }} Track
 */

/** @type {Track[]} */
const tracks = [];

// Used only to decode audio for the waveforms; playback goes through the media elements.
const audioContext = new (window.AudioContext || window.webkitAudioContext)();

const tracksEl = document.getElementById("tracks");
const playButton = document.getElementById("player-button");
const seekSlider = document.getElementById("player-seek");
const timeEl = document.getElementById("player-time");

let duration = 0;
let playing = false;
/** Playback position while paused, and where playback starts from. */
let offset = 0;
let frame = 0;

const fmt = (seconds) => {
    seconds = Math.max(0, Math.floor(seconds || 0));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

/** @param {AudioBuffer} buffer */
function computePeaks(buffer, count = 2048) {
    const data = buffer.getChannelData(0);
    const peaks = new Float32Array(count);
    const block = data.length / count;

    for (let i = 0; i < count; i++) {
        const start = Math.floor(i * block);
        const end = Math.min(data.length, Math.floor((i + 1) * block));
        let peak = 0;

        for (let j = start; j < end; j++) {
            const value = data[j];
            const abs = value < 0 ? -value : value;
            if (abs > peak) peak = abs;
        }

        peaks[i] = peak;
    }

    return peaks;
}

/** @param {Track} track */
function drawPeaks(track) {
    const { canvas, context: ctx, peaks } = track;
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);

    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!peaks) return;

    const mid = height / 2;
    // One bar per ~3 CSS pixels keeps the waveform readable on any width.
    const bars = Math.max(1, Math.floor(width / 3));
    const stride = peaks.length / bars;
    const barWidth = Math.max(1, (width / bars) * 0.72);
    const barSpace = width / bars;

    ctx.fillStyle = "#8495ff";

    for (let i = 0; i < bars; i++) {
        const start = Math.floor(i * stride);
        const end = Math.min(peaks.length, Math.max(start + 1, Math.floor((i + 1) * stride)));
        let peak = 0;

        for (let j = start; j < end; j++) {
            if (peaks[j] > peak) peak = peaks[j];
        }

        const amp = Math.max(1, peak * (height * 0.44));
        ctx.fillRect(i * barSpace, mid - amp, barWidth, amp * 2);
    }
}

function updateDuration() {
    duration = tracks.reduce((max, track) => {
        const trackDuration = track.audio.duration;
        return Number.isFinite(trackDuration) ? Math.max(max, trackDuration) : max;
    }, 0);
}

/**
 * Fetch and decode a track purely to draw its waveform. Playback uses the
 * audio element directly, and the decoded buffer is released once the peaks
 * are computed.
 *
 * @param {Track} track
 */
async function loadWaveform(track) {
    const res = await fetch(track.path);
    const buffer = await audioContext.decodeAudioData(await res.arrayBuffer());
    track.peaks = computePeaks(buffer);
    drawPeaks(track);
}

/**
 * Resolve once the audio element has metadata (duration), without waiting for
 * the audio itself to load.
 *
 * @param {Track} track
 */
function loadTrackMetadata(track) {
    const { audio } = track;
    if (audio.readyState >= HTMLMediaElement.HAVE_METADATA) return Promise.resolve();

    return new Promise((resolve) => {
        audio.addEventListener("loadedmetadata", resolve, { once: true });
    });
}

function currentTime() {
    if (!playing) return offset;
    // The longest track drives the playhead; shorter tracks stop at their end.
    return Math.min(duration, Math.max(0, ...tracks.map((track) => track.audio.currentTime || 0)));
}

/** @param {number} time */
function setPositions(time) {
    for (const track of tracks) {
        const { audio } = track;
        if (!Number.isFinite(audio.duration)) continue;
        audio.currentTime = clamp(time, 0, audio.duration);
    }
}

/** @param {number} [at] Position to settle on; defaults to the current playback time. */
function pause(at = currentTime()) {
    offset = at;
    playing = false;
    cancelAnimationFrame(frame);
    for (const track of tracks) {
        track.audio.pause();
    }
    updateButton();
    render();
}

function updateButton() {
    playButton.textContent = playing ? "Pause" : "Play";
    playButton.setAttribute("aria-pressed", String(playing));
}

/** @param {number} time */
function seek(time) {
    offset = clamp(time, 0, duration);

    if (offset >= duration) {
        pause(duration);
        return;
    }

    setPositions(offset);
    render();
}

function render() {
    const time = currentTime();
    const ratio = duration ? clamp(time / duration, 0, 1) : 0;

    seekSlider.value = String(Math.round(ratio * Number(seekSlider.max)));
    if (timeEl) timeEl.textContent = `${fmt(time)} / ${fmt(duration)}`;

    for (const track of tracks) {
        track.head.style.left = `${ratio * 100}%`;
    }
}

function tick() {
    if (!playing) return;

    if (currentTime() >= duration) {
        pause(duration);
        return;
    }

    render();
    frame = requestAnimationFrame(tick);
}

function play() {
    if (playing || duration <= 0) return;

    if (offset >= duration) offset = 0;
    setPositions(offset);
    playing = true;
    updateButton();
    cancelAnimationFrame(frame);

    for (const track of tracks) {
        track.audio.play();
    }

    tick();
}

playButton.addEventListener("click", () => {
    if (playing)
        pause();
    else
        play();
});

let resumeAfterSeek = undefined;
seekSlider.addEventListener("input", () => {
    if (resumeAfterSeek === undefined) {
        if (playing) {
            resumeAfterSeek = true;
            pause();
        } else {
            resumeAfterSeek = false;
        }
    }
    seek(Number(seekSlider.value) / Number(seekSlider.max) * duration);
});

seekSlider.addEventListener("change", () => {
    if (resumeAfterSeek) {
        play();
    }
    resumeAfterSeek = undefined;
})

async function loadTracks() {
    const res = await fetch("./data/index.json");
    const manifest = await res.json();
    const entries = Object.entries(manifest.files ?? manifest);
    const fragment = document.createDocumentFragment();

    for (let [name, path] of entries) {
        path = "./data/" + path;

        const row = document.createElement("div");
        row.classList.add("track");

        const elInfo = document.createElement("div");
        elInfo.classList.add("track-info");
        row.append(elInfo);

        const elName = document.createElement("span");
        elName.classList.add("track-name");
        elName.textContent = name;
        elInfo.append(elName);

        const elPath = document.createElement("span");
        elPath.classList.add("track-path");
        elPath.textContent = path;
        elInfo.append(elPath);

        const elVolume = document.createElement("input");
        elVolume.classList.add("track-volume");
        elVolume.type = "range";
        elVolume.min = "0";
        elVolume.max = "100";
        elVolume.value = "100";
        elVolume.setAttribute("aria-label", `${name} volume`);
        row.append(elVolume);

        const elWave = document.createElement("div");
        elWave.classList.add("track-wave");
        row.append(elWave);

        const elPeaks = document.createElement("canvas");
        elPeaks.classList.add("track-peaks");
        elPeaks.textContent = "Your browser doesn't support canvas";
        elWave.append(elPeaks);

        const elHead = document.createElement("div");
        elHead.classList.add("track-playhead");
        elWave.append(elHead);

        const audio = new Audio();
        audio.preload = "metadata";
        audio.src = path;

        /** @type {Track} */
        const track = {
            path,
            canvas: elPeaks,
            context: elPeaks.getContext("2d"),
            head: elHead,
            audio,
            peaks: null,
        };
        tracks.push(track);

        elVolume.addEventListener("input", () => {
            audio.volume = Number(elVolume.value) / 100;
        });

        let elWavePressed = false;
        let elWaveResumeAfterSeek = false;
        function waveSeek(event) {
            const rect = elWave.getBoundingClientRect();
            seek(((event.clientX - rect.left) / rect.width) * duration);
        }
        function waveSeekEnd(event) {
            elWave.releasePointerCapture(event.pointerId);
            elWavePressed = false;
            if (elWaveResumeAfterSeek) {
                play();
            }
        }
        elWave.addEventListener("pointerdown", (event) => {
            elWave.setPointerCapture(event.pointerId);
            if (playing) {
                elWaveResumeAfterSeek = playing;
                pause();
            } else {
                elWaveResumeAfterSeek = false;
            }
            elWavePressed = true;
            waveSeek(event);
        });
        elWave.addEventListener("pointermove", (event) => {
            if (!elWavePressed) return;
            waveSeek(event);
        });
        elWave.addEventListener("pointerup", waveSeekEnd);
        elWave.addEventListener("pointercancel", waveSeekEnd);

        fragment.append(row);
    }

    tracksEl.replaceChildren(fragment);

    await Promise.all(tracks.map(loadTrackMetadata));
    updateDuration();

    playButton.disabled = false;
    seekSlider.disabled = false;
    updateButton();
    render();

    // Waveform buffers are only needed for the canvases, so decode them in the
    // background instead of blocking playback on the whole files.
    for (const track of tracks) {
        loadWaveform(track);
    }
}

async function loadLyrics() {
    const res = await fetch("./data/lyrics.txt");
    const text = await res.text();

    const fragment = document.createDocumentFragment();

    for (let line of text.split("\n")) {
        line = line.trim();
        const el = (() => {
            if (line === "") return;
            let match;
            if ((match = line.match(/\[(.+?)\]/))) {
                const header = document.createElement("h3");
                header.textContent = match[1];
                return header;
            }
            const lyric = document.createElement("div");
            lyric.textContent = line;
            return lyric;
        })();
        if (el) fragment.append(el);
    }
    document.getElementById("lyrics").replaceChildren(fragment);
}

let resizeFrame = 0;
window.addEventListener("resize", () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
        for (const track of tracks) drawPeaks(track);
    });
});

loadTracks();
loadLyrics();
