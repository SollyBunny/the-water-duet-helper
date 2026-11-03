/**
 * @typedef {{
 *   name: string,
 *   path: string,
 *   row: HTMLDivElement,
 *   pathEl: HTMLSpanElement,
 *   canvas: HTMLCanvasElement,
 *   context: CanvasRenderingContext2D,
 *   head: HTMLDivElement,
 *   volume: number,
 *   buffer: AudioBuffer | null,
 *   peaks: Float32Array | null,
 *   gain: GainNode | null,
 *   source: AudioBufferSourceNode | null,
 * }} Track
 */

/** @type {Track[]} */
const tracks = [];

const audioContext = new (window.AudioContext || window.webkitAudioContext)();

const tracksEl = document.getElementById("tracks");
const playButton = document.getElementById("player-button");
const seekSlider = document.getElementById("player-seek");
const timeEl = document.getElementById("player-time");

let duration = 0;
let playing = false;
/** Audio-context time at which playback was scheduled to start. */
let startedAt = 0;
/** Position playback starts from, and the position while paused. */
let offset = 0;
let frame = 0;
let resumeAfterSeek = false;

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

/** @param {Track} track */
async function loadTrack(track) {
    const res = await fetch(track.path);
    if (!res.ok) throw new Error(`Could not fetch ${track.path} (${res.status})`);

    const buffer = await audioContext.decodeAudioData(await res.arrayBuffer());
    const gain = audioContext.createGain();
    gain.gain.value = track.volume;
    gain.connect(audioContext.destination);

    track.buffer = buffer;
    track.peaks = computePeaks(buffer);
    track.gain = gain;
    drawPeaks(track);
}

function currentTime() {
    if (!playing) return offset;
    return Math.min(duration, offset + Math.max(0, audioContext.currentTime - startedAt));
}

function stopSources() {
    for (const track of tracks) {
        if (!track.source) continue;
        try {
            track.source.stop();
        } catch (_) {}
        track.source.disconnect();
        track.source = null;
    }
}

function updateButton() {
    playButton.textContent = playing ? "Pause" : "Play";
    playButton.setAttribute("aria-pressed", String(playing));
}

/** @param {number} time */
function seekTo(time) {
    offset = clamp(time, 0, duration);

    if (offset >= duration) {
        playing = false;
        stopSources();
        cancelAnimationFrame(frame);
        updateButton();
        render();
        return;
    }

    if (playing) {
        playing = false;
        stopSources();
        cancelAnimationFrame(frame);
        play();
    } else {
        render();
    }
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
        playing = false;
        offset = duration;
        stopSources();
        updateButton();
        render();
        return;
    }

    render();
    frame = requestAnimationFrame(tick);
}

async function play() {
    if (playing || duration <= 0) return;

    await audioContext.resume();
    stopSources();

    // Give every source the same start time so the tracks stay in sync.
    const startAt = audioContext.currentTime + 0.05;

    for (const track of tracks) {
        if (!track.buffer || !track.gain) continue;
        if (offset >= track.buffer.duration) continue;

        const source = audioContext.createBufferSource();
        source.buffer = track.buffer;
        source.connect(track.gain);
        source.start(startAt, offset);
        track.source = source;
    }

    startedAt = startAt;
    playing = true;
    updateButton();
    cancelAnimationFrame(frame);
    tick();
}

function pause() {
    if (!playing) return;

    offset = currentTime();
    playing = false;
    stopSources();
    cancelAnimationFrame(frame);
    updateButton();
    render();
}

playButton.addEventListener("click", () => {
    if (playing) {
        pause();
    } else {
        // After reaching the end, Play starts over from the beginning.
        if (offset >= duration) offset = 0;
        play();
    }
});

seekSlider.addEventListener("input", () => {
    const value = Number(seekSlider.value);
    if (playing) {
        resumeAfterSeek = true;
        pause();
    }
    offset = (value / Number(seekSlider.max)) * duration;
    render();
});

seekSlider.addEventListener("change", () => {
    if (!resumeAfterSeek) return;
    resumeAfterSeek = false;
    if (offset < duration) play();
});

async function loadTracks() {
    const res = await fetch("./data/index.json");
    if (!res.ok) throw new Error(`Could not fetch index.json (${res.status})`);

    const manifest = await res.json();
    const entries = Object.entries(manifest.files ?? manifest);
    if (!entries.length) throw new Error("No audio files listed in index.json.");

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

        /** @type {Track} */
        const track = {
            name,
            path,
            row,
            pathEl: elPath,
            canvas: elPeaks,
            context: elPeaks.getContext("2d"),
            head: elHead,
            volume: 1,
            buffer: null,
            peaks: null,
            gain: null,
            source: null,
        };
        tracks.push(track);

        elVolume.addEventListener("input", () => {
            track.volume = Number(elVolume.value) / 100;
            if (track.gain) track.gain.gain.value = track.volume;
        });

        elWave.addEventListener("click", (event) => {
            const rect = elWave.getBoundingClientRect();
            seekTo(((event.clientX - rect.left) / rect.width) * duration);
        });

        fragment.append(row);
    }

    tracksEl.replaceChildren(fragment);

    const results = await Promise.allSettled(tracks.map(loadTrack));

    for (let i = 0; i < results.length; i++) {
        const result = results[i];
        if (result.status === "rejected") {
            const track = tracks[i];
            track.row.classList.add("track-error");
            track.pathEl.textContent = `${track.path} — ${result.reason?.message ?? result.reason}`;
            console.error(`Could not load ${track.path}`, result.reason);
        }
    }

    duration = tracks.reduce((max, track) => Math.max(max, track.buffer?.duration ?? 0), 0);

    if (duration <= 0) {
        throw new Error("None of the audio files could be loaded.");
    }

    playButton.disabled = false;
    seekSlider.disabled = false;
    updateButton();
    render();
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

loadTracks().catch((error) => {
    console.error(error);
    const el = document.createElement("p");
    el.textContent = `Could not load tracks: ${error.message}`;
    tracksEl.replaceChildren(el);
});

loadLyrics().catch((error) => {
    console.error(error);
    const el = document.createElement("p");
    el.textContent = `Could not load lyrics: ${error.message}`;
    document.getElementById("lyrics").replaceChildren(el);
});
