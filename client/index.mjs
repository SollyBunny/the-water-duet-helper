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
 *   audio: HTMLAudioElement,
 *   peaks: Float32Array | null,
 *   peaksLoading: boolean,
 * }} Track
 *
 * @typedef {{
 *   name: string,
 *   status: string,
 *   progress: any,
 *   error: string | null,
 *   meta: any,
 *   files: Array<{name: string, url: string, size: number, mtime: number}>,
 *   index: string,
 *   created: number,
 *   updated: number,
 * }} Output
 */

const audioContext = new (window.AudioContext || window.webkitAudioContext)();

const tracksEl = document.getElementById("tracks");
const playButton = document.getElementById("player-button");
const seekSlider = document.getElementById("player-seek");
const timeEl = document.getElementById("player-time");
const player = document.getElementById("player");
const playerView = document.getElementById("player-view");
const emptyEl = document.getElementById("empty");
const statusPanel = document.getElementById("status-panel");
const statusTitle = document.getElementById("status-title");
const statusMessage = document.getElementById("status-message");
const statusDetail = document.getElementById("status-detail");
const progressBar = document.getElementById("progress-bar");
const trackTitle = document.getElementById("track-title");
const trackMeta = document.getElementById("track-meta");
const lyricsSection = document.getElementById("lyrics-section");
const lyricsEl = document.getElementById("lyrics");
const outputsEl = document.getElementById("outputs");
const connectionEl = document.getElementById("connection");
const uploadForm = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const fileLabel = document.getElementById("file-label");
const nameInput = document.getElementById("name-input");
const uploadButton = document.getElementById("upload-button");
const uploadError = document.getElementById("upload-error");

/** @type {Map<string, Output>} */
const outputs = new Map();
let selectedName = null;
let loadedName = null;
let loadGeneration = 0;
let pendingSelection = null;
let autoName = "";

/** @type {Track[]} */
let tracks = [];
let trackGeneration = 0;

let duration = 0;
let playing = false;
/** Position playback starts from, and the position while paused. */
let offset = 0;
let frame = 0;
let masterTrack = null;

const fmt = (seconds) => {
    seconds = Math.max(0, Math.floor(seconds || 0));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

// ---------------------------------------------------------------------------
// waveforms (loaded separately from playback)
// ---------------------------------------------------------------------------

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

    const mid = height / 2;
    if (!peaks) {
        ctx.fillStyle = "#2b3342";
        ctx.fillRect(0, mid - 1, width, 2);
        return;
    }

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
async function loadPeaks(track, generation) {
    if (track.peaksLoading) return;
    track.peaksLoading = true;

    try {
        const res = await fetch(track.path);
        if (!res.ok) throw new Error(`Could not fetch ${track.path} (${res.status})`);
        const buffer = await audioContext.decodeAudioData(await res.arrayBuffer());
        if (generation !== trackGeneration) return;

        track.peaks = computePeaks(buffer);
        drawPeaks(track);
    } catch (error) {
        console.warn(`Could not load waveform for ${track.path}`, error);
    } finally {
        track.peaksLoading = false;
    }
}

// ---------------------------------------------------------------------------
// playback (HTMLAudioElement streams each track)
// ---------------------------------------------------------------------------

function currentTime() {
    if (!playing) return offset;
    const master = masterTrack ?? tracks[0];
    if (!master) return offset;
    const time = master.audio.currentTime;
    return Number.isFinite(time) ? time : offset;
}

function updateButton() {
    playButton.textContent = playing ? "Pause" : "Play";
    playButton.setAttribute("aria-pressed", String(playing));
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

function pauseAll() {
    for (const track of tracks) track.audio.pause();
}

function tick() {
    if (!playing) return;

    const master = masterTrack ?? tracks[0];
    if (!master) return;

    const time = master.audio.currentTime;
    if (Number.isFinite(time)) {
        // Correct small drift between the independently streaming elements.
        for (const track of tracks) {
            if (track === master) continue;
            const other = track.audio.currentTime;
            if (Number.isFinite(other) && Math.abs(other - time) > 0.08) {
                try {
                    track.audio.currentTime = time;
                } catch (_) {}
            }
        }
    }

    if (master.audio.ended || (duration > 0 && Number.isFinite(time) && time >= duration - 0.02)) {
        playing = false;
        offset = duration;
        pauseAll();
        updateButton();
        render();
        return;
    }

    render();
    frame = requestAnimationFrame(tick);
}

async function play() {
    if (playing || !tracks.length) return;
    if (duration > 0 && offset >= duration) offset = 0;

    playing = true;
    updateButton();

    masterTrack = tracks.find((track) => track.audio.readyState >= 1) ?? tracks[0];
    for (const track of tracks) {
        try {
            track.audio.currentTime = offset;
        } catch (_) {}
    }

    await Promise.allSettled(tracks.map((track) => track.audio.play()));
    if (!playing) return;

    cancelAnimationFrame(frame);
    tick();
}

function pause() {
    if (!playing) return;

    offset = currentTime();
    playing = false;
    pauseAll();
    cancelAnimationFrame(frame);
    updateButton();
    render();
}

/** @param {number} time */
function seekTo(time) {
    offset = clamp(time, 0, duration || 0);
    for (const track of tracks) {
        try {
            track.audio.currentTime = offset;
        } catch (_) {}
    }
    render();
}

playButton.addEventListener("click", () => {
    if (playing) pause();
    else play();
});

seekSlider.addEventListener("input", () => {
    const value = Number(seekSlider.value);
    seekTo((value / Number(seekSlider.max)) * duration);
});

// ---------------------------------------------------------------------------
// player view / output status
// ---------------------------------------------------------------------------

function clearPlayer() {
    playing = false;
    cancelAnimationFrame(frame);

    for (const track of tracks) {
        track.audio.pause();
        track.audio.removeAttribute("src");
        track.audio.load();
    }

    tracks = [];
    trackGeneration++;
    loadedName = null;
    duration = 0;
    offset = 0;
    masterTrack = null;

    tracksEl.replaceChildren();
    playButton.disabled = true;
    seekSlider.disabled = true;
    seekSlider.value = "0";
    timeEl.textContent = "0:00 / 0:00";
    updateButton();

    player.hidden = true;
    playerView.hidden = true;
    lyricsSection.hidden = true;
    lyricsEl.replaceChildren();
}

function clearView() {
    clearPlayer();
    statusPanel.hidden = true;
    emptyEl.hidden = false;
}

function setStatusPanel({ title, message, detail = "", progress = 0, tone = "" }) {
    statusTitle.textContent = title;
    statusMessage.textContent = message;
    statusDetail.textContent = detail;
    progressBar.style.width = `${clamp(progress, 0, 1) * 100}%`;
    statusPanel.dataset.tone = tone;
}

function describeMeta(meta) {
    const bits = [];
    if (meta?.duration_seconds) bits.push(fmt(meta.duration_seconds));
    for (const singer of meta?.singers ?? []) {
        if (!singer.median_f0_hz) continue;
        const name = String(singer.file ?? "").replace(/\.wav$/i, "");
        bits.push(`${name}: ${Math.round(singer.median_f0_hz)} Hz`);
    }
    return bits.join(" · ");
}

function renderStatus(out) {
    const progress = out.progress ?? {};
    let title = out.name;
    let message = progress.message || out.status;
    let tone = "";

    if (out.status === "queued") {
        title = `Queued: ${out.name}`;
        message = progress.message || "Waiting for the separator.";
    } else if (out.status === "processing") {
        title = `Separating: ${out.name}`;
        message = progress.message || "Working…";
    } else if (out.status === "deleting") {
        title = `Stopping: ${out.name}`;
        message = "Stopping the separator and deleting output…";
        tone = "warn";
    } else if (out.status === "error") {
        title = `Failed: ${out.name}`;
        message = out.error || progress.message || "Processing failed.";
        tone = "error";
    } else if (out.status === "incomplete") {
        title = `Incomplete: ${out.name}`;
        message = "This output was interrupted before it finished.";
        tone = "warn";
    }

    const detail = [];
    if (progress.chunks_done != null && progress.chunks_total) {
        detail.push(`chunk ${progress.chunks_done}/${progress.chunks_total}`);
    }
    if (progress.eta_seconds != null) detail.push(`ETA ${fmt(progress.eta_seconds)}`);
    if (progress.elapsed_seconds != null) detail.push(`${fmt(progress.elapsed_seconds)} elapsed`);

    setStatusPanel({
        title,
        message,
        detail: detail.join(" · "),
        progress: progress.overall ?? 0,
        tone,
    });
}

function showStatus(out) {
    clearPlayer();
    emptyEl.hidden = true;
    statusPanel.hidden = false;
    renderStatus(out);
}

function updateStatusPanel(out) {
    if (statusPanel.hidden) return;
    if (out.status === "complete") return;
    renderStatus(out);
}

function redrawTracks() {
    for (const track of tracks) drawPeaks(track);
}

function showPlayerView(out) {
    statusPanel.hidden = true;
    emptyEl.hidden = true;
    playerView.hidden = false;
    player.hidden = tracks.length === 0;
    trackTitle.textContent = out.name;
    trackMeta.textContent = describeMeta(out.meta);
    // The canvases have no layout size while the view is hidden.
    requestAnimationFrame(redrawTracks);
}

function showClientError(name, message) {
    clearPlayer();
    emptyEl.hidden = true;
    statusPanel.hidden = false;
    setStatusPanel({
        title: `Could not load ${name}`,
        message,
        tone: "error",
    });
}

// ---------------------------------------------------------------------------
// output list
// ---------------------------------------------------------------------------

const statusLabels = {
    queued: "queued",
    processing: "processing",
    deleting: "stopping",
    complete: "ready",
    error: "error",
    incomplete: "incomplete",
};

function applyOutputRow(row, out) {
    row.classList.toggle("selected", out.name === selectedName);
    row.classList.toggle("complete", out.status === "complete");
    row.classList.toggle("error", out.status === "error");

    const badge = row.querySelector(".output-status");
    if (badge) {
        badge.textContent = statusLabels[out.status] ?? out.status;
        badge.className = `output-status ${out.status}`;
    }

    const fill = row.querySelector(".output-bar-fill");
    if (fill) {
        let progress = out.progress?.overall ?? 0;
        if (out.status === "complete") progress = 1;
        if (out.status === "error" || out.status === "incomplete") progress = 1;
        fill.style.width = `${clamp(progress, 0, 1) * 100}%`;
    }
}

function updateOutputRow(out) {
    const row = outputsEl.querySelector(`.output[data-name="${CSS.escape(out.name)}"]`);
    if (row) applyOutputRow(row, out);
}

function markSelected(name) {
    for (const row of outputsEl.querySelectorAll(".output")) {
        row.classList.toggle("selected", row.dataset.name === name);
    }
}

function createOutputRow(out) {
    const row = document.createElement("li");
    row.className = "output";
    row.dataset.name = out.name;

    const select = document.createElement("button");
    select.type = "button";
    select.className = "output-select";
    select.title = `Open ${out.name}`;

    const nameEl = document.createElement("span");
    nameEl.className = "output-name";
    nameEl.textContent = out.name;

    const badge = document.createElement("span");
    badge.className = "output-status";
    select.append(nameEl, badge);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "output-delete";
    remove.textContent = "Delete";
    remove.title = `Delete ${out.name}`;
    remove.setAttribute("aria-label", `Delete ${out.name}`);

    const bar = document.createElement("div");
    bar.className = "output-bar";
    const fill = document.createElement("div");
    fill.className = "output-bar-fill";
    bar.append(fill);

    select.addEventListener("click", () => selectOutput(out.name));
    remove.addEventListener("click", (event) => {
        event.stopPropagation();
        deleteOutput(out.name);
    });

    row.append(select, remove, bar);
    applyOutputRow(row, out);
    return row;
}

function renderOutputs() {
    const list = [...outputs.values()].sort((a, b) => {
        return (b.updated || 0) - (a.updated || 0) || a.name.localeCompare(b.name);
    });

    if (!list.length) {
        const empty = document.createElement("li");
        empty.className = "hint";
        empty.textContent = "No outputs yet.";
        outputsEl.replaceChildren(empty);
        return;
    }

    const fragment = document.createDocumentFragment();
    for (const out of list) fragment.append(createOutputRow(out));
    outputsEl.replaceChildren(fragment);
}

function chooseDefault() {
    const wanted = new URLSearchParams(location.search).get("output");
    if (wanted && outputs.has(wanted)) {
        selectOutput(wanted, { updateUrl: false });
        return;
    }
    if (selectedName && outputs.has(selectedName)) {
        selectOutput(selectedName);
        return;
    }
    if (outputs.has("the-water")) {
        selectOutput("the-water");
        return;
    }

    const list = [...outputs.values()].sort((a, b) => (b.updated || 0) - (a.updated || 0));
    const preferred = list.find((out) => out.status === "complete") ?? list[0];
    if (preferred) {
        selectOutput(preferred.name);
        return;
    }

    selectedName = null;
    clearView();
    renderOutputs();
}

function selectOutput(name, { updateUrl = true } = {}) {
    const out = outputs.get(name);
    if (!out) return;

    selectedName = name;
    if (updateUrl) {
        const url = new URL(location.href);
        url.searchParams.set("output", name);
        history.replaceState(null, "", url);
    }

    markSelected(name);

    if (out.status === "complete") {
        if (loadedName === name) showPlayerView(out);
        else loadOutput(name);
    } else {
        showStatus(out);
    }
}

// ---------------------------------------------------------------------------
// manifest / track creation
// ---------------------------------------------------------------------------

async function loadOutput(name) {
    const gen = ++loadGeneration;
    loadedName = null;

    try {
        const res = await fetch(`/output/${encodeURIComponent(name)}/index.json`, {
            cache: "no-store",
        });
        if (!res.ok) throw new Error(`Could not load manifest (${res.status})`);
        const manifest = await res.json();
        if (gen !== loadGeneration || selectedName !== name) return;

        loadTracks(manifest, name);
        loadedName = name;

        const out = outputs.get(name);
        if (out) showPlayerView(out);
        loadLyrics(name, gen);
    } catch (error) {
        if (gen !== loadGeneration) return;
        console.error(error);
        showClientError(name, error?.message ?? String(error));
    }
}

function loadTracks(manifest, name) {
    const entries = Object.entries(manifest.files ?? manifest);
    if (!entries.length) throw new Error("No audio files in the manifest.");

    clearPlayer();

    const generation = trackGeneration;
    duration = Number(manifest.meta?.duration_seconds) || 0;
    const fragment = document.createDocumentFragment();

    for (const [trackName, path] of entries) {
        const row = document.createElement("div");
        row.classList.add("track");

        const elInfo = document.createElement("div");
        elInfo.classList.add("track-info");
        row.append(elInfo);

        const elName = document.createElement("span");
        elName.classList.add("track-name");
        elName.textContent = trackName;
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
        elVolume.setAttribute("aria-label", `${trackName} volume`);
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
        audio.volume = 1;

        /** @type {Track} */
        const track = {
            name: trackName,
            path,
            row,
            pathEl: elPath,
            canvas: elPeaks,
            context: elPeaks.getContext("2d"),
            head: elHead,
            volume: 1,
            audio,
            peaks: null,
            peaksLoading: false,
        };
        tracks.push(track);

        audio.addEventListener("loadedmetadata", () => {
            if (generation !== trackGeneration) return;
            if (Number.isFinite(audio.duration) && audio.duration > duration) {
                duration = audio.duration;
                seekSlider.disabled = false;
            }
            if (playing) {
                try {
                    audio.currentTime = offset;
                } catch (_) {}
            }
            render();
        });

        audio.addEventListener("error", () => {
            track.row.classList.add("track-error");
            track.pathEl.textContent = `${path} — could not load audio`;
        });

        elVolume.addEventListener("input", () => {
            track.volume = Number(elVolume.value) / 100;
            track.audio.volume = track.volume;
        });

        elWave.addEventListener("click", (event) => {
            const rect = elWave.getBoundingClientRect();
            seekTo(((event.clientX - rect.left) / rect.width) * duration);
        });

        drawPeaks(track);
        loadPeaks(track, generation);
        fragment.append(row);
    }

    tracksEl.replaceChildren(fragment);
    // Size the canvases now that the rows are in the layout.
    for (const track of tracks) drawPeaks(track);

    playButton.disabled = tracks.length === 0;
    seekSlider.disabled = !(duration > 0);
    updateButton();
    render();

    trackTitle.textContent = name;
    trackMeta.textContent = describeMeta(manifest.meta);
}

async function loadLyrics(name, gen) {
    lyricsSection.hidden = true;
    lyricsEl.replaceChildren();

    let text = null;
    try {
        const res = await fetch(`/output/${encodeURIComponent(name)}/lyrics.txt`);
        if (res.ok) text = await res.text();
    } catch (_) {}

    if (text == null || gen !== loadGeneration || selectedName !== name) return;

    const fragment = document.createDocumentFragment();
    for (let line of text.split("\n")) {
        line = line.trim();
        if (line === "") continue;

        let match;
        if ((match = line.match(/\[(.+?)\]/))) {
            const header = document.createElement("h3");
            header.textContent = match[1];
            fragment.append(header);
            continue;
        }

        const lyric = document.createElement("div");
        lyric.textContent = line;
        fragment.append(lyric);
    }

    const hasContent = fragment.childElementCount > 0;
    lyricsEl.replaceChildren(fragment);
    lyricsSection.hidden = !hasContent;
}

// ---------------------------------------------------------------------------
// upload / delete
// ---------------------------------------------------------------------------

function setUploadError(message) {
    uploadError.textContent = message ?? "";
    uploadError.hidden = !message;
}

function defaultName(filename) {
    return filename
        .replace(/\.[^.]+$/, "")
        .replace(/[\\/]+/g, "-")
        .replace(/^\.+/, "")
        .trim() || "song";
}

fileInput.addEventListener("change", () => {
    const file = fileInput.files?.[0];
    fileLabel.textContent = file ? file.name : "Choose audio file";
    if (!file) return;

    const suggested = defaultName(file.name);
    if (!nameInput.value || nameInput.value === autoName) {
        nameInput.value = suggested;
        autoName = suggested;
    }
});

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    setUploadError("");

    const file = fileInput.files?.[0];
    if (!file) {
        setUploadError("Choose an audio file first.");
        return;
    }

    const name = nameInput.value.trim();
    if (!name) {
        setUploadError("Give the output a name.");
        return;
    }

    if (outputs.has(name) && !confirm(`Replace existing output "${name}"?`)) {
        return;
    }

    uploadButton.disabled = true;
    uploadButton.textContent = "Uploading…";

    try {
        const form = new FormData();
        form.append("file", file, file.name);

        const res = await fetch(`/api/output/${encodeURIComponent(name)}`, {
            method: "POST",
            body: form,
        });
        if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);

        pendingSelection = name;
        fileInput.value = "";
        fileLabel.textContent = "Choose audio file";
        nameInput.value = "";
        autoName = "";

        if (!outputs.has(name)) {
            const now = Date.now() / 1000;
            outputs.set(name, {
                name,
                status: "queued",
                progress: { stage: "queued", overall: 0, message: "Queued" },
                error: null,
                meta: null,
                files: [],
                index: `/output/${encodeURIComponent(name)}/index.json`,
                created: now,
                updated: now,
            });
            renderOutputs();
        }
        selectOutput(name);
    } catch (error) {
        setUploadError(error?.message ?? String(error));
    } finally {
        uploadButton.disabled = false;
        uploadButton.textContent = "Separate";
    }
});

async function deleteOutput(name) {
    if (!confirm(`Delete "${name}"? This stops processing and removes its files.`)) {
        return;
    }

    try {
        const res = await fetch(`/api/output/${encodeURIComponent(name)}`, {
            method: "DELETE",
        });
        if (!res.ok && res.status !== 404) {
            throw new Error((await res.text()) || `HTTP ${res.status}`);
        }

        if (res.status === 404) {
            outputs.delete(name);
        } else {
            const data = await res.json();
            if (data.status === "deleted") {
                outputs.delete(name);
            } else if (data.status === "cancelling") {
                const out = outputs.get(name);
                if (out) {
                    out.status = "deleting";
                    out.progress = {
                        stage: "deleting",
                        overall: 0,
                        message: "Stopping",
                    };
                    updateOutputRow(out);
                    if (selectedName === name) showStatus(out);
                }
                return;
            }
        }

        if (selectedName === name) {
            selectedName = null;
            clearView();
        }
        renderOutputs();
        chooseDefault();
    } catch (error) {
        alert(`Could not delete ${name}: ${error?.message ?? error}`);
    }
}

// ---------------------------------------------------------------------------
// websocket state
// ---------------------------------------------------------------------------

function setConnection(state, text) {
    connectionEl.textContent = text;
    connectionEl.className = `connection ${state}`;
}

function fileSignature(files) {
    return (files ?? []).map((file) => `${file.name}:${file.size}`).join("|");
}

function syncSelected(previous) {
    const out = outputs.get(selectedName);
    if (!out) {
        clearView();
        return;
    }

    if (out.status === "complete") {
        const changed = !previous
            || previous.status !== "complete"
            || fileSignature(previous.files) !== fileSignature(out.files)
            || previous.updated !== out.updated;
        if (changed || loadedName !== out.name) loadOutput(out.name);
        else showPlayerView(out);
    } else {
        showStatus(out);
    }
}

function handleMessage(message) {
    switch (message.type) {
        case "state": {
            const previous = new Map(outputs);
            outputs.clear();
            for (const out of message.outputs ?? []) outputs.set(out.name, out);
            renderOutputs();

            if (!selectedName || !outputs.has(selectedName)) chooseDefault();
            else syncSelected(previous.get(selectedName));
            break;
        }

        case "add":
        case "update": {
            const out = message.output;
            if (!out) break;
            const previous = outputs.get(out.name);
            outputs.set(out.name, out);
            if (previous) updateOutputRow(out);
            else renderOutputs();

            if (pendingSelection === out.name) {
                pendingSelection = null;
                selectOutput(out.name);
            } else if (!selectedName) {
                chooseDefault();
            } else if (selectedName === out.name) {
                syncSelected(previous);
            }
            break;
        }

        case "progress": {
            const out = outputs.get(message.name);
            if (!out) break;
            out.progress = message.progress;
            if (message.status) out.status = message.status;
            updateOutputRow(out);
            if (selectedName === out.name) {
                if (statusPanel.hidden && out.status !== "complete") showStatus(out);
                else updateStatusPanel(out);
            }
            break;
        }

        case "delete": {
            const wasSelected = selectedName === message.name;
            outputs.delete(message.name);
            renderOutputs();
            if (wasSelected) {
                selectedName = null;
                clearView();
                chooseDefault();
            }
            break;
        }
    }
}

let socket = null;
let retryCount = 0;

function connectSocket() {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${location.host}/api/ws`);
    socket = ws;
    setConnection("connecting", "connecting…");

    ws.addEventListener("open", () => {
        retryCount = 0;
        setConnection("live", "live");
    });

    ws.addEventListener("message", (event) => {
        let message;
        try {
            message = JSON.parse(event.data);
        } catch (_) {
            return;
        }
        handleMessage(message);
    });

    ws.addEventListener("close", () => {
        socket = null;
        setConnection("offline", "offline");
        const delay = Math.min(10000, 500 * 2 ** retryCount++);
        setTimeout(connectSocket, delay);
    });

    ws.addEventListener("error", () => ws.close());
}

let resizeFrame = 0;
window.addEventListener("resize", () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
        for (const track of tracks) drawPeaks(track);
    });
});

renderOutputs();
connectSocket();
