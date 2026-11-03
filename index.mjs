/** @type {{ name: string, path: string, context: CanvasRenderingContext2D, audio: HTMLAudioElement, peaks: Float32Array, peaksLoading: false }[]} */
let tracks = [];

const audioContext = new (window.AudioContext || window.webkitAudioContext)();

/**
 * @param {typeof tracks[number]} track
 */
async function drawPeaks(track) {
    if (track.peaksLoading) return;
    const ctx = track.context;
    let width, height;
    {
        const dpr = window.devicePixelRatio || 1;
        const bounding = ctx.canvas.getBoundingClientRect();
        width = ctx.canvas.width = Math.max(1, Math.round(bounding.width * dpr));
        height = ctx.canvas.height = Math.max(1, Math.round(bounding.height * dpr));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    ctx.clearRect(0, 0, width, height);

    const peaks = track.peaks;
    const mid = height / 2, step = width / peaks.length;
    ctx.fillStyle = "#8495ff";
    for (let i = 0; i < peaks.length; i++) {
        const amp = Math.max(1, peaks[i] * (height * 0.44));
        ctx.fillRect(i * step, mid - amp, Math.max(1, step * 0.72), amp * 2);
    }
}

async function loadPeaks(track) {
    if (track.peaksLoading) return;
    track.peaksLoading = true;

    const res = await fetch(track.path);
    const audio = await audioContext.decodeAudioData(await res.arrayBuffer());

    const data = audio.getChannelData(0);
    const count = 128;
    const peaks = new Float32Array(count);

    const block = Math.ceil(data.length / count);
    for (let i = 0; i < count; i++) {
        const start = i * block;
        const end = Math.min(data.length, start + block);
        let peak = 0;

        for (let j = start; j < end; j++) {
            const v = data[j];
            const abs = v < 0 ? -v : v;
            if (abs > peak) peak = abs;
        }

        peaks[i] = peak;
    }

    track.peaks = peaks;
    track.peaksLoading = false;
    drawPeaks(track);
}

async function loadTracks() {
    const res = await fetch("./index.json");
    tracks = Object.entries((await res.json()).files).map(([name, path]) => ({
        name,
        path,
    }));

    const fragment = document.createDocumentFragment();

    for (const track of tracks) {
        const el = document.createElement("div");
        fragment.append(el);
        el.classList.add("track");

        const elInfo = document.createElement("div");
        el.append(elInfo);
        elInfo.classList.add("track-info");
        {
            const elName = document.createElement("span");
            elInfo.append(elName);
            elName.classList.add("track-name");
            elName.textContent = track.name;

            const elPath = document.createElement("span");
            elInfo.append(elPath);
            elPath.classList.add("track-path");
            elPath.textContent = track.path;
        }

        const elVolume = document.createElement("input");
        el.append(elVolume);
        elVolume.classList.add("track-volume");
        elVolume.type = "range";
        elVolume.orient = "vertical";
        elVolume.min = "0";
        elVolume.max = "100";
        elVolume.value = "100";

        const elPeaks = document.createElement("canvas");
        el.append(elPeaks);
        elPeaks.classList.add("track-peaks");
        elPeaks.textContent = "Your browser doesn't support canvas";

        track.context = elPeaks.getContext("2d");
        track.context.element = elPeaks;
        track.audio = new Audio(track.path);
    }

    document.getElementById("tracks").replaceChildren(fragment);

    for (const track of tracks)
        loadPeaks(track);
    window.addEventListener("resize", () => {
        for (const track of tracks)
            drawPeaks(track);
    })
}


async function loadLyrics() {
    const res = await fetch("./lyrics.txt");
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

loadTracks();
loadLyrics();
