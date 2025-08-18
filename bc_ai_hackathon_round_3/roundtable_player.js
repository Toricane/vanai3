/* Roundtable Playback Module
 * Loads audio manifest + cluster metadata and enables sequential playback
 */
(function () {
    const state = {
        initialized: false,
        questionKey: null,
        lines: [],
        index: 0,
        playing: false,
        autoScroll: true,
        audioEl: null,
        manifest: null,
        clustersMeta: null,
        colorMap: {},
        speed: 1.0,
        preloadCount: 2,
    };

    // Utility
    function $(sel, root = document) {
        return root.querySelector(sel);
    }
    function el(tag, cls, text) {
        const n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text) n.textContent = text;
        return n;
    }
    function fetchJSON(url) {
        return fetch(url).then((r) => {
            if (!r.ok) throw new Error("Fetch failed " + url);
            return r.json();
        });
    }

    async function fetchFirst(paths) {
        let lastErr;
        for (const p of paths) {
            try {
                return await fetchJSON(p);
            } catch (e) {
                lastErr = e;
            }
        }
        throw lastErr || new Error("All fetch attempts failed");
    }

    function log(msg) {
        console.log("[Roundtable]", msg);
    }

    function buildModule() {
        const mount = document.getElementById("roundtable-module");
        if (!mount) return;
        mount.innerHTML = "";
        const card = el("div", "rt-card");

        // Header & description
        const header = el("div", "rt-header");
        const h = el("h3", "rt-title", "Roundtable Dialogue");
        const d = el(
            "p",
            "rt-desc",
            "AI-generated dialogue between thematic response clusters. Distinct voices reflect cluster styles; a final synthesis summarizes common ground and tensions."
        );
        header.appendChild(h);
        header.appendChild(d);
        card.appendChild(header);

        // Controls row
        const controls = el("div", "rt-controls");
        const btnPlay = el("button", "", "Play");
        const btnPrev = el("button", "secondary", "◀");
        const btnNext = el("button", "secondary", "▶");
        const btnReplay = el("button", "secondary", "↺");
        const btnScroll = el("button", "secondary", "Auto-Scroll: On");
        const speedSel = el("select", "rt-speed-select");
        [0.75, 1, 1.1, 1.25].forEach((v) => {
            const o = el("option", "", v + "x");
            o.value = v;
            if (v === 1) o.selected = true;
            speedSel.appendChild(o);
        });

        controls.append(
            btnPlay,
            btnPrev,
            btnNext,
            btnReplay,
            btnScroll,
            speedSel
        );
        card.appendChild(controls);

        // Legend
        const legend = el("div", "rt-legend");
        legend.id = "rt-legend";
        card.appendChild(legend);

        // Transcript
        const transcriptContainer = el("div");
        transcriptContainer.id = "rt-transcript-container";
        card.appendChild(transcriptContainer);

        const status = el("div");
        status.id = "rt-status";
        status.textContent = "Idle";
        card.appendChild(status);

        mount.appendChild(card);

        // Save refs
        state.refs = {
            mount,
            card,
            controls,
            btnPlay,
            btnPrev,
            btnNext,
            btnReplay,
            btnScroll,
            speedSel,
            legend,
            transcriptContainer,
            status,
        };

        btnPlay.addEventListener("click", togglePlay);
        btnPrev.addEventListener("click", () => jumpRelative(-1));
        btnNext.addEventListener("click", () => jumpRelative(1));
        btnReplay.addEventListener("click", replayLine);
        btnScroll.addEventListener("click", () => {
            state.autoScroll = !state.autoScroll;
            btnScroll.textContent =
                "Auto-Scroll: " + (state.autoScroll ? "On" : "Off");
        });
        speedSel.addEventListener("change", () => {
            state.speed = parseFloat(speedSel.value);
            if (state.audioEl) state.audioEl.playbackRate = state.speed;
        });

        // Keyboard shortcuts
        window.addEventListener("keydown", (e) => {
            if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName))
                return;
            if (e.code === "Space") {
                e.preventDefault();
                togglePlay();
            } else if (e.code === "ArrowRight") {
                jumpRelative(1);
            } else if (e.code === "ArrowLeft") {
                jumpRelative(-1);
            } else if (e.code === "KeyR") {
                replayLine();
            }
        });
    }

    function init(questionKey) {
        if (state.initialized && state.questionKey === questionKey) {
            return;
        }
        state.questionKey = questionKey;
        buildModule();
        Promise.all([
            fetchFirst([
                // Most likely when served with root at bc_ai_hackathon_round_3
                "../roundtable_audio/audio_manifest.json",
                "./roundtable_audio/audio_manifest.json",
                "/roundtable_audio/audio_manifest.json",
                "../../roundtable_audio/audio_manifest.json",
            ]),
            fetchFirst([
                "../clustered_responses.json",
                "./clustered_responses.json",
                "/clustered_responses.json",
            ]),
        ])
            .then(([manifest, clusters]) => {
                state.manifest = manifest;
                // question-specific entry
                const qEntry = manifest[questionKey];
                if (!qEntry) {
                    state.refs.status.textContent =
                        "No audio manifest entry for this question.";
                    return;
                }
                state.lines = qEntry.lines;
                // clusters meta for legend
                state.clustersMeta = clusters[questionKey];
                if (state.clustersMeta) {
                    state.colorMap = state.clustersMeta.clusters || {};
                }
                renderLegend();
                renderTranscript();
                createAudioEl();
                preloadUpcoming();
                state.initialized = true;
                state.refs.status.textContent = "Ready.";
            })
            .catch((err) => {
                state.refs.status.textContent = "Failed to load audio assets.";
                log(err);
            });
    }

    function createAudioEl() {
        const a = new Audio();
        a.preload = "none";
        a.addEventListener("ended", () => advance(1));
        a.addEventListener("error", () => {
            state.refs.status.textContent = "Audio error; skipping.";
            advance(1);
        });
        state.audioEl = a;
    }

    function buildLineURL(line) {
        if (!line || !line.audio) return "";
        let p = line.audio.replace(/\\/g, "/"); // normalize Windows paths
        // Ensure relative prefix from per-question page directory
        if (p.startsWith("roundtable_audio/")) {
            return "../" + p;
        }
        if (p.includes("/roundtable_audio/")) {
            // already has some leading path, leave as-is
            return p;
        }
        return p;
    }

    function renderLegend() {
        const { legend } = state.refs;
        legend.innerHTML = "";
        if (!state.clustersMeta) return;
        const clusterColors = state.colorMap;
        const vibeMap = {}; // cluster_label -> vibe
        if (state.manifest && state.manifest[state.questionKey]) {
            state.manifest[state.questionKey].lines.forEach((l) => {
                if (l.vibe && l.cluster_label !== "SYNTHESIS")
                    vibeMap[l.cluster_label] = l.vibe;
            });
        }
        Object.keys(clusterColors).forEach((label) => {
            const item = el("div", "rt-legend-item");
            const dot = el("span", "rt-color-dot");
            dot.style.background = clusterColors[label];
            item.appendChild(dot);
            const txt = el("span", "", label);
            item.appendChild(txt);
            if (vibeMap[label]) {
                const vibe = el("div", "rt-vibe", vibeMap[label]);
                item.appendChild(vibe);
            }
            legend.appendChild(item);
        });
    }

    function renderTranscript() {
        const root = state.refs.transcriptContainer;
        root.innerHTML = "";
        state.lines.forEach((ln, i) => {
            const div = el(
                "div",
                "rt-line" +
                    (ln.cluster_label === "SYNTHESIS" ? " synthesis" : "")
            );
            div.dataset.index = i;
            const color = state.colorMap[ln.cluster_label];
            if (color) div.style.borderLeftColor = color;
            const speaker = el("span", "rt-speaker", ln.cluster_label + ":");
            const text = el("span", "rt-text", " " + ln.text);
            div.appendChild(speaker);
            div.appendChild(text);
            div.addEventListener("click", () => {
                jumpTo(i, true);
            });
            root.appendChild(div);
        });
    }

    function highlight() {
        const nodes =
            state.refs.transcriptContainer.querySelectorAll(".rt-line");
        nodes.forEach((n) => n.classList.remove("active"));
        const current = state.refs.transcriptContainer.querySelector(
            '.rt-line[data-index="' + state.index + '"]'
        );
        if (current) {
            current.classList.add("active");
            if (state.autoScroll) {
                current.scrollIntoView({ block: "center", behavior: "smooth" });
            }
        }
    }

    function togglePlay() {
        if (!state.audioEl || state.lines.length === 0) {
            return;
        }
        if (!state.playing) {
            playCurrent(true);
        } else {
            state.audioEl.pause();
            state.playing = false;
            state.refs.btnPlay.textContent = "Play";
            state.refs.status.textContent = "Paused";
        }
    }

    function playCurrent(loadNew) {
        const line = state.lines[state.index];
        if (!line) {
            return;
        }
        const url = buildLineURL(line);
        if (loadNew) {
            state.audioEl.src = url;
            state.audioEl.playbackRate = state.speed;
        }
        state.audioEl
            .play()
            .then(() => {
                state.playing = true;
                state.refs.btnPlay.textContent = "Pause";
                state.refs.status.textContent =
                    "Playing line " +
                    (state.index + 1) +
                    "/" +
                    state.lines.length;
                highlight();
                preloadUpcoming();
            })
            .catch((err) => {
                state.refs.status.textContent = "Cannot play line; skipping.";
                log(err);
                advance(1);
            });
    }

    function jumpRelative(delta) {
        const next = state.index + delta;
        if (next < 0 || next >= state.lines.length) return;
        state.index = next;
        playCurrent(true);
    }
    function jumpTo(idx, forcePlay = false) {
        if (idx < 0 || idx >= state.lines.length) return;
        state.index = idx;
        if (forcePlay) playCurrent(true);
        else highlight();
    }
    function replayLine() {
        playCurrent(true);
    }
    function advance(delta) {
        state.index += delta;
        if (state.index >= state.lines.length) {
            state.playing = false;
            state.refs.btnPlay.textContent = "Play";
            state.refs.status.textContent = "Finished";
            return;
        }
        playCurrent(true);
    }

    function preloadUpcoming() {
        for (let i = 1; i <= state.preloadCount; i++) {
            const idx = state.index + i;
            if (idx >= state.lines.length) break;
            const line = state.lines[idx];
            if (!line || line._preloaded) continue;
            const url = buildLineURL(line);
            const a = new Audio();
            a.src = url;
            line._preloaded = true;
        }
    }

    // Public bootstrap called by inline script in each page
    window.initRoundtableForQuestion = function (questionKey) {
        init(questionKey);
    };

    // Auto-load if launch button & inline QUESTION_KEY present
    document.addEventListener("DOMContentLoaded", () => {
        const btn = document.getElementById("rt-init-btn");
        // QUESTION_KEY is defined by inline script in each page template
        if (btn && typeof QUESTION_KEY !== "undefined") {
            btn.click();
            // Hide launch section after auto-init
            const launch = document.getElementById("roundtable-launch");
            if (launch) launch.style.display = "none";
        }
    });
})();
