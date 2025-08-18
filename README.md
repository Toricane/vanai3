## BC AI Survey Data Storytelling

Interactive semantic maps + narrated roundtable dialogues built from 1,001 British Columbians’ open‑ended responses about artificial intelligence.

Two core experiences:

1. 3D Semantic Maps – Explore clustered answer space (rotate / zoom / hover) to see how ideas group.
2. Roundtable Dialogues – Auto‑generated multi‑voice conversation between cluster personas with per‑line audio + synthesis summary.

---

## 1. Quick Start

```powershell
# (optional) virtual env
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r bc_ai_hackathon_round_3/requirements.txt
Copy-Item bc_ai_hackathon_round_3/.env.template bc_ai_hackathon_round_3/.env
# edit .env -> set OPENAI_API_KEY

cd bc_ai_hackathon_round_3
python validate_data.py                     # optional sanity check
python embeddings_of_data.py                # create embeddings CSV
python visualizations_3d_silhouette_focus_sized.py  # build maps + clustered_responses.json
python generate_roundtable_transcripts.py   # create dialogue transcripts
python generate_audio_from_transcripts.py   # synthesize per‑line audio
```

Serve locally (one option):

```powershell
python -m http.server 8000
# open http://localhost:8000/bc_ai_hackathon_round_3/index.html
```

---

## 2. Why This Exists

Free‑text is messy, rich, and resistant to flat counts. By embedding responses into semantic space then clustering, we expose _shape and nuance_: which hopes, worries, and priorities co‑occur or diverge. The roundtable narration then bridges analytic structure and human storytelling.

---

## 3. Data Snapshot

| Metric                 | Value                                      |
| ---------------------- | ------------------------------------------ |
| Respondents            | 1,001 (BC residents)                       |
| Open‑ended text fields | 16 (plus select video transcript variants) |
| Embedding model        | `text-embedding-3-large` (3,072 dims)      |
| Clustering method      | K‑Means (silhouette‑optimized k)           |
| Visualization          | 3D t‑SNE + frequency‑scaled centroids      |

Themes covered: personal metaphors for AI, sector impacts (±), creativity, community benefit, desired “superpowers,” perceived beneficiaries, trust, worries, future priorities, Indigenous involvement, advice to leaders.

---

## 4. Feature Overview

### 4.1 Semantic Maps

-   Adaptive perplexity 3D t‑SNE.
-   Automatic k selection (2–30) via silhouette score.
-   Aggregation of identical strings (size = cube‑root(frequency)).
-   Color palette mapped deterministically to clusters (reused in dialogue legend).

### 4.2 Roundtable Dialogues & Audio

-   Transcript per question: alternating cluster personas + final synthesis line.
-   One descriptive “vibe” sentence per cluster (cached) guides tone.
-   Deterministic voice assignment from supported pool (alloy, echo, fable, onyx, nova, shimmer, coral, verse, ballad, ash, sage).
-   Per‑line MP3 + manifest enabling progressive playback, highlighting, speed control, auto‑scroll, keyboard shortcuts (Space / ← / → / R).

### 4.3 Consolidated Export

`clustered_responses.json` supplies question metadata, cluster colors, and aggregated responses → single source for transcripts + audio build.

---

## 5. Processing & Narrative Pipeline

1. Validate & clean text (drop empty / placeholder; aggregate identical answers).
2. Embed each unique response (3,072‑D vector).
3. Reduce to 3D with t‑SNE (per‑question adaptive perplexity).
4. Sweep K‑Means k=2..30 → choose silhouette max.
5. Build plot dataset: centroid positions, frequency, cluster, color.
6. Export `clustered_responses.json` (metadata, clusters, responses).
7. Generate cluster vibe sentences (cached) + persona transcript.
8. Synthesize per‑line audio (style via instructions; not spoken).
9. Write `audio_manifest.json` linking lines → (cluster, text, voice, vibe, path).
10. Front‑end loads manifest + JSON → renders legend, transcript, playback.

---

## 6. Key Design Choices

| Decision                    | Rationale                                                                        |
| --------------------------- | -------------------------------------------------------------------------------- |
| t‑SNE (3D)                  | Strong local neighborhood feel for moderate sample sizes; intuitive exploration. |
| Silhouette k selection      | Removes manual tuning; guards against over‑fragmentation.                        |
| Cube‑root size scaling      | Prevents heavy repetition dominating visual salience.                            |
| Aggregate identical strings | Speeds rendering + reveals consensus strength.                                   |
| Cached vibe sentences       | Cuts repeated LLM cost during audio regeneration.                                |
| Per‑line audio files        | Enables granular preload, skipping, highlighting sync.                           |

---

## 7. Repository Layout (Essentials)

```
bc_ai_hackathon_round_3/
 ├─ embeddings_of_data.py
 ├─ visualizations_3d_silhouette_focus_sized.py
 ├─ generate_roundtable_transcripts.py
 ├─ generate_audio_from_transcripts.py
 ├─ question_context.py
 ├─ roundtable_player.js / roundtable_player.css
 ├─ tsne_visualizations_3d_silhouette_focus_sized/
 │   ├─ plots/ (raw Plotly plots)
 │   └─ page_*.html (context + playback embedded)
 ├─ roundtable_transcripts/
 ├─ roundtable_audio/
 │   ├─ audio_manifest.json
 │   └─ <Q>/line_XXX_cY.mp3
 ├─ clustered_responses.json
 └─ requirements.txt
```

Root files: survey CSV, LICENSE, this README.

---

## 8. Running Individual Steps

| Task                    | Command (PowerShell)                                 | Output                                  |
| ----------------------- | ---------------------------------------------------- | --------------------------------------- |
| Validate                | `python validate_data.py`                            | Console stats                           |
| Embeddings              | `python embeddings_of_data.py`                       | `bc_ai_data_with_embeddings.csv`        |
| Visualizations + export | `python visualizations_3d_silhouette_focus_sized.py` | Plot pages + `clustered_responses.json` |
| Transcripts             | `python generate_roundtable_transcripts.py`          | `roundtable_transcripts/` + manifest    |
| Audio                   | `python generate_audio_from_transcripts.py`          | `roundtable_audio/` MP3s + manifest     |

Selective question audio:

```powershell
python generate_audio_from_transcripts.py --questions Q10_AI_superpower_to_help_community_other_OE
```

Force rebuild of vibes + audio:

```powershell
python generate_audio_from_transcripts.py --force
```

Neutral tone (skip vibes):

```powershell
python generate_audio_from_transcripts.py --skip-vibes
```

---

## 9. Customization Cheatsheet

| Goal                         | Where                                         |
| ---------------------------- | --------------------------------------------- |
| Use cheaper embeddings       | `embeddings_of_data.py` `EMBEDDING_MODEL`     |
| Reduce processed questions   | Adjust list of columns in embedding script    |
| Change cluster search bounds | `KMEANS_MIN_K` / `KMEANS_MAX_K` in viz script |
| Switch to UMAP               | Add `umap-learn`; replace t‑SNE section       |
| Different marker scaling     | Modify `BASE_MARKER_SIZE` & sizing formula    |
| Additional playback voices   | Extend allowed voice pool in audio script     |

---

## 10. Performance (Indicative)

| Stage                 | Approx Time                                   |
| --------------------- | --------------------------------------------- |
| Validation            | < 1 min                                       |
| Embeddings            | 5–10 min (API latency dominated)              |
| Visualization build   | 3–5 min                                       |
| Transcript generation | ~2–4 min (LLM)                                |
| Audio synthesis       | Depends on length (~1–3 min per question set) |

Memory peak during visualization: ~600MB.

---

## 11. Responsible Use & Limitations

-   Clusters reflect **linguistic proximity**, not correctness or endorsement.
-   t‑SNE layout is stochastic (fixed seed used here for reproducibility) and emphasizes local neighborhoods—distances between far clusters are less reliable.
-   The roundtable dialogue is synthetic: it paraphrases & composites themes; avoid quoting it as verbatim respondent speech.
-   Do not attempt re‑identification; treat results as aggregated public opinion texture, not individual profiling.

---

## 12. Minimal Exploration Snippet

```python
import pandas as pd
df = pd.read_csv('Hackathon round 3 with demos[48].csv')
col = 'Q13_AI_impact_worries_text_OE'
for i,t in enumerate(df[col].dropna().head(5),1):
  print(f"{i}. {t[:100]}{'…' if len(t)>100 else ''}")
```

---

## 13. Changelog (Final Phase Highlights)

-   Added consolidated `clustered_responses.json` export.
-   Implemented silhouette‑optimized clustering & frequency scaling.
-   Added transcript generator + per‑cluster vibe caching.
-   Added per‑line TTS audio pipeline with deterministic voice mapping.
-   Integrated roundtable playback module (legend, controls, auto‑init, highlighting).
-   Enhanced landing `index.html` with feature cards + dynamic response/cluster counts.

---

## 14. License & Attribution

Apache 2.0 (see `LICENSE`). Cite as: _BC AI Survey Data Storytelling (2025) – Semantic Maps & Roundtable Dialogues._

---

## 15. Contact

Open an issue or discussion for questions, improvements, or adaptations.

---

Enjoy exploring the semantic landscape—and now, hearing the thematic conversation—of BC’s AI perspectives.
