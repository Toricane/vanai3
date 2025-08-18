## BC AI Survey Data Storytelling: 3D Semantic Visualizations

An end‑to‑end data storytelling project turning 1,001 British Columbians' open‑ended survey responses about artificial intelligence into interactive, clusterable 3D semantic maps you can explore in your browser.

The original lightweight README has been replaced with the full, current project documentation (adapted from the inner `bc_ai_hackathon_round_3/README.md`) to reflect the implemented embedding + visualization pipeline. The deprecated implementation summary file has been removed (it duplicated information now integrated here).

---

## 1. Project Overview

Traditional tables or bar charts can’t capture the nuance of thousands of free‑text responses. This project uses modern embedding models + unsupervised learning to reveal thematic structure: similar answers cluster together in a navigable 3D landscape.

You can rotate, zoom, and hover to read authentic voices while visually grasping how perspectives relate.

---

## 2. Dataset Snapshot

Key characteristics:

-   1,001 complete survey responses (BC residents)
-   16 targeted open‑ended text fields (plus corresponding video transcriptions for some questions)
-   Diverse demographics (age, region, education, income)
-   Sentiment and categorical responses (not all visualized here)

Geographic distribution (approx.): Vancouver 76.6% • Victoria 13.7% • Rural BC 9.7%

### Open‑Ended Columns Visualized (16)

AI experience metaphors, sector impacts (positive & negative), creative perceptions, community hopes, AI superpowers, beneficiaries, trusted guides, worries, BC AI future priorities, Indigenous involvement, and advice to leaders (text + limited transcription sets).

---

## 3. Technical Approach

1. Embedding Generation: OpenAI `text-embedding-3-large` (3,072 dims) for each response in all 16 columns.
2. Dimensionality Reduction: 3D t‑SNE (adaptive perplexity per column size).
3. Optimal Clustering: K-Means, sweeping k=2..30, selecting best silhouette score.
4. Response Aggregation: Identical strings merged; marker size ∝ cube root of frequency (perceptually fair volume scaling).
5. Interactive Output: Plotly 3D scatter (one HTML per question) + unified dashboard (`index.html`).
6. Contextual Pages: Each visualization wrapped with metadata (question wording, type, options) from `question_context.py`.

---

## 4. Repository Layout (Essential Parts)

```
.
├── README.md                             # You are here (full consolidated docs)
├── Hackathon round 3 with demos[48].csv  # Raw survey data (input)
├── BC_AI_Survey_Updated.docx             # Survey methodology / instrument
├── bc_ai_hackathon_round_3/
│   ├── embeddings_of_data.py             # Generates embeddings CSV
│   ├── visualizations_3d_silhouette_focus_sized.py  # Final 3D viz pipeline
│   ├── visualize_embeddings*.py          # Earlier / iterative stages
│   ├── validate_data.py                  # Pre‑flight dataset check
│   ├── question_context.py               # Question metadata
│   ├── requirements.txt                  # Dependencies
│   ├── .env.template                     # API key template (added)
│   ├── index.html                        # Dashboard linking all plots
│   └── tsne_visualizations_3d_silhouette_focus_sized/  # Generated HTML plots
└── LICENSE
```

Removed: `IMPLEMENTATION_SUMMARY.md` (its contents are merged here to reduce duplication).

---

## 5. Environment & Setup

Windows PowerShell examples shown (adjust paths as needed).

```powershell
# (Optional) create & activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r bc_ai_hackathon_round_3/requirements.txt
```

Copy and fill environment variables:

```powershell
Copy-Item bc_ai_hackathon_round_3/.env.template bc_ai_hackathon_round_3/.env
# Then edit .env to set OPENAI_API_KEY
```

---

## 6. Data Validation (Optional but Recommended)

Before spending tokens on embeddings, validate the presence and sparsity of target columns:

```powershell
cd bc_ai_hackathon_round_3
python validate_data.py
```

You’ll get counts of non‑empty responses and sample previews.

---

## 7. Generate Embeddings

Cost: Roughly USD $1–2 with `text-embedding-3-large` for this dataset. (Switch to `text-embedding-3-small` in `embeddings_of_data.py` if you need a cheaper run.)

```powershell
python embeddings_of_data.py
```

Output: `bc_ai_data_with_embeddings.csv` (~600MB) stored in `bc_ai_hackathon_round_3/`.

---

## 8. Build 3D Visualizations

```powershell
python visualizations_3d_silhouette_focus_sized.py
```

Outputs:

-   16 Plotly HTML plot files in `tsne_visualizations_3d_silhouette_focus_sized/plots/`
-   16 wrapper context pages `page_<question>.html`
-   Updated `index.html` (central dashboard)

Open the dashboard:

```powershell
start index.html
```

---

## 9. Interpretation & Design Choices

Why t‑SNE over UMAP: t‑SNE provided tighter local neighborhood preservation for small and mid‑sized columns (some under 100 responses) without additional hyperparameter tuning complexity. For future scalability + global structure, UMAP could be added as an alternative.

Why cube root for sizing: Human perception of area / volume is non‑linear; cube root stabilizes marker volume differences so clusters aren’t visually dominated by repeated common phrases (e.g., “Not sure”).

Silhouette score bounds: With very small n the upper bound on k is automatically reduced (`max_k = min(30, n-1)`) to avoid degenerate clustering.

Identical text aggregation: Reduces overplotting, speeds rendering, and highlights consensus frequency directly.

---

## 10. Customization Tips

| Goal                    | Where to Change                                              |
| ----------------------- | ------------------------------------------------------------ |
| Use cheaper embeddings  | Edit `EMBEDDING_MODEL` in `embeddings_of_data.py`            |
| Limit columns processed | Reduce `COLUMNS_TO_EMBED` list                               |
| Try UMAP                | Add `umap-learn` to requirements and modify viz script       |
| Adjust cluster search   | Edit `KMEANS_MAX_K` / `KMEANS_MIN_K` in visualization script |
| Change marker scaling   | Modify `BASE_MARKER_SIZE` and sizing formula                 |
| Add demographic filters | Extend `index.html` with JS + precomputed slices             |

---

## 11. Performance Notes

Approximate runtime on a typical laptop (dependent on API latency):

-   Embeddings: 5–10 minutes
-   Visualization generation: 3–5 minutes
    Memory peak: ~600MB (embedding CSV in memory + intermediate arrays)

---

## 12. Future Enhancements

1. Add demographic facet filtering (client‑side or pre‑clustered subsets)
2. Provide UMAP / PCA toggle for alternative spatial perspectives
3. Integrate lightweight LLM summarization per cluster
4. Export cluster membership CSV for downstream analysis
5. Add search / keyword highlight across plots
6. Bundle a minimal static site build pipeline (e.g., GitHub Pages deploy action)

---

## 13. Minimal Python Snippet (Ad‑hoc Exploration)

```python
import pandas as pd
df = pd.read_csv('Hackathon round 3 with demos[48].csv')
target = 'Q13_AI_impact_worries_text_OE'
sample = df[target].dropna().head(5)
for i, t in enumerate(sample, 1):
  print(f"{i}. {t[:120]}{'...' if len(t)>120 else ''}")
```

---

## 14. License & Responsible Use

See `LICENSE`. Use respectfully—avoid attempts at re‑identification and frame interpretations contextually (clusters show proximity of wording, not moral / factual correctness).

---

## 15. Attribution

BC AI Survey Data Storytelling: 3D Semantic Visualizations (2025).
If you build on this, cite the project and note any methodological changes.

---

## 16. Contact

Please open issues or discussions in this repository for improvements, bugs, or collaboration ideas.

---

Enjoy exploring the semantic landscape of BC’s AI perspectives.
