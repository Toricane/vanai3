import ast
import os
import textwrap

import numpy as np
import pandas as pd
import plotly.express as px
from question_context import question_context_data
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# --- Configuration ---
INPUT_CSV_PATH = "bc_ai_data_with_embeddings.csv"
ORIGINAL_TEXT_COLUMNS = list(question_context_data.keys())
OUTPUT_HTML_DIR = "tsne_visualizations_3d_silhouette_focus_sized"
MAIN_INDEX_PATH = "index.html"

TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000
TSNE_LEARNING_RATE = "auto"
TSNE_INIT = "pca"
RANDOM_STATE = 42

KMEANS_MAX_K = 30
KMEANS_MIN_K = 2

HOVER_MAX_LINE_WIDTH = 60
BASE_MARKER_SIZE = 5


# --- Color Utility ---
def build_discrete_palette(required_n: int):
    """Return a list of at least required_n distinct colors.
    Combines multiple Plotly qualitative palettes and, if needed, appends
    procedurally generated HSL colors to ensure enough distinct values.
    """
    base = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.D3
        + px.colors.qualitative.G10
        + px.colors.qualitative.T10
        + px.colors.qualitative.Set3
        + px.colors.qualitative.Dark24
    )
    if required_n <= len(base):
        return base[:required_n]
    # Generate extra colors evenly spaced around the hue wheel
    extra_needed = required_n - len(base)
    extra = [f"hsl({int(360 * i / extra_needed)},70%,50%)" for i in range(extra_needed)]
    return base + extra


# --- HTML Helper Functions ---
def get_question_context(column_name):
    return question_context_data.get(column_name, {})


def create_context_html(context):
    if not context:
        return ""

    choices_html = ""
    if "choices" in context and context["choices"]:
        choices_list = "".join([f"<li>{item}</li>" for item in context["choices"]])
        choices_html = f"<h4>Answer Choices:</h4><ul>{choices_list}</ul>"

    html = f"""
    <div class="context-box">
        <h2>{context.get("title", "No Title")}</h2>
        <p><strong>Question Type:</strong> {context.get("type", "N/A")}</p>
        <hr>
        <h3>Survey Question:</h3>
        <p><em>{context.get("main_question", "N/A")}</em></p>
    """
    if "follow_up" in context:
        html += f"<p><em>↳ {context.get('follow_up', '')}</em></p>"

    html += f"""
        {choices_html}
        <h3>Description:</h3>
        <p>{context.get("description", "No description available.")}</p>
    </div>
    """
    return html


def create_main_html_with_iframe(context, plot_filename_only):
    context_html = create_context_html(context)
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{context.get("title", "AI Survey Visualization")}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .context-box {{
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }}
        h2, h3, h4 {{
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        iframe {{
            width: 100%;
            height: 80vh;
            border: none;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }}
        .back-link {{
            display: inline-block;
            margin-bottom: 20px;
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            text-decoration: none;
            border-radius: 10px;
            font-weight: bold;
            transition: background 0.3s;
        }}
        .back-link:hover {{
            background: rgba(255, 255, 255, 0.4);
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="../{MAIN_INDEX_PATH}" class="back-link">← Back to Main Dashboard</a>
        {context_html}
        <iframe src="plots/{plot_filename_only}"></iframe>
    </div>
</body>
</html>
"""


# --- Data Processing Helper Functions ---
def parse_embedding_string(embedding_str):
    if (
        pd.isna(embedding_str)
        or not isinstance(embedding_str, str)
        or not embedding_str.startswith("[")
    ):
        return None
    try:
        return ast.literal_eval(embedding_str)
    except (ValueError, SyntaxError):
        return None


def wrap_text_for_hover(text, width):
    if not isinstance(text, str):
        return str(text)
    return "<br>".join(textwrap.wrap(text, width=width))


# --- Main Script ---
def main():
    output_plot_dir = os.path.join(OUTPUT_HTML_DIR, "plots")
    os.makedirs(output_plot_dir, exist_ok=True)
    print(f"HTML visualizations will be saved in: '{OUTPUT_HTML_DIR}/'")
    print(f"Plot files will be saved in: '{output_plot_dir}/'")

    try:
        df_full = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File not found: {INPUT_CSV_PATH}")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    for original_col_name in ORIGINAL_TEXT_COLUMNS:
        embedding_col_name = f"{original_col_name}_embedding"
        print(f"\nProcessing: '{original_col_name}'")

        context = get_question_context(original_col_name)
        if not context:
            print(f"  Warning: No context found for '{original_col_name}'. Skipping.")
            continue

        if (
            original_col_name not in df_full.columns
            or embedding_col_name not in df_full.columns
        ):
            print(f"  Warning: Missing columns for '{original_col_name}'. Skipping.")
            continue

        df_question = df_full[[original_col_name, embedding_col_name]].copy()
        df_question["text_is_valid"] = df_question[original_col_name].apply(
            lambda x: isinstance(x, str) and x.strip() and x.strip().lower() != "n/a"
        )
        df_filtered_text = df_question[df_question["text_is_valid"]].copy()

        if df_filtered_text.empty:
            print("  No valid text found. Skipping.")
            continue

        df_filtered_text["parsed_embedding"] = df_filtered_text[
            embedding_col_name
        ].apply(parse_embedding_string)
        df_valid_embeddings = df_filtered_text.dropna(subset=["parsed_embedding"])
        df_valid_embeddings = df_valid_embeddings[
            df_valid_embeddings["parsed_embedding"].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            )
        ]

        if df_valid_embeddings.empty:
            print("  No valid embeddings. Skipping.")
            continue

        embedding_matrix = np.array(df_valid_embeddings["parsed_embedding"].tolist())
        num_samples = embedding_matrix.shape[0]
        print(f"  Working with {num_samples} valid samples.")

        current_perplexity = min(TSNE_PERPLEXITY, max(1, num_samples - 1))
        if num_samples <= 3:
            print(f"  Too few samples ({num_samples}) for t-SNE. Skipping.")
            continue

        print(f"  Running 3D t-SNE with perplexity {current_perplexity}...")
        tsne = TSNE(
            n_components=3,
            perplexity=current_perplexity,
            max_iter=TSNE_MAX_ITER,
            learning_rate=TSNE_LEARNING_RATE,
            init=TSNE_INIT,
            random_state=RANDOM_STATE,
        )
        tsne_3d = tsne.fit_transform(embedding_matrix)

        max_k = min(KMEANS_MAX_K, num_samples - 1)
        min_k = min(KMEANS_MIN_K, max_k)

        if min_k >= max_k:
            print(
                f"  Cannot perform clustering (min_k={min_k}, max_k={max_k}). Skipping."
            )
            continue

        best_k, best_silhouette = min_k, -1
        print(f"  Finding optimal clusters (k={min_k} to {max_k})...")
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            cluster_labels = kmeans.fit_predict(tsne_3d)
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(tsne_3d, cluster_labels)
                if silhouette_avg > best_silhouette:
                    best_silhouette, best_k = silhouette_avg, k

        print(f"  Best k: {best_k} (silhouette score: {best_silhouette:.3f})")

        kmeans_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
        final_labels = kmeans_final.fit_predict(tsne_3d)

        plot_df = pd.DataFrame(
            {
                "x": tsne_3d[:, 0],
                "y": tsne_3d[:, 1],
                "z": tsne_3d[:, 2],
                "text": df_valid_embeddings[original_col_name].tolist(),
                "cluster": final_labels,
                "cluster_label": [
                    f"Cluster {c}" for c in final_labels
                ],  # categorical label
            }
        )

        text_groups = (
            plot_df.groupby("text")
            .agg(
                x=("x", "mean"),
                y=("y", "mean"),
                z=("z", "mean"),
                cluster=("cluster", "first"),
                cluster_label=("cluster_label", "first"),
                frequency=("text", "size"),
            )
            .reset_index()
        )

        text_groups["marker_size"] = BASE_MARKER_SIZE * np.cbrt(
            text_groups["frequency"]
        )
        text_groups["hover_text"] = text_groups.apply(
            lambda row: f"<b>Frequency:</b> {int(row['frequency'])}<br><b>{row['cluster_label']}</b><br><b>Text:</b><br>{wrap_text_for_hover(row['text'], HOVER_MAX_LINE_WIDTH)}",
            axis=1,
        )

        print("  Creating 3D visualization (discrete colors)...")
        palette = build_discrete_palette(best_k)
        fig = px.scatter_3d(
            text_groups,
            x="x",
            y="y",
            z="z",
            color="cluster_label",
            size="marker_size",
            hover_data={"x": False, "y": False, "z": False},
            hover_name="hover_text",
            title=f"<b>{context.get('title', original_col_name)}</b><br><sub>3D t-SNE Visualization of Responses</sub>",
            color_discrete_sequence=palette,
        )
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            legend_title_text="Clusters",
            scene=dict(
                xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3"
            ),
        )
        fig.update_traces(
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.8)",
                font_size=12,
                font_family="Arial",
                bordercolor="rgba(0, 0, 0, 0.5)",
            )
        )

        plot_filename_only = f"plot_{original_col_name}.html"
        plot_filepath = os.path.join(output_plot_dir, plot_filename_only)
        fig.write_html(plot_filepath, include_plotlyjs="cdn")
        print(f"  Saved plot to: {plot_filepath}")

        main_html_filename = f"page_{original_col_name}.html"
        main_html_filepath = os.path.join(OUTPUT_HTML_DIR, main_html_filename)
        main_html_content = create_main_html_with_iframe(context, plot_filename_only)

        with open(main_html_filepath, "w", encoding="utf-8") as f:
            f.write(main_html_content)
        print(f"  Saved context page to: {main_html_filepath}")

    print("\n--- All visualizations generated successfully! ---")


if __name__ == "__main__":
    main()
