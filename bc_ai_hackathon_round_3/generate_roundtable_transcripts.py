import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Input / Output
CLUSTER_JSON_PATH = "clustered_responses.json"
OUTPUT_DIR = "roundtable_transcripts"
MODEL = "gpt-5"
SYSTEM_PROMPT = """You are an expert qualitative research narrator. You will be given a survey question, its type, and clustered open-ended responses with frequencies. Create a creative but faithful 'roundtable discussion' transcript. Each cluster becomes a distinct persona. Do NOT invent facts not implied by responses. Use natural dialogue turns. Summarize recurring identical responses succinctly. Reflect diversity of tone. Conclude with a neutral synthesis paragraph."""

# Prompt template
ROUND_TABLE_TEMPLATE = """
QUESTION TITLE: {title}
QUESTION: {main}
TYPE: {qtype}
FOLLOW_UP: {follow_up}

Generate a dialogue. Each cluster is a participant. Name them Cluster 1, Cluster 2, etc. Use color tags inline like <CLUSTER:Cluster 1>. Keep total length under ~900 words.

Cluster Data:
{cluster_lines}
"""


def load_clusters(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_cluster_lines(responses: List[Dict]) -> str:
    # group by cluster_label
    by_cluster: Dict[str, List[Dict]] = {}
    for r in responses:
        by_cluster.setdefault(r["cluster_label"], []).append(r)
    lines = []
    for label, items in by_cluster.items():
        examples = []
        # sort by frequency desc, then text
        items_sorted = sorted(items, key=lambda x: (-x["frequency"], x["text"]))
        for entry in items_sorted[
            :12
        ]:  # limit examples per cluster to keep prompt size manageable
            freq = entry["frequency"]
            snippet = entry["text"].strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            if freq > 1:
                examples.append(f"(x{freq}) {snippet}")
            else:
                examples.append(snippet)
        lines.append(f"{label}: " + " | ".join(examples))
    return "\n".join(lines)


def generate_transcript(client: OpenAI, question_key: str, meta: Dict) -> str:
    cluster_lines = build_cluster_lines(meta["responses"])
    prompt = ROUND_TABLE_TEMPLATE.format(
        title=meta.get("question_title", question_key),
        main=meta.get("question_main", ""),
        qtype=meta.get("question_type", ""),
        follow_up=meta.get("follow_up", ""),
        cluster_lines=cluster_lines,
    )
    # print(prompt)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # temperature=0.7,
        # max_tokens=1500,
    )
    return response.choices[0].message.content


def main():
    if not Path(CLUSTER_JSON_PATH).exists():
        print(f"Missing {CLUSTER_JSON_PATH}. Run visualizations script first.")
        return

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set in environment.")
        return

    client = OpenAI(api_key=api_key)
    data = load_clusters(CLUSTER_JSON_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    manifest = {}
    for qkey, meta in data.items():
        print(f"Generating transcript for {qkey} ...")
        try:
            transcript = generate_transcript(client, qkey, meta)
            out_path = os.path.join(OUTPUT_DIR, f"roundtable_{qkey}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            manifest[qkey] = {"file": out_path, "title": meta.get("question_title")}
        except Exception as e:
            print(f"  Failed for {qkey}: {e}")

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    print("Done. Transcripts written.")


if __name__ == "__main__":
    main()
