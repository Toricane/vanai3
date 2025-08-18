"""Generate per-line TTS audio for roundtable transcripts.

Workflow prerequisites:
1. Run visualizations_3d_silhouette_focus_sized.py to produce clustered_responses.json
2. Run generate_roundtable_transcripts.py to produce transcripts in roundtable_transcripts/
3. Run this script to:
   - Derive / request "vibe" guidance per cluster for each question (cached)
   - Parse each transcript into (cluster, line_text)
   - Synthesize audio using gpt-4o-mini-tts (or fallback) with cluster-specific voice + vibe
   - Save audio files into roundtable_audio/<question_key>/line_<NN>.mp3
   - Write an audio_manifest.json describing mapping (for front-end playback / highlighting)

You can re-run safely; existing audio files are skipped unless --force provided.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

CLUSTER_JSON_PATH = "clustered_responses.json"
TRANSCRIPTS_DIR = Path("roundtable_transcripts")
AUDIO_ROOT = Path("roundtable_audio")
AUDIO_MANIFEST = AUDIO_ROOT / "audio_manifest.json"
VIBE_CACHE = AUDIO_ROOT / "cluster_vibes.json"
TTS_MODEL = "gpt-4o-mini-tts"  # user-specified target (native speech endpoint)
USE_CHAT_AUDIO = (
    False  # set True if native speech endpoint unavailable; uses chat audio modality
)
VIBE_MODEL = "gpt-5"  # could be upgraded to gpt-5 if desired
DEFAULT_EXT = ".mp3"
MAX_LINE_CHARS = 900  # guardrail for TTS API
SLEEP_BETWEEN_CALLS = 0.3  # light pacing to avoid rate limits

# Supported voices (per API error message)
SUPPORTED_VOICES = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
    "coral",
    "verse",
    "ballad",
    "ash",
    "sage",
]

# Primary rotation pool (can be reordered for aesthetic variety)
VOICE_POOL = SUPPORTED_VOICES  # reuse full set

VOICE_STYLE_SYSTEM_INSTRUCTION = "You generate concise, one-sentence voice style directives (tone/pacing/emotion) based strictly on clustered verbatim survey responses."
VIBE_PROMPT_TEMPLATE = """Question Title: {title}\nCluster Label: {cluster_label}\nRepresentative Responses (frequency weighted, truncated):\n{examples}\n\nReturn a SINGLE sentence (<=22 words) giving an appropriate narration style (tone, pacing, emotional color) for this cluster's persona. Do not add explanations."""

LINE_PATTERN = re.compile(r"^<CLUSTER:Cluster (\d+)>\s*(.*)")


@dataclass
class LineEntry:
    question_key: str
    cluster_index: int
    cluster_label: str
    text: str
    line_number: int
    audio_path: Optional[str] = None


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def assign_voice(cluster_index: int) -> str:
    """Return a valid voice for a given cluster index.

    If VOICE_POOL contains any deprecated entries (should not now), they will be
    mapped to the first supported voice.
    """
    if not VOICE_POOL:
        return SUPPORTED_VOICES[0]
    voice = VOICE_POOL[cluster_index % len(VOICE_POOL)]
    if voice not in SUPPORTED_VOICES:
        voice = SUPPORTED_VOICES[cluster_index % len(SUPPORTED_VOICES)]
    return voice


def gather_lines(question_key: str, transcript_text: str) -> List[LineEntry]:
    lines = []
    for i, raw in enumerate(transcript_text.splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        m = LINE_PATTERN.match(raw)
        if not m:
            # Non-dialogue (e.g., synthesis paragraph) -> mark cluster -1
            lines.append(LineEntry(question_key, -1, "SYNTHESIS", raw, i))
            continue
        c_idx = int(m.group(1))
        content = m.group(2).strip()
        lines.append(LineEntry(question_key, c_idx, f"Cluster {c_idx}", content, i))
    return lines


def load_transcripts() -> Dict[str, str]:
    transcripts = {}
    for p in TRANSCRIPTS_DIR.glob("roundtable_*.txt"):
        key = p.stem.replace("roundtable_", "")
        transcripts[key] = p.read_text(encoding="utf-8")
    return transcripts


def build_vibe_prompt(
    question_meta: Dict, cluster_label: str, cluster_responses: List[Dict]
) -> str:
    # sort by frequency desc then text, get up to 10 examples
    sorted_resps = sorted(
        cluster_responses, key=lambda r: (-r["frequency"], r["text"])
    )[:10]
    example_lines = []
    for r in sorted_resps:
        snippet = r["text"].replace("\n", " ").strip()
        if len(snippet) > 140:
            snippet = snippet[:137] + "..."
        freq = r["frequency"]
        if freq > 1:
            example_lines.append(f"(x{freq}) {snippet}")
        else:
            example_lines.append(snippet)
    examples = "\n".join(example_lines)
    return VIBE_PROMPT_TEMPLATE.format(
        title=question_meta.get(
            "question_title", question_meta.get("question_main", "")
        ),
        cluster_label=cluster_label,
        examples=examples,
    )


def get_cluster_vibes(client: OpenAI, clusters_data: Dict, force: bool = False) -> Dict:
    if VIBE_CACHE.exists() and not force:
        return load_json(VIBE_CACHE)

    vibes: Dict[str, Dict[str, str]] = {}
    for qkey, meta in clusters_data.items():
        print(f"Vibes: {qkey}")
        # group responses by cluster_label
        cluster_map: Dict[str, List[Dict]] = {}
        for r in meta["responses"]:
            cluster_map.setdefault(r["cluster_label"], []).append(r)
        vibes[qkey] = {}
        for label, resp_list in cluster_map.items():
            prompt = build_vibe_prompt(meta, label, resp_list)
            try:
                result = client.chat.completions.create(
                    model=VIBE_MODEL,
                    messages=[
                        {"role": "system", "content": VOICE_STYLE_SYSTEM_INSTRUCTION},
                        {"role": "user", "content": prompt},
                    ],
                    # temperature=0.5,
                    # max_tokens=100,
                )
                sentence = result.choices[0].message.content.strip().replace("\n", " ")
            except Exception as e:
                sentence = (
                    f"Neutral, clear, mid-paced narration (fallback due to error: {e})."
                )
            vibes[qkey][label] = sentence
            time.sleep(0.15)
    save_json(VIBE_CACHE, vibes)
    return vibes


def synth_line(
    client: OpenAI, line: LineEntry, voice: str, vibe_sentence: Optional[str]
) -> bytes:
    """Synthesize a single line of audio.

    - Pass style guidance via 'instructions' so it is not spoken.
    - Supports native speech endpoint or chat audio fallback.
    """
    tts_input = line.text
    if len(tts_input) > MAX_LINE_CHARS:
        tts_input = tts_input[: MAX_LINE_CHARS - 3] + "..."

    instructions = None
    if vibe_sentence:
        cleaned = vibe_sentence.strip().replace("\n", " ")
        if len(cleaned) > 180:
            cleaned = cleaned[:177] + "..."
        instructions = cleaned

    audio_bytes: Optional[bytes] = None

    try:
        if not USE_CHAT_AUDIO:
            # Native endpoint first
            try:
                speech = client.audio.speech.create(
                    model=TTS_MODEL,
                    voice=voice,
                    input=tts_input,
                    **({"instructions": instructions} if instructions else {}),
                )
            except TypeError:
                # Retry without instructions if unsupported
                speech = client.audio.speech.create(
                    model=TTS_MODEL,
                    voice=voice,
                    input=tts_input,
                )

            for attr in ("audio", "content", "data"):
                if hasattr(speech, attr):
                    candidate = getattr(speech, attr)
                    if isinstance(candidate, (bytes, bytearray)):
                        audio_bytes = bytes(candidate)
                        break
            if audio_bytes is None and hasattr(speech, "read"):
                audio_bytes = speech.read()
            if audio_bytes is None and isinstance(speech, dict):
                maybe = speech.get("audio") or speech.get("content")
                if isinstance(maybe, (bytes, bytearray)):
                    audio_bytes = bytes(maybe)
        else:
            # Chat audio fallback
            chat_resp = client.chat.completions.create(
                model=TTS_MODEL,
                modalities=["text", "audio"],
                audio={"voice": voice, "format": "mp3"},
                messages=[
                    {
                        "role": "system",
                        "content": "You convert user content to spoken audio only.",
                    },
                    {
                        "role": "user",
                        "content": (instructions + "\n" + tts_input)
                        if instructions
                        else tts_input,
                    },
                ],
                max_tokens=1,
            )
            try:
                audio_part = chat_resp.choices[0].message.audio
                if hasattr(audio_part, "data") and isinstance(
                    audio_part.data, (bytes, bytearray)
                ):
                    audio_bytes = bytes(audio_part.data)
                elif hasattr(audio_part, "b64_json"):
                    import base64

                    audio_bytes = base64.b64decode(audio_part.b64_json)
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"TTS request failed: {e}")

    if audio_bytes is None:
        raise RuntimeError("Could not extract audio bytes from TTS response")
    return audio_bytes


def process(
    force: bool = False,
    limit_questions: Optional[List[str]] = None,
    skip_vibes: bool = False,
):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set.")
        return
    client = OpenAI(api_key=api_key)

    if not Path(CLUSTER_JSON_PATH).exists():
        print("Missing clustered_responses.json. Run visualization script first.")
        return
    clusters_data = load_json(Path(CLUSTER_JSON_PATH))
    transcripts = load_transcripts()

    if limit_questions:
        clusters_data = {k: v for k, v in clusters_data.items() if k in limit_questions}
        transcripts = {k: v for k, v in transcripts.items() if k in clusters_data}

    vibes = {} if skip_vibes else get_cluster_vibes(client, clusters_data, force=force)

    AUDIO_ROOT.mkdir(exist_ok=True, parents=True)
    manifest = (
        load_json(AUDIO_MANIFEST) if AUDIO_MANIFEST.exists() and not force else {}
    )

    for qkey, transcript_text in transcripts.items():
        print(f"Audio: {qkey}")
        q_dir = AUDIO_ROOT / qkey
        q_dir.mkdir(parents=True, exist_ok=True)

        lines = gather_lines(qkey, transcript_text)
        for line in lines:
            voice = (
                assign_voice(line.cluster_index)
                if line.cluster_index >= 0
                else assign_voice(0)
            )
            vibe_sentence = (
                None
                if skip_vibes or line.cluster_index < 0
                else vibes.get(qkey, {}).get(line.cluster_label)
            )
            filename = f"line_{line.line_number:03d}_c{line.cluster_index if line.cluster_index >= 0 else 'S'}.mp3"
            out_path = q_dir / filename
            if out_path.exists() and not force:
                line.audio_path = str(out_path)
                continue
            try:
                audio_bytes = synth_line(client, line, voice, vibe_sentence)
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                line.audio_path = str(out_path)
                time.sleep(SLEEP_BETWEEN_CALLS)
            except Exception as e:
                print(f"  Line {line.line_number} failed: {e}")
                continue
        # Update manifest
        manifest[qkey] = {
            "question_title": clusters_data[qkey].get("question_title"),
            "lines": [
                {
                    "line_number": line_entry.line_number,
                    "cluster_index": line_entry.cluster_index,
                    "cluster_label": line_entry.cluster_label,
                    "voice": assign_voice(
                        line_entry.cluster_index if line_entry.cluster_index >= 0 else 0
                    ),
                    "vibe": (
                        None
                        if skip_vibes
                        else vibes.get(qkey, {}).get(line_entry.cluster_label)
                    ),
                    "audio": line_entry.audio_path,
                    "text": line_entry.text,
                }
                for line_entry in lines
            ],
        }
    save_json(AUDIO_MANIFEST, manifest)
    print("Audio generation complete. Manifest written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TTS audio for roundtable transcripts"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-generate all audio & vibes"
    )
    parser.add_argument(
        "--skip-vibes",
        action="store_true",
        help="Skip vibe generation & use neutral tone",
    )
    parser.add_argument(
        "--questions", nargs="*", help="Subset of question keys to process"
    )
    args = parser.parse_args()
    process(
        force=args.force, limit_questions=args.questions, skip_vibes=args.skip_vibes
    )
