import os
import json
import time
import requests

API_KEY = os.environ.get("FREESOUND_API_KEY")
BASE_URL = "https://freesound.org/apiv2/search/text/"

OUTPUT_DIR = "data/samples"
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.json")

LAYER_QUERIES = {
    "kick": "kick drum one shot",
    "snare": "snare drum one shot",
    "hihat": "hi hat one shot",
    "clap": "clap one shot",
    "bass": "bass one shot",
    "melody": "synth stab one shot",
    "pad": "pad one shot",
    "fx": "fx one shot",
}

SAMPLES_PER_LAYER = 15


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for layer in LAYER_QUERIES:
        os.makedirs(os.path.join(OUTPUT_DIR, layer), exist_ok=True)


def search_sounds(query, page_size=50):
    headers = {
        "Authorization": f"Token {API_KEY}"
    }

    params = {
        "query": query,
        "filter": "duration:[0 TO 5] channels:1 OR channels:2",
        "sort": "downloads_desc",
        "page_size": page_size,
        "fields": "id,name,previews,license,username,duration,type,tags"
    }

    response = requests.get(BASE_URL, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def safe_name(name):
    cleaned = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        elif ch in (" ", "."):
            cleaned.append("_")
    result = "".join(cleaned).strip("_")
    if not result:
        result = "sample"
    return result[:80]


def download_file(url, out_path):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def already_have_enough(layer_dir, target_count):
    files = []
    for name in os.listdir(layer_dir):
        if name.endswith(".mp3") or name.endswith(".wav") or name.endswith(".ogg"):
            files.append(name)
    return len(files) >= target_count


def main():
    if not API_KEY:
        raise ValueError("FREESOUND_API_KEY is not set in the environment.")

    ensure_dirs()

    manifest = []

    for layer, query in LAYER_QUERIES.items():
        print(f"\nLayer: {layer}")
        print(f"Query: {query}")

        layer_dir = os.path.join(OUTPUT_DIR, layer)

        if already_have_enough(layer_dir, SAMPLES_PER_LAYER):
            print(f"Skipping {layer}, already has enough files")
            continue

        data = search_sounds(query)
        results = data.get("results", [])

        count = 0
        used_ids = set()

        for sound in results:
            if count >= SAMPLES_PER_LAYER:
                break

            sound_id = sound.get("id")
            if sound_id in used_ids:
                continue

            previews = sound.get("previews", {})
            preview_url = previews.get("preview-hq-mp3")

            if not preview_url:
                continue

            base_name = safe_name(sound.get("name", f"{layer}_{sound_id}"))
            filename = f"{count:02d}_{sound_id}_{base_name}.mp3"
            out_path = os.path.join(layer_dir, filename)

            try:
                download_file(preview_url, out_path)

                manifest.append({
                    "layer": layer,
                    "index": count,
                    "sound_id": sound_id,
                    "filename": filename,
                    "path": out_path,
                    "query": query,
                    "license": sound.get("license"),
                    "username": sound.get("username"),
                    "duration": sound.get("duration"),
                    "tags": sound.get("tags", []),
                    "preview_url": preview_url,
                })

                used_ids.add(sound_id)
                count += 1
                print(f"Downloaded {filename}")

                time.sleep(0.2)

            except Exception as e:
                print(f"Failed to download sound {sound_id}: {e}")

        print(f"Saved {count} samples for {layer}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {MANIFEST_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()