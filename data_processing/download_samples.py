import os
import json
import requests
import time

# ===== Path setup =====
# This script is located in data_processing/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root is the parent directory of data_processing/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Output directory for downloaded samples
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "samples")

# Manifest file to track downloaded files
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.json")


# ===== Layer queries =====
# Each key represents a sound category (layer)
# Values are search keywords used to query Freesound API
LAYER_QUERIES = {
    "kick": "kick drum",
    "snare": "snare drum",
    "hihat": "hi hat",
    "clap": "clap",
    "bass": "bass",
    "melody": "melody",
    "pad": "pad",
    "fx": "fx"
}


def ensure_dirs():
    """
    Create output directory and subdirectories for each layer.
    This ensures all required folders exist before downloading.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for layer in LAYER_QUERIES:
        layer_path = os.path.join(OUTPUT_DIR, layer)
        os.makedirs(layer_path, exist_ok=True)


def load_manifest():
    """
    Load manifest file if it exists.
    The manifest keeps track of already downloaded files
    to avoid duplicates.
    """
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    """
    Save updated manifest to disk.
    """
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def download_samples(api_key, num_per_layer=10):
    """
    Download audio samples from Freesound API for each layer.

    Args:
        api_key (str): Freesound API key
        num_per_layer (int): number of samples to download per layer
    """
    headers = {"Authorization": f"Token {api_key}"}

    manifest = load_manifest()

    for layer, query in LAYER_QUERIES.items():
        print(f"Downloading {layer} samples...")

        layer_dir = os.path.join(OUTPUT_DIR, layer)

        # API request
        url = "https://freesound.org/apiv2/search/text/"
        params = {
            "query": query,
            "fields": "id,name,previews",
            "page_size": num_per_layer
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Failed to fetch {layer}: {response.status_code}")
            continue

        results = response.json().get("results", [])

        for sound in results:
            sound_id = str(sound["id"])

            # Skip if already downloaded
            if sound_id in manifest:
                continue

            preview_url = sound["previews"]["preview-hq-mp3"]
            file_path = os.path.join(layer_dir, f"{sound_id}.mp3")

            try:
                audio_data = requests.get(preview_url).content

                with open(file_path, "wb") as f:
                    f.write(audio_data)

                manifest[sound_id] = {
                    "layer": layer,
                    "file": file_path
                }

                print(f"Downloaded {file_path}")

                # Sleep to avoid hitting API rate limits
                time.sleep(0.5)

            except Exception as e:
                print(f"Error downloading {sound_id}: {e}")

    save_manifest(manifest)
    print("Download complete.")


def main():
    """
    Entry point for script execution.
    Replace YOUR_API_KEY with your Freesound API key.
    """
    API_KEY = "YOUR_API_KEY"

    ensure_dirs()
    download_samples(API_KEY)


if __name__ == "__main__":
    main()