import os
import numpy as np
import pretty_midi

# ===== Path setup =====
# This script is located in data_processing/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root is the parent directory of data_processing/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Input: raw Groove MIDI files
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "groove")

# Output: processed drum grids
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "groove_grids.npy")

# ===== Drum pitch mapping =====
# Updated to 8 channels so Groove MIDI is transformed into an 8x16 grid
KICK_PITCHES = [35, 36]
SNARE_PITCHES = [38, 40]
CLOSED_HIHAT_PITCHES = [42, 44]
OPEN_HIHAT_PITCHES = [46]
CLAP_PITCHES = [39]
TOM_PITCHES = [41, 43, 45, 47, 48, 50]
CYMBAL_PITCHES = [49, 51, 52, 53, 55, 57, 59]
PERC_OTHER_PITCHES = [
    54, 56, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81
]


def pitch_to_channel(pitch):
    """
    Map a MIDI pitch to one of eight drum channels.
    0 = kick
    1 = snare
    2 = closed hi-hat
    3 = open hi-hat
    4 = clap
    5 = tom
    6 = cymbal
    7 = other percussion
    """
    if pitch in KICK_PITCHES:
        return 0
    if pitch in SNARE_PITCHES:
        return 1
    if pitch in CLOSED_HIHAT_PITCHES:
        return 2
    if pitch in OPEN_HIHAT_PITCHES:
        return 3
    if pitch in CLAP_PITCHES:
        return 4
    if pitch in TOM_PITCHES:
        return 5
    if pitch in CYMBAL_PITCHES:
        return 6
    if pitch in PERC_OTHER_PITCHES:
        return 7
    return -1


def process_file(midi_path):
    """
    Convert one MIDI file into a list of 8x16 drum grids.
    Each grid represents one 4-beat segment.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)

    drum_inst = None
    for inst in pm.instruments:
        if inst.is_drum:
            drum_inst = inst
            break

    if drum_inst is None:
        return []

    beats = pm.get_beats()
    if len(beats) < 5:
        return []

    grids = []

    # Use every 4 beats as one segment
    for i in range(0, len(beats) - 4, 4):
        start = beats[i]
        end = beats[i + 4]

        # Grid shape: 8 channels x 16 time steps
        # This matches the required 8x16 Groove representation
        grid = np.zeros((8, 16), dtype=np.float32)

        for note in drum_inst.notes:
            if not (start <= note.start < end):
                continue

            channel = pitch_to_channel(note.pitch)
            if channel == -1:
                continue

            relative = (note.start - start) / (end - start)
            step = int(relative * 16)

            if step < 0:
                step = 0
            if step > 15:
                step = 15

            grid[channel][step] = 1.0

        grids.append(grid)

    return grids


def main():
    # Check whether the input directory exists
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    midi_files = []

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                midi_files.append(os.path.join(root, file))

    print(f"Found {len(midi_files)} MIDI files")

    all_grids = []

    for path in midi_files:
        grids = process_file(path)
        all_grids.extend(grids)

    if len(all_grids) == 0:
        print("No valid drum grids found")
        return

    all_grids = np.array(all_grids, dtype=np.float32)
    print("Output shape:", all_grids.shape)

    # Make sure the output directory exists before saving
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    np.save(OUTPUT_PATH, all_grids)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()