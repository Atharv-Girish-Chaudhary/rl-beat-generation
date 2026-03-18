import os
import numpy as np
import pretty_midi

INPUT_DIR = "data/raw/groove"
OUTPUT_PATH = "data/processed/groove_grids.npy"

KICK_PITCHES = [35, 36]
SNARE_PITCHES = [38, 40]
HIHAT_PITCHES = [42, 44, 46]
OTHER_PITCHES = [41, 43, 45, 47, 48, 50, 49, 51]


def pitch_to_channel(pitch):
    if pitch in KICK_PITCHES:
        return 0
    if pitch in SNARE_PITCHES:
        return 1
    if pitch in HIHAT_PITCHES:
        return 2
    if pitch in OTHER_PITCHES:
        return 3
    return -1


def process_file(midi_path):
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

    for i in range(0, len(beats) - 4, 4):
        start = beats[i]
        end = beats[i + 4]

        grid = np.zeros((4, 16), dtype=np.float32)

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
    midi_files = []

    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".mid"):
                midi_files.append(os.path.join(root, file))

    print(f"Found {len(midi_files)} MIDI files")

    all_grids = []

    for path in midi_files:
        grid = process_file(path)
        all_grids.extend(grid)

    if len(all_grids) == 0:
        print("No valid drum grids found")
        return

    all_grids = np.array(all_grids, dtype=np.float32)
    print("Output shape:", all_grids.shape)

    np.save(OUTPUT_PATH, all_grids)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()