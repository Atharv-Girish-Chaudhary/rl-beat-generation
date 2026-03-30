import os
import numpy as np

from data_processing.process_groove import process_file


def test_process_file_basic():
    """
    Test if process_file returns valid 8x16 drum grids.
    """

    # ===== locate project root =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # ===== find one MIDI file =====
    groove_dir = os.path.join(project_root, "data", "raw", "groove")

    midi_file = None
    for root, dirs, files in os.walk(groove_dir):
        for file in files:
            if file.endswith(".mid") or file.endswith(".midi"):
                midi_file = os.path.join(root, file)
                break
        if midi_file:
            break

    # ===== assert MIDI exists =====
    assert midi_file is not None, "No MIDI file found for testing"

    # ===== run processing =====
    grids = process_file(midi_file)

    # ===== basic checks =====
    assert isinstance(grids, list), "Output should be a list"

    if len(grids) > 0:
        grid = grids[0]

        assert isinstance(grid, np.ndarray), "Each grid should be a numpy array"

        # Updated to 8x16 to match new drum representation
        assert grid.shape == (8, 16), "Grid shape should be (8, 16)"

        # values should be 0 or 1
        assert np.all((grid == 0) | (grid == 1)), "Grid values must be binary"