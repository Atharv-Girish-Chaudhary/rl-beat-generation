import os

from data_processing.download_samples import ensure_dirs, LAYER_QUERIES


def test_ensure_dirs():
    """
    Test if directory structure is created correctly.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    output_dir = os.path.join(project_root, "data", "samples")

    ensure_dirs()

    assert os.path.exists(output_dir), "Output directory not created"

    for layer in LAYER_QUERIES:
        layer_path = os.path.join(output_dir, layer)
        assert os.path.exists(layer_path), f"{layer} directory missing"