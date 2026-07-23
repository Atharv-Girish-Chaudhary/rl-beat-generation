import os

from scripts.download_samples import ensure_dirs, LAYER_QUERIES, OUTPUT_DIR


def test_output_dir_is_inside_repo():
    """
    OUTPUT_DIR must resolve to data/samples inside the project root,
    not outside the repo (regression test for the stale two-dirname
    PROJECT_ROOT computation).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    expected = os.path.join(project_root, "data", "samples")

    assert OUTPUT_DIR == expected, (
        f"OUTPUT_DIR resolves to {OUTPUT_DIR}, expected {expected}"
    )


def test_ensure_dirs():
    """
    Test if directory structure is created correctly.
    """
    ensure_dirs()

    assert os.path.exists(OUTPUT_DIR), "Output directory not created"

    for layer in LAYER_QUERIES:
        layer_path = os.path.join(OUTPUT_DIR, layer)
        assert os.path.exists(layer_path), f"{layer} directory missing"
