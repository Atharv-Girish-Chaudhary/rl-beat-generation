from setuptools import setup, find_packages

# torch/torchaudio are intentionally NOT listed here — install them first with the
# platform-appropriate command (see README "Setup"). requirements.txt is the full
# Linux/CUDA lockfile.
setup(
    name="beat_rl",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium",
        "numpy",
        "matplotlib",
        "librosa",
        "soundfile",
        "mido",
        "pretty_midi",
        "PyYAML",
    ],
)
