# Future Work

Tracked improvements beyond the current shippable baseline. The CS 5180 submission and Phase 1/2 milestones are complete; everything below is voluntary polish, research, or coordinated team improvements.

## Quick Wins (solo, low effort)

### Fresh-clone reproducibility test
Verify macOS and Linux/CUDA install paths in the README resolve on a clean machine. Clone to a fresh directory, run setup, smoke-test:
- `streamlit run app.py`
- `python evaluation/evaluate.py --phase 1` and `--phase 2`
- `pytest tests/`

### Compiled report as GitHub Release
Compile `docs/beat_report.tex`, upload the PDF as a release asset (e.g. tag `v1.0-report`), link from the README. Lets viewers read the report without installing TeX.

### `extras_require` refactor in `setup.py`
Split dependencies into `install_requires` (core / inference) and `extras_require` groups (`train`, `demo`, `data`, `dev`). Demote `requirements.txt` to an explicit Linux/CUDA lockfile. Update README install paths to match.

## Research Extensions (coordinated, multi-day+)

### Phase 2 density-spam fix
Phase 2 agent exploits high beat density for reward instead of learning groove — root cause and proposed fix in `docs/phase2_diagnostic.md`. **Owner:** Taha (reward function in `beat_rl/env/reward.py`); coordinate before any reward changes.

### SAC extension
`configs/sac.yaml` is scaffolding for a Soft Actor-Critic comparison study. Requires full SAC implementation (replay buffer, twin Q-networks, target networks, entropy temperature tuning), evaluation parity with PPO, and writeup. Multi-week effort. Only worth pursuing if tied to a paper, thesis, or follow-up course project.

## Notes

- The audit baseline (clean README, working install commands, hygiene fixes) was completed in May 2026.
- For collaboration questions, see the Contributors section in the main [README](../README.md).
