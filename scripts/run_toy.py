"""Run the CPU toy end to end (python scripts/run_toy.py [--config configs/toy.yaml] [--save student.pt]); same as
python -m drift_vsr.demo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drift_vsr.demo import main  # noqa: E402

if __name__ == "__main__":
    main()
