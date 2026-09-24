"""Print Colab setup guidance; this is not a training notebook or evidence of a run."""

from pathlib import Path

if __name__ == "__main__":
    print((Path(__file__).resolve().parents[1] / "docs" / "TRAINING.md").read_text())
