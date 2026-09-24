"""Parse every inline/external application JS file with Node (no npm dependencies)."""

from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run(["node", "--check", str(ROOT / "frontend/app.js")], check=True)
    with tempfile.TemporaryDirectory() as directory:
        for page in (ROOT / "frontend").glob("*.html"):
            for index, script in enumerate(
                re.findall(r"<script>(.*?)</script>", page.read_text(), re.S)
            ):
                path = Path(directory) / f"{page.stem}-{index}.js"
                path.write_text(script)
                subprocess.run(["node", "--check", str(path)], check=True)
    print("All frontend JavaScript parses")


if __name__ == "__main__":
    main()
