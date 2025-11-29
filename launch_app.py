import sys
from pathlib import Path

root_dir = Path(__file__).parent
sys.path.append(str(root_dir / "src"))

from TokEye.app.__main__ import create_app

cwd = Path.cwd()
for directory in ["cache", "outputs", "annotations", "model", "data"]:
    (cwd / directory).mkdir(exist_ok=True)
demo = create_app()

if __name__ == "__main__":
    demo.launch()
