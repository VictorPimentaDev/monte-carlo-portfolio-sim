"""Original social image, reproducible from the same simulation as the README."""

from pathlib import Path

from PIL import Image

from portfolio_sim.model import simulate
from portfolio_sim.render import BACKGROUND, MUTED, Chart

path = Path(__file__).resolve().parents[1] / "assets/social-preview.png"
path.parent.mkdir(exist_ok=True)
chart = Chart(simulate())
try:
    chart.fig.set_size_inches(12.8, 6.4)
    chart.fig.text(
        0.955,
        0.862,
        "PYTHON / NUMPY / MATPLOTLIB",
        ha="right",
        fontfamily="DejaVu Sans Mono",
        fontsize=7,
        color=MUTED,
    )
    chart.fig.savefig(path, dpi=100, facecolor=BACKGROUND)
finally:
    chart.close()
with Image.open(path) as picture:
    assert picture.size == (1280, 640)
assert path.stat().st_size < 1_000_000
print(path)
