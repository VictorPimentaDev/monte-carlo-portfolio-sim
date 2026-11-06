import json
import shutil
import subprocess

import numpy as np
import pytest
from matplotlib.animation import writers
from PIL import Image

from portfolio_sim.cli import main
from portfolio_sim.model import Config, simulate
from portfolio_sim.render import Chart, export_animation, export_png


@pytest.mark.parametrize("layout, size", [("wide", (1280, 720)), ("vertical", (720, 1280))])
def test_png_renders_without_display(tmp_path, layout, size):
    path = export_png(simulate(Config(paths=12)), tmp_path / "plot.png", layout=layout)
    with Image.open(path) as picture:
        assert picture.size == size
        assert picture.getpixel((0, 0))[:3] == (7, 7, 11)


def test_distribution_counts_every_visible_path_and_shares_y_axis():
    result = simulate(Config(paths=30))
    chart = Chart(result)
    try:
        for progress in [0, 0.05, 0.5, 1]:
            chart.update(progress)
            count = max(1, int(np.ceil(30 * min(1, progress * 4))))
            assert sum(bar.get_width() for bar in chart.bars) == count
            assert chart.hist.get_ylim() == chart.ax.get_ylim()
            assert chart.ax.get_yscale() == chart.hist.get_yscale() == "log"
            assert chart.ax.get_ylim()[0] < result.values.min()
            assert chart.ax.get_ylim()[1] > result.values.max()
            np.testing.assert_allclose(np.diff(np.log(chart.bins)), np.diff(np.log(chart.bins))[0])
        assert chart.stat_texts["loss_probability"].get_text().endswith("%")
        assert chart.counter.get_text() == "PATHS 030 / 30"
    finally:
        chart.close()


@pytest.mark.ffmpeg
@pytest.mark.skipif(
    not writers.is_available("ffmpeg"), reason="ffmpeg unavailable: animation smoke"
)
def test_real_mp4_and_gif_encode(tmp_path):
    simulation = simulate(Config(paths=6, years=1))
    mp4 = export_animation(simulation, tmp_path / "smoke.mp4", seconds=0.5, fps=4)
    gif = export_animation(simulation, tmp_path / "smoke.gif", seconds=0.5, fps=4)
    assert mp4.stat().st_size > 1000
    with Image.open(gif) as frames:
        assert frames.size == (1280, 720)
        assert frames.n_frames >= 2
    if shutil.which("ffprobe"):
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(mp4),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(probe.stdout)["streams"][0] == {"width": 1280, "height": 720}


def test_missing_ffmpeg_has_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setattr(writers, "is_available", lambda name: False)
    with pytest.raises(ValueError, match="install it or choose --formats png"):
        export_animation(simulate(Config(paths=2)), tmp_path / "test.mp4")


def test_cli_outputs_reproducible_report_and_offline_provenance(tmp_path):
    main(["--ticker", "WEGE3", "--offline", "--paths", "8", "--output", str(tmp_path)])
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["provenance"]["kind"] == "synthetic_fixture"
    assert report["config"]["paths"] == 8
    assert report["analytical"]["median"] > 0
    assert (tmp_path / "simulation-wide.png").exists()


def test_cli_rejects_bad_input_without_export(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--vol", "-1", "--output", str(tmp_path)])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []
