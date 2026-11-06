import json
import shutil
import subprocess

import numpy as np
import pytest
from matplotlib.animation import writers
from matplotlib.text import Text
from PIL import Image

from portfolio_sim.cli import main
from portfolio_sim.model import Config, simulate
from portfolio_sim.render import SPECTRUM, Chart, export_animation, export_png, money


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


@pytest.mark.parametrize("layout", ["wide", "vertical"])
def test_reference_elements_match_the_statistics(layout):
    result = simulate()
    chart = Chart(result, layout)
    try:
        texts = [text.get_text() for text in chart.fig.texts]
        assert "MONTE CARLO SIMULATION" in texts
        assert "PORTFOLIO VALUE · $100K START · 10 YEARS · RETURN 8% · VOL 17%" in texts
        assert "DISTRIBUTION" in texts
        assert chart.counter.get_text() == "PATHS 320 / 320"
        assert chart.year.get_text() == "YEAR 10.0"
        for key, value in result.statistics.items():
            if key in chart.lines:
                assert chart.lines[key].get_ydata()[-1] == pytest.approx(value)
                assert chart.labels[key].get_position()[1] == pytest.approx(value)
                assert chart.stat_texts[key].get_text() == money(value)
        # The gradient spends its whole range: extreme outcomes reach coral and purple.
        np.testing.assert_allclose(chart.colors[np.argmin(result.values[:, -1])], SPECTRUM(0.0))
        np.testing.assert_allclose(chart.colors[np.argmax(result.values[:, -1])], SPECTRUM(1.0))
    finally:
        chart.close()


@pytest.mark.parametrize("layout", ["wide", "vertical"])
@pytest.mark.parametrize(
    "config",
    [
        Config(),
        Config(start=5_000_000, years=20, annual_return=0.12, volatility=0.4),
        Config(volatility=0),
        Config(years=1, annual_return=0.01, volatility=0.02),
        Config(paths=1),
    ],
)
def test_labels_stay_legible_in_every_frame(config, layout):
    chart = Chart(simulate(config), layout, "SYNTHETIC OFFLINE DATA")
    try:
        ticks = chart.ax.yaxis.get_major_locator()()
        low, high = chart.ax.get_ylim()
        assert np.count_nonzero((ticks >= low) & (ticks <= high)) >= 2
        for progress in (0.02, 0.1, 0.5, 1):
            chart.update(progress)
            chart.fig.canvas.draw()
            renderer = chart.fig.canvas.get_renderer()
            figure = chart.fig.bbox
            for text in chart.fig.findobj(Text):
                if text.get_text() and text.get_visible():
                    box = text.get_window_extent(renderer)
                    assert box.x0 >= 0 and box.x1 <= figure.x1, text.get_text()
                    assert box.y0 >= 0 and box.y1 <= figure.y1, text.get_text()
            for group in (chart.stat_texts, chart.labels):
                boxes = [t.get_window_extent(renderer) for t in group.values() if t.get_visible()]
                for first, second in zip(boxes, boxes[1:]):
                    assert not first.overlaps(second), [t.get_text() for t in group.values()]
    finally:
        chart.close()


def test_money_keeps_short_labels_faithful():
    assert [money(value) for value in (837, 2_500, 99_600, 999_600, 5e6, 235.65e6, 1.2e9)] == [
        "$837",
        "$2.5K",
        "$100K",
        "$1M",
        "$5M",
        "$236M",
        "$1.2B",
    ]


def test_rendering_is_pixel_deterministic():
    frames = []
    for _ in range(2):
        chart = Chart(simulate(Config(paths=20)))
        try:
            chart.fig.canvas.draw()
            frames.append(np.asarray(chart.fig.canvas.buffer_rgba()).copy())
        finally:
            chart.close()
    np.testing.assert_array_equal(*frames)


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
                "stream=width,height,color_space",
                "-of",
                "json",
                str(mp4),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        assert (stream["width"], stream["height"]) == (1920, 1080)
        assert stream["color_space"] == "bt709"


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
