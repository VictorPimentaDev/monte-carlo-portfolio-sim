"""A black canvas, paths colored by terminal value, and a shared distribution axis."""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, writers
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .model import Simulation, summarize

BACKGROUND = "#07070b"
WHITE = "#f1f1f4"
MUTED = "#858591"
SPECTRUM = LinearSegmentedColormap.from_list(
    "outcomes", ["#e66b85", "#efa16a", "#e8d86b", "#8ecb8d", "#40c5bc", "#459cda", "#9260cf"]
)
SIZES = {"wide": (1280, 720), "vertical": (720, 1280)}


def money(value):
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:,.0f}K"
    return f"${value:,.0f}"


class Chart:
    def __init__(self, simulation: Simulation, layout="wide", source="ILLUSTRATIVE ASSUMPTIONS"):
        if layout not in SIZES:
            raise ValueError("layout must be wide or vertical")
        self.simulation = simulation
        self.vertical = layout == "vertical"
        width, height = SIZES[layout]
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=BACKGROUND)
        config = simulation.config
        font = {"fontfamily": "DejaVu Sans Mono", "color": WHITE}
        self.font = font
        self.fig.text(
            0.105,
            0.925 if self.vertical else 0.91,
            "MONTE CARLO SIMULATION",
            fontsize=16 if self.vertical else 20,
            weight="bold",
            **font,
        )
        for index, color in enumerate(SPECTRUM(np.linspace(0, 1, 7))):
            self.fig.add_artist(
                Rectangle(
                    (0.055 + index * 0.0052, 0.926 if self.vertical else 0.911),
                    0.0035,
                    0.014 if self.vertical else 0.024,
                    transform=self.fig.transFigure,
                    color=color,
                    linewidth=0,
                )
            )
        subtitle = f"PORTFOLIO VALUE · {money(config.start)} START · {config.years:g} YEARS"
        parameters = f"RETURN {config.annual_return:.1%} · VOL {config.volatility:.1%}"
        self.fig.text(
            0.055,
            0.884 if self.vertical else 0.862,
            subtitle + ("\n" if self.vertical else " · ") + parameters,
            fontsize=8 if self.vertical else 9,
            linespacing=2,
            **{**font, "color": MUTED},
        )
        self.counter = self.fig.text(
            0.955,
            0.816 if self.vertical else 0.917,
            "",
            ha="right",
            fontsize=8 if self.vertical else 9,
            **font,
        )
        bottom, top = (0.225, 0.775) if self.vertical else (0.225, 0.785)
        self.ax = self.fig.add_axes(
            [
                0.11 if self.vertical else 0.085,
                bottom,
                0.55 if self.vertical else 0.585,
                top - bottom,
            ]
        )
        self.hist = self.fig.add_axes(
            [0.755, bottom, 0.20 if self.vertical else 0.19, top - bottom], sharey=self.ax
        )
        self.fig.text(
            0.755,
            top + 0.014,
            "DISTRIBUTION",
            fontsize=7 if self.vertical else 8,
            weight="bold",
            **{**font, "color": MUTED},
        )
        self.year = self.fig.text(
            0.66 if self.vertical else 0.67,
            top + 0.014,
            "",
            ha="right",
            fontsize=7 if self.vertical else 8,
            **font,
        )
        maximum = float(simulation.values.max()) * 1.055
        self.ax.set_ylim(0, maximum)
        self.ax.set_xlim(0, config.years)
        self.ax.set_xlabel(
            "Y E A R S", color=MUTED, fontfamily="DejaVu Sans Mono", fontsize=8, labelpad=15
        )
        self.ax.xaxis.set_major_locator(MaxNLocator(5))
        self.ax.yaxis.set_major_locator(MaxNLocator(6))
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: money(value)))
        for axis in (self.ax, self.hist):
            axis.set_facecolor(BACKGROUND)
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.tick_params(colors=MUTED, labelsize=7, length=0, pad=9)
        self.ax.spines["left"].set_visible(True)
        self.ax.spines["bottom"].set_visible(True)
        self.ax.spines["left"].set_color("#373740")
        self.ax.spines["bottom"].set_color("#373740")
        self.hist.tick_params(axis="y", labelleft=False)
        self.hist.set_xticks([])
        self.ax.grid(axis="y", color="#17171e", linewidth=0.6, alpha=0.7)
        final_median = float(np.median(simulation.values[:, -1]))
        self.normalizer = TwoSlopeNorm(
            vmin=0,
            vcenter=final_median,
            vmax=max(float(simulation.values[:, -1].max()), final_median * 1.01),
        )
        self.colors = SPECTRUM(self.normalizer(simulation.values[:, -1]))
        self.paths = LineCollection([], colors=self.colors, linewidths=0.6, alpha=0.40, zorder=2)
        self.ax.add_collection(self.paths)
        (self.median,) = self.ax.plot([], [], color=WHITE, linewidth=1.7, zorder=4)
        self.bins = np.linspace(0, maximum, 37)
        centers = (self.bins[1:] + self.bins[:-1]) / 2
        self.bars = self.hist.barh(
            centers,
            np.zeros(36),
            height=np.diff(self.bins) * 0.84,
            color=SPECTRUM(self.normalizer(centers)),
            alpha=0.95,
        )
        self.labels = {}
        self.guides = {}
        for key, name, color in (
            ("p95", "P95", "#59c9bd"),
            ("median", "MEDIAN", WHITE),
            ("p5", "P5", "#ec9b7c"),
        ):
            self.labels[key] = self.ax.text(
                1.024,
                0,
                name,
                transform=self.ax.get_yaxis_transform(),
                clip_on=False,
                color=color,
                fontsize=6 if self.vertical else 7,
                va="center",
                fontfamily="DejaVu Sans Mono",
                weight="bold",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": BACKGROUND,
                    "edgecolor": color,
                    "linewidth": 0.6,
                    "alpha": 0.95,
                },
            )
            self.guides[key] = self.hist.axhline(
                0, color=color, alpha=0.28, linewidth=0.8, linestyle=(0, (1, 3))
            )
        self.stat_texts = {}
        for index, (key, label, color) in enumerate(
            [
                ("p95", "95TH PCT", "#59c9bd"),
                ("median", "MEDIAN", WHITE),
                ("p5", "5TH PCT", "#ec9b7c"),
                ("mean", "MEAN", "#46b9cb"),
                ("loss_probability", "LOSS PROB.", "#e66b85"),
            ]
        ):
            x = 0.055 + index * 0.185
            self.fig.text(
                x,
                0.142 if self.vertical else 0.109,
                "• " + label,
                fontsize=6 if self.vertical else 8,
                **{**font, "color": MUTED},
            )
            self.stat_texts[key] = self.fig.text(
                x,
                0.113 if self.vertical else 0.064,
                "",
                fontsize=14 if self.vertical else 23,
                weight="bold",
                **{**font, "color": color},
            )
        self.fig.text(
            0.055,
            0.057 if self.vertical else 0.017,
            f"GBM · MONTHLY · SEED {config.seed} · {source.upper()}",
            fontsize=5.5 if self.vertical else 6.5,
            **{**font, "color": MUTED},
        )
        self.update(1)

    def update(self, progress):
        simulation = self.simulation
        # Reveal paths early, then spend the remaining frames growing the time horizon.
        count = max(1, int(np.ceil(simulation.config.paths * min(1, progress * 4))))
        step = min(len(simulation.times) - 1, int(round(progress * (len(simulation.times) - 1))))
        values = simulation.values[:count, : step + 1]
        times = simulation.times[: step + 1]
        segments = np.stack((np.broadcast_to(times, values.shape), values), axis=-1)
        self.paths.set_segments(segments)
        self.paths.set_color(self.colors[:count])
        self.median.set_data(times, np.median(values, axis=0))
        stats = summarize(values[:, -1], simulation.config.start)
        counts, _ = np.histogram(values[:, -1], bins=self.bins)
        self.hist.set_xlim(0, max(1, counts.max()) * 1.05)
        for bar, count_in_bin in zip(self.bars, counts, strict=True):
            bar.set_width(count_in_bin)
        for key in self.labels:
            self.labels[key].set_y(stats[key])
            self.labels[key].set_visible(step > 0)
            self.guides[key].set_ydata([stats[key], stats[key]])
        for key, label in self.stat_texts.items():
            label.set_text(f"{stats[key]:.1%}" if key == "loss_probability" else money(stats[key]))
        self.counter.set_text(f"PATHS {count:03d} / {simulation.config.paths}")
        self.year.set_text(f"YEAR {times[-1]:04.1f}")
        return []

    def close(self):
        plt.close(self.fig)


def export_png(simulation, path, *, layout="wide", source="ILLUSTRATIVE ASSUMPTIONS"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chart = Chart(simulation, layout, source)
    try:
        chart.fig.savefig(path, dpi=100, facecolor=BACKGROUND)
    finally:
        chart.close()
    return path


def export_animation(
    simulation, path, *, layout="wide", seconds=8, fps=15, source="ILLUSTRATIVE ASSUMPTIONS"
):
    """Encode MP4 with Matplotlib/ffmpeg; GIF uses ffmpeg's palette for bounded memory."""
    import subprocess

    path = Path(path)
    if path.suffix.lower() not in (".mp4", ".gif"):
        raise ValueError("animation extension must be .mp4 or .gif")
    if not np.isfinite(seconds) or seconds < 0.5 or seconds > 120:
        raise ValueError("seconds must be between 0.5 and 120")
    if not isinstance(fps, int) or fps < 1 or fps > 60:
        raise ValueError("fps must be an integer between 1 and 60")
    if not writers.is_available("ffmpeg"):
        raise ValueError("ffmpeg is required for MP4/GIF; install it or choose --formats png")
    path.parent.mkdir(parents=True, exist_ok=True)
    chart = Chart(simulation, layout, source)
    frames = max(2, round(seconds * fps))
    progress = np.minimum(1, np.linspace(0, 1 / 0.85, frames))
    animation = FuncAnimation(
        chart.fig, chart.update, frames=progress, interval=1000 / fps, blit=False, repeat=False
    )
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2400,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        with TemporaryDirectory(prefix="portfolio-sim-") as temporary:
            video = path if path.suffix.lower() == ".mp4" else Path(temporary) / "animation.mp4"
            animation.save(video, writer=writer, dpi=100)
            if path.suffix.lower() == ".gif":
                subprocess.run(
                    [
                        writer.bin_path(),
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(video),
                        "-filter_complex",
                        "[0:v]split[a][b];[a]palettegen=max_colors=128[p];"
                        "[b][p]paletteuse=dither=bayer:bayer_scale=3",
                        "-loop",
                        "0",
                        str(path),
                    ],
                    check=True,
                )
    finally:
        chart.close()
    return path
