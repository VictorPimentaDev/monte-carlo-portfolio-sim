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
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

from .model import Simulation, summarize

BACKGROUND = "#07070b"
WHITE = "#f1f1f4"
MUTED = "#858591"
SPECTRUM = LinearSegmentedColormap.from_list(
    "outcomes",
    [
        "#e0607e",
        "#ec8674",
        "#f2ab60",
        "#ecce62",
        "#a6d17c",
        "#4fc6ac",
        "#3fa9d6",
        "#5a80da",
        "#8a5fd4",
    ],
)
SIZES = {"wide": (1280, 720), "vertical": (720, 1280)}
MP4_DPI = 150  # 1920×1080 and 1080×1920; GIF and PNG stay at the 100 dpi base size.
BINS = 30


def money(value):
    if value >= 999_500_000:
        return f"${value / 1e9:,.3g}B"
    if value >= 999_500:
        return f"${value / 1e6:.3g}M"
    if value >= 9_950:
        return f"${value / 1e3:.0f}K"
    if value >= 1_000:
        return f"${value / 1e3:.1f}K"
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
            0.885 if self.vertical else 0.91,
            "MONTE CARLO SIMULATION",
            fontsize=16 if self.vertical else 20,
            weight="bold",
            **font,
        )
        for index, color in enumerate(SPECTRUM(np.linspace(0, 1, 7))):
            self.fig.add_artist(
                Rectangle(
                    (0.055 + index * 0.0052, 0.886 if self.vertical else 0.911),
                    0.0035,
                    0.014 if self.vertical else 0.024,
                    transform=self.fig.transFigure,
                    color=color,
                    linewidth=0,
                )
            )
        self.fig.text(
            0.055,
            0.858 if self.vertical else 0.862,
            f"PORTFOLIO VALUE · {money(config.start)} START · {config.years:g} YEAR"
            f"{'' if config.years == 1 else 'S'}"
            f" · RETURN {config.annual_return * 100:.3g}% · VOL {config.volatility * 100:.3g}%",
            fontsize=8 if self.vertical else 9,
            **{**font, "color": MUTED},
        )
        self.counter = self.fig.text(
            0.955,
            0.889 if self.vertical else 0.917,
            "",
            ha="right",
            fontsize=8 if self.vertical else 9,
            **font,
        )
        # Vertical keeps the top 11% and bottom 15% clear of reel UI overlays.
        bottom, top = (0.29, 0.815) if self.vertical else (0.225, 0.785)
        self.ax = self.fig.add_axes(
            [
                0.11 if self.vertical else 0.085,
                bottom,
                0.55 if self.vertical else 0.585,
                top - bottom,
            ]
        )
        hist_left = 0.74 if self.vertical else 0.73
        self.hist = self.fig.add_axes(
            [hist_left, bottom, 0.955 - hist_left, top - bottom], sharey=self.ax
        )
        self.fig.text(
            hist_left,
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
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#22222b", "edgecolor": "none"},
            **font,
        )
        minimum = float(simulation.values.min()) / 1.15
        maximum = float(simulation.values.max()) * 1.10
        self.ax.set_yscale("log", base=2)
        self.ax.set_ylim(minimum, maximum)
        self.ax.set_xlim(0, config.years)
        self.ax.set_xlabel(
            "Y E A R S", color=MUTED, fontfamily="DejaVu Sans Mono", fontsize=8, labelpad=15
        )
        self.ax.xaxis.set_major_locator(MaxNLocator(5))
        magnitude = 10 ** np.floor(np.log10(config.start))
        low, high = np.log2([minimum / magnitude, maximum / magnitude])
        step = max(1, int(np.ceil((high - low) / 8)))
        while (high - low) / step < 2:  # A narrow fan still needs a readable scale.
            step /= 2
        exponents = np.arange(np.ceil(low / step), np.floor(high / step) + 1) * step
        self.ax.yaxis.set_major_locator(FixedLocator(magnitude * 2.0**exponents))
        self.ax.yaxis.set_minor_locator(NullLocator())
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: money(value)))
        self.fig.text(
            0.11 if self.vertical else 0.085,
            top + 0.014,
            "LOG SCALE",
            fontsize=6 if self.vertical else 7,
            **{**font, "color": MUTED},
        )
        for axis in (self.ax, self.hist):
            axis.set_facecolor(BACKGROUND)
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.tick_params(
                colors=MUTED, labelsize=7, length=0, pad=9, labelfontfamily="DejaVu Sans Mono"
            )
        self.ax.spines["left"].set_visible(True)
        self.ax.spines["bottom"].set_visible(True)
        self.ax.spines["left"].set_color("#373740")
        self.ax.spines["bottom"].set_color("#373740")
        self.hist.tick_params(axis="y", labelleft=False)
        self.hist.set_xticks([])
        self.ax.grid(axis="y", color="#17171e", linewidth=0.6, alpha=0.7)
        # Log-space, so the lognormal tails reach both ends with the median at the center.
        final = np.log(simulation.values[:, -1])
        center = float(np.median(final))
        normalizer = TwoSlopeNorm(
            vmin=min(float(final.min()), center - 0.05),
            vcenter=center,
            vmax=max(float(final.max()), center + 0.05),
        )
        self.color_of = lambda value: SPECTRUM(normalizer(np.log(value)))
        self.colors = self.color_of(simulation.values[:, -1])
        self.paths = LineCollection([], colors=self.colors, linewidths=0.7, alpha=0.5, zorder=2)
        self.ax.add_collection(self.paths)
        self.bins = np.geomspace(minimum, maximum, BINS + 1)
        centers = np.sqrt(self.bins[1:] * self.bins[:-1])
        self.bars = self.hist.barh(
            centers,
            np.zeros(BINS),
            height=np.diff(self.bins) * 0.84,
            color=self.color_of(centers),
            alpha=0.95,
        )
        self.labels = {}
        self.guides = {}
        self.lines = {}
        for key, name, color in (
            ("p95", "P95", "#59c9bd"),
            ("median", "MEDIAN", WHITE),
            ("p5", "P5", "#ec9b7c"),
        ):
            (self.lines[key],) = self.ax.plot(
                [],
                [],
                color=color,
                linewidth=1.8 if key == "median" else 1.2,
                marker="o",
                markersize=4,
                markevery=[-1],
                clip_on=False,
                zorder=4 if key == "median" else 3,
            )
            self.labels[key] = self.ax.text(
                1.03,
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
            x = 0.14 + index * 0.18
            caption = self.fig.text(
                x,
                0.215 if self.vertical else 0.109,
                label,
                ha="center",
                fontsize=6.5 if self.vertical else 8,
                **{**font, "color": MUTED},
            )
            self.ax.annotate(
                "●",
                xy=(0, 0.5),
                xycoords=caption,
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                color=color,
                fontsize=4 if self.vertical else 5,
                annotation_clip=False,
            )
            self.stat_texts[key] = self.fig.text(
                x,
                0.18 if self.vertical else 0.058,
                "",
                ha="center",
                fontsize=20 if self.vertical else 24,
                weight="bold",
                **{**font, "color": color, "fontfamily": "DejaVu Sans"},
            )
        self.fig.text(
            0.055,
            0.155 if self.vertical else 0.017,
            f"GBM · MONTHLY · SEED {config.seed} · {source.upper()}",
            fontsize=6 if self.vertical else 6.5,
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
        percentiles = np.percentile(values, [95, 50, 5], axis=0)
        for line, series in zip(self.lines.values(), percentiles, strict=True):
            line.set_data(times, series)
        stats = summarize(values[:, -1], simulation.config.start)
        counts, _ = np.histogram(values[:, -1], bins=self.bins)
        self.hist.set_xlim(0, max(1, counts.max()) * 1.05)
        for bar, count_in_bin in zip(self.bars, counts, strict=True):
            bar.set_width(count_in_bin)
        # Pills separate when the fan is still narrow; lines and guides keep the exact values.
        low, high = self.ax.get_ylim()
        gap = (high / low) ** (0.035 if self.vertical else 0.06)
        pills = {
            "p95": max(stats["p95"], stats["median"] * gap),
            "median": stats["median"],
            "p5": min(stats["p5"], stats["median"] / gap),
        }
        for key in self.labels:
            self.labels[key].set_y(pills[key])
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
    is_gif = path.suffix.lower() == ".gif"
    frames = max(2, round(seconds * fps))
    progress = np.minimum(1, np.linspace(0, 1 / 0.85, frames))
    animation = FuncAnimation(
        chart.fig, chart.update, frames=progress, interval=1000 / fps, blit=False, repeat=False
    )
    # GIF: lossless intermediate, so the palette sees the real colors. MP4: tagged BT.709,
    # otherwise HD players decode untagged BT.601 pixels with shifted hues.
    writer = (
        FFMpegWriter(fps=fps, codec="png")
        if is_gif
        else FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=[
                "-vf",
                "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-crf",
                "20",
                "-preset",
                "slow",
                "-movflags",
                "+faststart",
            ],
        )
    )
    try:
        with TemporaryDirectory(prefix="portfolio-sim-") as temporary:
            video = Path(temporary) / "animation.mkv" if is_gif else path
            animation.save(video, writer=writer, dpi=100 if is_gif else MP4_DPI)
            if is_gif:
                subprocess.run(
                    [
                        writer.bin_path(),
                        "-y",
                        "-v",
                        "error",
                        "-i",
                        str(video),
                        "-filter_complex",
                        "[0:v]split[a][b];[a]palettegen=max_colors=256[p];"
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
