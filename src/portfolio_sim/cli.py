import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .market import load_prices
from .model import Config, analytical_statistics, simulate
from .render import export_animation, export_png


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Seeded GBM portfolio simulations and black charts."
    )
    parser.add_argument("--start", type=float, default=100_000)
    parser.add_argument("--years", type=float, default=10)
    parser.add_argument(
        "--return",
        dest="annual_return",
        type=float,
        default=0.08,
        help="annual arithmetic drift as decimal, e.g. 0.08",
    )
    parser.add_argument("--vol", type=float, default=0.17, help="annual volatility, e.g. 0.17")
    parser.add_argument("--paths", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    market = parser.add_mutually_exclusive_group()
    market.add_argument(
        "--ticker", help="estimate drift/vol from five years of adjusted daily prices"
    )
    market.add_argument("--ibov", action="store_true", help="calibrate from ^BVSP")
    parser.add_argument(
        "--offline", action="store_true", help="use clearly labeled SYNTHETIC fixture"
    )
    parser.add_argument("--layout", choices=["wide", "vertical", "both"], default="wide")
    parser.add_argument("--formats", nargs="+", choices=["png", "mp4", "gif"], default=["png"])
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args(argv)
    try:
        provenance = {"kind": "illustrative_assumptions", "source": "user-supplied GBM parameters"}
        source = "ILLUSTRATIVE ASSUMPTIONS"
        annual_return, volatility = args.annual_return, args.vol
        if args.ticker or args.ibov or args.offline:
            calibration = load_prices("^BVSP" if args.ibov else args.ticker, offline=args.offline)
            annual_return, volatility = calibration.annual_return, calibration.volatility
            provenance = calibration.provenance
            source = "SYNTHETIC OFFLINE DATA" if args.offline else f"YAHOO · {provenance['ticker']}"
        config = Config(args.start, args.years, annual_return, volatility, args.paths, args.seed)
        result = simulate(config)
        analytics = analytical_statistics(config)
        args.output.mkdir(parents=True, exist_ok=True)
        layouts = ["wide", "vertical"] if args.layout == "both" else [args.layout]
        for layout in layouts:
            for extension in dict.fromkeys(args.formats):
                path = args.output / f"simulation-{layout}.{extension}"
                if extension == "png":
                    export_png(result, path, layout=layout, source=source)
                else:
                    export_animation(
                        result,
                        path,
                        layout=layout,
                        seconds=args.seconds,
                        fps=args.fps,
                        source=source,
                    )
                print(f"Saved {path}")
        report = {
            "config": asdict(config),
            "empirical": result.statistics,
            "analytical": analytics,
            "provenance": provenance,
            "model": "exact GBM; monthly grid; nominal values; no contributions, taxes or fees",
        }
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
