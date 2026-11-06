"""Verify the delivered media, independently of the render implementation."""

import json
import subprocess
from pathlib import Path

from PIL import Image

root = Path(__file__).resolve().parents[1] / "assets"
for layout, size in (("wide", (1280, 720)), ("vertical", (720, 1280))):
    for extension in ("png", "gif"):
        path = root / f"simulation-{layout}.{extension}"
        with Image.open(path) as picture:
            assert picture.size == size, path
            if extension == "gif":
                duration = 0
                for frame in range(picture.n_frames):
                    picture.seek(frame)
                    duration += picture.info.get("duration", 0)
                assert abs(duration / 1000 - 8) < 0.1, (path, duration)
                assert picture.n_frames >= 100
        assert path.stat().st_size < 10_000_000, path
    video = root / f"simulation-{layout}.mp4"
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,duration,r_frame_rate,nb_frames,color_space,pix_fmt",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert (stream["width"] * 2, stream["height"] * 2) == (size[0] * 3, size[1] * 3), stream
    assert (stream["color_space"], stream["pix_fmt"]) == ("bt709", "yuv420p"), stream
    assert float(stream["duration"]) == 8
    assert stream["r_frame_rate"] == "15/1"
    assert int(stream["nb_frames"]) == 120
    print(
        f"{layout}: PNG/GIF {size[0]}×{size[1]}, MP4 {stream['width']}×{stream['height']} BT.709, "
        "8 seconds, 120 MP4 frames; sizes OK"
    )
with Image.open(root / "social-preview.png") as picture:
    assert picture.size == (1280, 640)
assert (root / "social-preview.png").stat().st_size < 1_000_000
print("social-preview: 1280×640, below 1 MB")
