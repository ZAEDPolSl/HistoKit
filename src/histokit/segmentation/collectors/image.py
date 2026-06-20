from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np
from .base import OutputCollector, OutputKind, PipelineOutput


class ImageOutputCollector(OutputCollector):
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)

    def emit(self, output: PipelineOutput) -> None:
        if output.kind not in {OutputKind.IMAGE, OutputKind.MASK}:
            return

        basename = output.metadata.get("basename", "output")
        step = output.metadata.get("step", "images")

        save_dir = self.out_dir / step
        save_dir.mkdir(parents=True, exist_ok=True)

        arr = np.asarray(output.data)

        if arr.dtype == bool:
            arr = arr.astype(np.uint8) * 255

        Image.fromarray(arr).save(save_dir / f"{basename}.png")

class ThumbnailCollector(OutputCollector):
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)

    def emit(self, output: PipelineOutput) -> None:
        if output.name != "thumbnail":
            return

        basename = output.metadata.get("basename", "thumbnail")

        save_dir = self.out_dir / "thumbnails"
        save_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(np.asarray(output.data)).save(
            save_dir / f"{basename}.png"
        )


class SegmentationOverlayCollector(OutputCollector):
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)

    def emit(self, output: PipelineOutput) -> None:
        if output.name != "tissue_overlay":
            return

        basename = output.metadata.get("basename", "overlay")
        image = np.asarray(output.metadata["image"])
        mask = np.asarray(output.data)

        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        else:
            mask = mask.astype(np.uint8)

        save_dir = self.out_dir / "tissue_detection_overlay"
        save_dir.mkdir(parents=True, exist_ok=True)

        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 5)

        Image.fromarray(overlay).save(save_dir / f"{basename}.png")


class ArtifactOverlayCollector(OutputCollector):
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)

    def emit(self, output: PipelineOutput) -> None:
        if output.name != "artifact_overlay":
            return

        basename = output.metadata.get("basename", "artifact")
        image = np.asarray(output.metadata["image"])
        mask = np.asarray(output.data)
        colors = output.metadata["colors"]

        overlay_dir = self.out_dir / "artifact_detection_overlay"
        map_dir = self.out_dir / "artifact_detection_map"

        overlay_dir.mkdir(parents=True, exist_ok=True)
        map_dir.mkdir(parents=True, exist_ok=True)

        mask = cv2.resize(
            mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        color_map = np.zeros((*mask.shape, 3), dtype=np.uint8)

        for _, (artifact_id, color) in colors.items():
            color_map[mask == artifact_id] = color

        overlay = cv2.addWeighted(image, 0.6, color_map, 0.4, 0)

        Image.fromarray(color_map).save(map_dir / f"{basename}.png")
        Image.fromarray(overlay).save(overlay_dir / f"{basename}.png")


class HistogramCollector(OutputCollector):
    def __init__(
        self,
        out_dir,
        width: int = 600,
        height_per_channel: int = 150,
    ):
        self.out_dir = Path(out_dir)
        self.width = width
        self.height_per_channel = height_per_channel

    def emit(self, output: PipelineOutput) -> None:
        if output.name != "histograms":
            return

        data = output.data
        basename = output.metadata.get("basename", "histogram")

        save_dir = self.out_dir / "histograms"
        save_dir.mkdir(parents=True, exist_ok=True)

        total_height = 3 * self.height_per_channel

        img = Image.new(
            "RGB",
            (self.width, total_height),
            "white",
        )

        draw = ImageDraw.Draw(img)

        channels = [
            ("R", data["R"], (255, 0, 0)),
            ("G", data["G"], (0, 180, 0)),
            ("B", data["B"], (0, 0, 255)),
        ]

        bins = len(data["R"])

        left_margin = 45
        right_margin = 10
        top_margin = 10
        bottom_margin = 28

        axis_color = (180, 190, 200)
        grid_color = (230, 235, 240)
        text_color = (80, 80, 80)

        for idx, (name, hist, color) in enumerate(channels):
            y0 = idx * self.height_per_channel
            y1 = y0 + self.height_per_channel

            hist_log = np.log10(hist + 1)
            max_val = hist_log.max()

            if max_val == 0:
                continue

            px0 = left_margin
            py0 = y0 + top_margin
            px1 = self.width - right_margin
            py1 = y1 - bottom_margin

            plot_width = px1 - px0
            plot_height = py1 - py0

            for i in range(6):
                frac = i / 5
                y = int(py1 - frac * plot_height)

                draw.line(
                    [(px0, y), (px1, y)],
                    fill=grid_color,
                    width=1,
                )

                draw.text(
                    (5, y - 7),
                    f"{frac * max_val:.1f}",
                    fill=text_color,
                )

            for tick in [0, 64, 128, 192, 255]:
                x = int(px0 + (tick / (bins - 1)) * plot_width)

                draw.line(
                    [(x, py1), (x, py1 + 4)],
                    fill=axis_color,
                    width=1,
                )

                draw.text(
                    (x - 10, py1 + 7),
                    str(tick),
                    fill=text_color,
                )

            draw.line([(px0, py0), (px0, py1)], fill=axis_color, width=1)
            draw.line([(px0, py1), (px1, py1)], fill=axis_color, width=1)

            x_scale = plot_width / bins

            for i, value in enumerate(hist_log):
                bar_height = int((value / max_val) * plot_height)
                x = int(px0 + i * x_scale)

                draw.line(
                    [(x, py1), (x, py1 - bar_height)],
                    fill=color,
                    width=max(1, int(x_scale)),
                )

            thr_x = int(px0 + data["thr"][name] * x_scale)

            draw.line(
                [(thr_x, py0), (thr_x, py1)],
                fill=(255, 165, 0),
                width=2,
            )

            draw.text(
                (px0 + 5, py0),
                name,
                fill=(0, 0, 0),
            )

            draw.text(
                (thr_x - 75, py0),
                f"thr={round(data['thr'][name])}",
                fill=(255, 165, 0),
            )

        img.save(save_dir / f"{basename}.png")