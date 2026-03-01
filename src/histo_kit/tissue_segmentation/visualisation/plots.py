from pathlib import Path
import numpy as np
from PIL import ImageDraw, ImageFont
from PIL import Image
from .base import VisualizationStep
from ...slide.slide import Slide


class ThresholdingHistograms(VisualizationStep):

    def __call__(self, data: dict,
                 slide: Slide = None,
                 out_dir: Path = None,
                 sub_dir_name ="mask_overlay",
                 vis_mag: float = 0.625):

        save_dir = out_dir / sub_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)
        slide = data["slide"]
        img = self.plot_rgb_hist(data["R"], data["G"], data["B"], data["thr"])
        img.save(save_dir / f"{slide.name}.png")

    @staticmethod
    def plot_rgb_hist(R, G, B, thr, width=800, height_per_channel=250, margin=100,
                      font_size_title=16, font_size_axis=14):
        """
        Plot RGB channel histograms as horizontal bar charts using PIL.

        Each channel is rendered as a separate subplot stacked vertically.
        Bars grow upward from a baseline, with pixel intensity bins along the
        x-axis and log-scaled counts on the y-axis. A vertical threshold line
        is overlaid on each channel's plot.

        Parameters
        ----------
        R : ndarray of shape (256,)
            Histogram of pixel counts for the Red channel.
        G : ndarray of shape (256,)
            Histogram of pixel counts for the Green channel.
        B : ndarray of shape (256,)
            Histogram of pixel counts for the Blue channel.
        thr : dict
            Dictionary with threshold values for each channel:
            - ``"R"`` : threshold for Red channel
            - ``"G"`` : threshold for Green channel
            - ``"B"`` : threshold for Blue channel
        width : int, optional
            Total image width in pixels. Defaults to ``800``.
        height_per_channel : int, optional
            Height allocated for each channel subplot. Defaults to ``250``.
        margin : int, optional
            Margin size in pixels around the plot area. Defaults to ``100``.
        font_size_title : int, optional
            Font size for channel title text. Defaults to ``24``.
        font_size_axis : int, optional
            Font size for axis label text. Defaults to ``18``.

        Returns
        -------
        img : PIL.Image.Image
            Rendered RGB histogram image.
        """
        total_height = 3 * height_per_channel + margin
        img = Image.new("RGB", (width, total_height), "white")
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size_title)
            font_axis = ImageFont.truetype("DejaVuSans.ttf", font_size_axis)
        except IOError:
            font_title = ImageFont.load_default()
            font_axis = ImageFont.load_default()

        channels = [
            ("R", R, (247, 93, 69)),
            ("G", G, (124, 209, 107)),
            ("B", B, (54, 143, 245)),
        ]

        bins_per_group = 10
        plot_width = width - 2 * margin
        plot_height = height_per_channel - 80

        x_tick_values = [0, 64, 128, 192, 255]
        y_tick_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for i, (name, hist, color) in enumerate(channels):
            y_offset = i * height_per_channel + margin // 2
            baseline = y_offset + plot_height


            hist = np.array(hist, dtype=float)
            grouped_hist = hist[: bins_per_group * (256 // bins_per_group)].reshape(-1, bins_per_group).sum(axis=1)
            n_bins = len(grouped_hist)

            grouped_hist = np.log1p(grouped_hist)
            grouped_hist /= grouped_hist.max() + 1e-8

            bar_width = plot_width / n_bins

            for j, val in enumerate(grouped_hist):
                bar_height = val * plot_height
                x0 = int(margin + j * bar_width)
                x1 = int(margin + (j + 1) * bar_width)
                y0 = int(baseline - bar_height)
                y1 = baseline
                draw.rectangle([x0, y0, x1, y1], fill=color, outline=(40, 43, 41))


            thr_x = int(margin + (thr[name] / 255) * plot_width)
            draw.line([thr_x, y_offset, thr_x, baseline], fill=(250, 230, 52), width=5)
            draw.text((margin+200, y_offset - 20), f"{name} channel   thr={thr[name]:.0f}", fill="black", font=font_title)

            for tick_val in y_tick_values:
                tick_y = int(baseline - tick_val * plot_height)
                draw.line([margin - 6, tick_y, margin, tick_y], fill="black", width=1)
                label = f"{tick_val:.2f}"
                bbox_text = draw.textbbox((0, 0), label, font=font_axis)
                text_w = bbox_text[2] - bbox_text[0]
                text_h = bbox_text[3] - bbox_text[1]
                draw.text((margin - 10 - text_w, tick_y - text_h // 2), label, fill="black", font=font_axis)

            for tick_val in x_tick_values:
                tick_x = int(margin + (tick_val / 255) * plot_width)
                draw.line([tick_x, baseline, tick_x, baseline + 6], fill="black", width=1)
                label = str(tick_val)
                bbox_text = draw.textbbox((0, 0), label, font=font_axis)
                text_w = bbox_text[2] - bbox_text[0]
                draw.text((tick_x - text_w // 2, baseline + 8), label, fill="black", font=font_axis)

            label_img = Image.new("RGBA", (100, 30), (255, 255, 255, 0))
            label_draw = ImageDraw.Draw(label_img)
            label_draw.text((0, 0), "log(count)", fill="gray", font=font_axis)
            label_img = label_img.rotate(90, expand=True)
            img.paste(label_img, (margin - 65, int(y_offset + plot_height // 2) - 60), label_img)
            draw.text((margin + plot_width // 2 - 40, baseline + 30), "Pixel value", fill="gray", font=font_axis)

        return img