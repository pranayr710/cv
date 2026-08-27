"""Drawing helpers for the live windows, so they look like an application.

``cv2.putText`` only offers Hershey stroke fonts, which is why every OpenCV
demo looks the same and looks rough. These helpers render text with Pillow
using a real system font and composite the result back into the BGR frame, and
give panels, bars and sparklines a consistent palette.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

#: Dark palette, BGR because that is what OpenCV frames are in.
INK = (238, 240, 242)
MUTED = (150, 150, 155)
PANEL = (26, 24, 23)
PANEL_2 = (38, 35, 34)
LINE = (58, 54, 52)
ACCENT = (120, 200, 90)
WARN = (60, 170, 235)
BAD = (70, 90, 235)
DIM = (90, 88, 92)

GAZE_COLOURS = {
    "teacher": (120, 200, 90),
    "left": (235, 160, 80),
    "right": (215, 120, 190),
    "down": (60, 170, 235),
    "back": (120, 120, 125),
}

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/arial.ttf",
)
_BOLD_CANDIDATES = (
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


@lru_cache(maxsize=32)
def font(size: int, bold: bool = False):
    """Load a system UI font at ``size``, falling back to Pillow's default.

    Args:
        size: Point size.
        bold: Whether to prefer the bold face.

    Returns:
        A Pillow font object. Cached, because loading per frame is wasteful.
    """
    from PIL import ImageFont

    for path in (_BOLD_CANDIDATES if bold else _FONT_CANDIDATES):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Canvas:
    """A frame being drawn on, with text batched through one Pillow pass.

    Compositing to Pillow per string would cost several conversions per frame.
    This collects every text call and pays that cost once, in :meth:`finish`.
    """

    def __init__(self, image: np.ndarray) -> None:
        """Wrap an image for drawing.

        Args:
            image: A BGR frame. Drawn on in place for shapes; text is applied
                when :meth:`finish` is called.
        """
        self.image = image
        self._text: list[tuple] = []

    # -- shapes: straight to the array ------------------------------------ #

    def rect(self, x, y, w, h, colour, alpha: float = 1.0, radius: int = 0):
        """Draw a filled rectangle, optionally translucent and rounded."""
        import cv2

        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h <= 0:
            return
        target = self.image[max(0, y):y + h, max(0, x):x + w]
        if target.size == 0:
            return
        patch = np.empty_like(target)
        patch[:] = colour
        if radius > 0:
            mask = np.zeros(target.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (radius, 0), (target.shape[1] - radius, target.shape[0]), 255, -1)
            cv2.rectangle(mask, (0, radius), (target.shape[1], target.shape[0] - radius), 255, -1)
            for cx, cy in ((radius, radius), (target.shape[1] - radius, radius),
                           (radius, target.shape[0] - radius),
                           (target.shape[1] - radius, target.shape[0] - radius)):
                cv2.circle(mask, (cx, cy), radius, 255, -1)
            blend = (mask[..., None] / 255.0) * alpha
        else:
            blend = alpha
        target[:] = (target * (1 - blend) + patch * blend).astype(np.uint8)

    def outline(self, x, y, w, h, colour, thickness: int = 2):
        """Draw a rectangle outline."""
        import cv2

        cv2.rectangle(self.image, (int(x), int(y)), (int(x + w), int(y + h)),
                      colour, thickness, cv2.LINE_AA)

    def bar(self, x, y, w, h, fraction: float, colour, track=LINE):
        """Draw a horizontal progress bar filled to ``fraction`` of its width."""
        self.rect(x, y, w, h, track, radius=h // 2)
        filled = max(0, min(1.0, fraction)) * w
        if filled >= 1:
            self.rect(x, y, filled, h, colour, radius=h // 2)

    def stacked_bar(self, x, y, w, h, parts: list[tuple[float, tuple]]):
        """Draw one bar split into coloured segments.

        Args:
            parts: ``(weight, colour)`` pairs; weights are normalised.
        """
        total = sum(p[0] for p in parts) or 1.0
        cursor = float(x)
        for weight, colour in parts:
            seg = w * weight / total
            if seg >= 1:
                self.rect(cursor, y, seg + 1, h, colour)
            cursor += seg

    def sparkline(self, x, y, w, h, values: list[float | None], colour):
        """Plot a 0-1 series, leaving gaps where a value is missing."""
        import cv2

        if not values:
            return
        step = w / max(len(values) - 1, 1)
        previous = None
        for i, v in enumerate(values):
            if v is None:
                previous = None
                continue
            px = int(x + i * step)
            py = int(y + h - max(0.0, min(1.0, v)) * h)
            if previous is not None:
                cv2.line(self.image, previous, (px, py), colour, 2, cv2.LINE_AA)
            previous = (px, py)

    # -- text: batched ----------------------------------------------------- #

    def text(self, x, y, string, size=16, colour=INK, bold=False, anchor="la"):
        """Queue a string for drawing when :meth:`finish` runs."""
        self._text.append((int(x), int(y), str(string), size, colour, bold, anchor))

    def finish(self) -> np.ndarray:
        """Render every queued string and return the finished frame."""
        if not self._text:
            return self.image
        from PIL import Image, ImageDraw

        pil = Image.fromarray(self.image[:, :, ::-1])
        draw = ImageDraw.Draw(pil)
        for x, y, string, size, colour, bold, anchor in self._text:
            draw.text((x, y), string, font=font(size, bold),
                      fill=(colour[2], colour[1], colour[0]), anchor=anchor)
        self.image[:, :, :] = np.asarray(pil)[:, :, ::-1]
        self._text.clear()
        return self.image


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale an image to fill ``width`` x ``height``, preserving aspect.

    Args:
        image: Source BGR image.
        width: Target width.
        height: Target height.

    Returns:
        A new image of exactly the target size, letterboxed on the short axis.
    """
    import cv2

    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = PANEL
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas
