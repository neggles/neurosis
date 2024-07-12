"""Concept stolen and reworked from https://github.com/Birch-san/grid-printer"""

from dataclasses import dataclass
from math import ceil, sqrt
from textwrap import TextWrapper
from typing import NamedTuple, Optional

import numpy as np
from PIL import Image, ImageDraw
from PIL.ImageFont import FreeTypeFont

from neurosis.data import get_image_font
from neurosis.utils.misc import batched


@dataclass
class FontMetrics:
    ct: int
    cl: int
    cw: int
    ch: int
    ls: int


class BBox(NamedTuple):
    top: int
    left: int
    bottom: int
    right: int


# comments are with size=24 and NotoSansMono
def get_font_metrics(font: FreeTypeFont):
    """Get metrics for font. Assumes monospaced."""
    cl, ct, cw, cb = font.getbbox("M", anchor="la")  # (0, 8, 14, 26) l, t, r, b
    ch = cb - ct  # 18

    draw = ImageDraw.Draw(Image.new("RGB", (100, 100)))
    _, lt, _, lb = draw.textbbox((0, 0), "M\nM", font=font)  # (0, 8, 29, 56)
    ls = lb - lt - (2 * ch)  # 12

    return FontMetrics(ct=ct, cl=cl, cw=cw, ch=ch, ls=ls)


class FontWrapper:
    def __init__(
        self,
        font: FreeTypeFont,
        padding: BBox = BBox(8, 8, 8, 8),
    ):
        self.font = font
        self.padding = padding
        self.metrics = get_font_metrics(font)
        self.tw = TextWrapper()

    def __call__(self, text: str, img_w: int):
        return self.wrap(text, img_w)

    def wrap(self, text: str, img_w: int):
        wrap_at = ((img_w - (self.padding.left + self.padding.right)) // self.metrics.cw) - 1
        if wrap_at < 1:
            raise ValueError("Image is too small to fit text!")
        self.tw.width = wrap_at
        return self.tw.wrap(text)

    def get_height(self, nlines: int):
        pad_total = self.padding.top + self.padding.bottom
        return nlines * self.metrics.ch + (nlines - 1) * self.metrics.ls + pad_total


def wrap_captions(
    captions: list[str | bytes],
    wrapper: FontWrapper,
    ncols: int,
    img_w: int,
) -> list[str]:
    """Wrap captions to fit in image."""
    r_captions: list[list[str]] = []
    r_heights: list[int] = []

    for col in batched(captions, ncols):
        r_lines = [wrapper.wrap(x, img_w) for x in col]
        max_lines = max(len(x) for x in r_lines)

        r_heights.append(wrapper.get_height(max_lines))
        r_captions.append(["\n".join(x) for x in r_lines])

    r_heights = np.array(r_heights)
    r_ypos = np.roll(r_heights.cumsum(), 1)
    r_ypos[0] = 0

    return r_captions, r_ypos, r_heights


class CaptionGrid:
    def __init__(
        self,
        font: Optional[FontWrapper | FreeTypeFont | int] = None,
        tfont: Optional[FontWrapper | FreeTypeFont | int] = None,
    ):
        if font is None:
            font = get_image_font(size=18)
        elif isinstance(font, int):
            font = get_image_font(size=font)

        if tfont is None:
            tfont = get_image_font(size=48)
        elif isinstance(font, int):
            tfont = get_image_font(size=tfont)

        self.font = font if isinstance(font, FontWrapper) else FontWrapper(font)
        self.tfont = tfont if isinstance(tfont, FontWrapper) else FontWrapper(tfont)

    @property
    def _t_xpos(self):
        return self.tfont.padding.left - self.tfont.metrics.cl

    @property
    def _t_ypos(self):
        return self.tfont.padding.top - self.tfont.metrics.ct

    @property
    def t_pos(self):
        return self._t_xpos, self._t_ypos

    def __call__(
        self,
        images: list[Image.Image],
        captions: list[str | bytes],
        title: Optional[str | bytes] = None,
        ncols: Optional[int] = None,
    ) -> Image.Image:
        if len(images) == 0:
            raise ValueError("Must provide at least one image!")
        if len(images) != len(captions):
            raise ValueError("Number of images and captions must match!")

        if ncols is None:
            ncols = ceil(sqrt(len(images)))

        if not isinstance(captions, list):
            captions = [captions]

        nrows = ceil(len(images) / ncols)
        img_w, img_h = images[0].size

        if title is None:
            t_wrapped, t_height = None, 0
        else:
            # handle title wrapping and height
            t_lines = self.tfont.wrap(title, img_w * ncols)
            t_height = self.tfont.get_height(len(t_lines))
            t_wrapped = "\n".join(t_lines)

        # handle caption wrapping and height
        c_wrapped, c_ypos, c_heights = wrap_captions(captions, self.font, ncols, img_w)

        # calculate image height
        row_height = c_heights.sum() + nrows * (self.font.padding.top + self.font.padding.bottom + img_h)

        grid_width = img_w * ncols
        grid_height = row_height + t_height

        # create image
        image = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
        dw = ImageDraw.Draw(image)

        # draw title
        if t_height > 0:
            dw.rectangle((0, 0, grid_width, t_height), fill=(235, 235, 235))
            dw.multiline_text(self.t_pos, t_wrapped, font=self.tfont.font, fill=(0, 0, 0))

        # draw captions and images
        c_xoffs = self.font.padding.left - self.font.metrics.cl
        c_yoffs = self.font.padding.top - self.font.metrics.ct

        for idx, (r_imgs, r_capts, r_yoffs, c_height) in enumerate(
            zip(batched(images, ncols), c_wrapped, c_ypos, c_heights)
        ):
            r_ypos = t_height + r_yoffs + idx * (self.font.padding.top + self.font.padding.bottom + img_h)

            img_y = r_ypos + self.font.padding.top + c_height + self.font.padding.bottom
            capt_y = r_ypos + c_yoffs
            for c_idx, (img, capt) in enumerate(zip(r_imgs, r_capts)):
                img_x = c_idx * img_w
                capt_x = img_x + c_xoffs
                dw.multiline_text((capt_x, capt_y), capt, font=self.font.font, fill=(0, 0, 0))
                image.paste(img, box=(img_x, img_y))

        return image
