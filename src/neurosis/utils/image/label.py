from typing import Optional

from PIL import Image, ImageDraw

from neurosis.data import get_image_font


def label_image(
    image: Image.Image,
    idx: Optional[int] = None,
    step: Optional[int] = None,
    idx_len: int = 2,
    step_len: int = 5,
    colour: str = "white",
    outline: str = "black",
) -> Image.Image:
    if step is None and idx is None:
        raise ValueError("At least one of idx or step must be provided")

    font_sz = 24 if min(image.size) >= 256 else 16
    font = get_image_font(size=font_sz)
    offs = font_sz // 2
    stroke = max(font_sz // 8, 2)

    labels = []
    if step is not None:
        labels.append(f"S{step:0{step_len}d}")
    if idx is not None:
        labels.append(f"{idx:0{idx_len}d}")

    draw = ImageDraw.Draw(image, mode="RGBA")
    draw.text(
        (offs, offs),
        "-".join(labels),
        font=font,
        fill=colour,
        stroke_fill=outline,
        stroke_width=stroke,
        anchor="lt",
    )
    return image.convert("RGB")


def label_batch(
    images: list[Image.Image],
    step: int,
    step_len: int = 5,
    colour: str = "white",
    outline: str = "black",
    copy: bool = False,
):
    idx_len = max(len(str(len(images))), 2)
    labeled = [
        label_image(
            image.copy() if copy else image,
            idx,
            step,
            idx_len,
            step_len,
            colour,
            outline,
        )
        for idx, image in enumerate(images)
    ]
    return labeled
