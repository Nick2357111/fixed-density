"""错误提示卡片：当谱面无法分析时，用一张与插件卡片风格相近的图片替换分析结果。"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CARD_WIDTH = 475
PADDING = 22
BACKGROUND = (20, 22, 28, 255)
BORDER = (54, 60, 74, 255)
TITLE_COLOR = (246, 251, 255, 255)
BODY_COLOR = (168, 176, 192, 255)
META_COLOR = (120, 128, 145, 255)
BADGE_COLOR = (249, 77, 121, 255)

_BOLD_CANDIDATES = (
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
_REGULAR_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)


@lru_cache(maxsize=8)
def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont:
    for candidate in (_BOLD_CANDIDATES if bold else _REGULAR_CANDIDATES):
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for char in paragraph:
            candidate = current + char
            if current and draw.textlength(candidate, font=font) > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        lines.append(current)
    return lines


def render_error_card(reason: str, *, title: str = "无法分析该谱面", subtitle: str = "") -> bytes:
    """生成错误提示卡片 PNG 字节。"""
    title_font = _font(26, bold=True)
    body_font = _font(17, bold=False)
    meta_font = _font(14, bold=False)

    probe = Image.new("RGB", (CARD_WIDTH, 10))
    probe_draw = ImageDraw.Draw(probe)
    max_text_width = CARD_WIDTH - PADDING * 2 - 46

    body_lines = _wrap(probe_draw, reason, body_font, max_text_width)
    sub_lines = _wrap(probe_draw, subtitle, meta_font, max_text_width) if subtitle else []

    body_height = len(body_lines) * 26 + (len(sub_lines) * 21 if sub_lines else 0)
    card_height = PADDING * 2 + 42 + body_height + 12

    scale = 2
    image = Image.new("RGBA", (CARD_WIDTH * scale, card_height * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (0, 0, CARD_WIDTH * scale - 1, card_height * scale - 1),
        radius=18 * scale,
        fill=BACKGROUND,
        outline=BORDER,
        width=2 * scale,
    )

    def px(value: float) -> int:
        return int(round(value * scale))

    draw.ellipse(
        (px(PADDING), px(PADDING), px(PADDING + 30), px(PADDING + 30)),
        fill=BADGE_COLOR,
    )
    badge_font = _font(20, bold=True)
    badge_box = draw.textbbox((0, 0), "!", font=badge_font)
    draw.text(
        (
            px(PADDING + 15) - (badge_box[2] - badge_box[0]) / 2,
            px(PADDING + 15) - (badge_box[3] + badge_box[1]) / 2 - px(1),
        ),
        "!",
        font=badge_font,
        fill=(255, 255, 255, 255),
    )

    draw.text((px(PADDING + 44), px(PADDING + 3)), title, font=title_font, fill=TITLE_COLOR)

    y = PADDING + 48
    for line in body_lines:
        draw.text((px(PADDING), px(y)), line, font=body_font, fill=BODY_COLOR)
        y += 26
    for line in sub_lines:
        draw.text((px(PADDING), px(y)), line, font=meta_font, fill=META_COLOR)
        y += 21

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
