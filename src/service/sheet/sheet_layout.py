import io

from PIL import Image, ImageDraw, ImageFont

# The character sheet people review: one tile per view, labels drawn by code (never by the image model).
TILE_H = 480
PAD = 24
LABEL_H = 36
BG = (240, 242, 245)
INK = (40, 46, 56)


def compose_sheet(views: list[tuple[str, bytes]]) -> bytes:
    """Lay the views out in a row, each scaled to the same height, with its name underneath."""
    tiles = []
    for name, data in views:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        width = round(image.width * TILE_H / image.height)
        tiles.append((name.replace("_", " "), image.resize((width, TILE_H))))
    total = PAD + sum(t.width + PAD for _, t in tiles)
    sheet = Image.new("RGB", (total, TILE_H + LABEL_H + 2 * PAD), BG)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    x = PAD
    for name, tile in tiles:
        sheet.paste(tile, (x, PAD))
        draw.text((x + tile.width / 2, PAD + TILE_H + 8), name, fill=INK, font=font, anchor="ma")
        x += tile.width + PAD
    out = io.BytesIO()
    sheet.save(out, format="PNG")
    return out.getvalue()
