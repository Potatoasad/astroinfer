"""Write astroinfer-logo.svg and astroinfer-logo-dark.svg with the wordmark as vector paths.

    uv run --with fonttools --with uharfbuzz python assets/make_logos.py path/to/Inter_28pt-SemiBold.ttf

The wordmark is set in Inter SemiBold (SIL Open Font License; the 28pt optical size is drawn for
display text) at the size, tracking and position of the reference logo, with the font's own
kerning. Each glyph outline becomes an SVG path, so the logo does not depend on installed fonts
and the font file itself is not needed afterwards. Copy the results to docs/assets/ and run
assets/export_pngs.sh to update the PNG files.
"""
import math
import sys
from pathlib import Path

import uharfbuzz as hb
from fontTools.misc.transform import Transform
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

NAVY, BLUE, AMBER, PALE = "#17243D", "#418FC5", "#E0A348", "#E8EDF3"
FONT_SIZE = 87          # px, as in the reference logo
TRACKING = -3.2         # px between letters
X0, BASELINE = 170, 104
HEIGHT = 160
RIGHT_MARGIN = 12       # matches the margin left of the mark

MARK = f"""  <g transform="translate(12 12)">
    <g fill="none" stroke="{BLUE}" stroke-width="4.4">
      <ellipse cx="68" cy="68" rx="61" ry="31" transform="rotate(-38 68 68)"/>
      <ellipse cx="68" cy="68" rx="40" ry="20" transform="rotate(-38 68 68)" opacity=".65"/>
      <ellipse cx="68" cy="68" rx="19" ry="9.5" transform="rotate(-38 68 68)" opacity=".4"/>
    </g>
    <circle cx="68" cy="68" r="7.5" fill="{AMBER}"/>
  </g>"""


def number(v):
    return f"{v:.2f}".rstrip("0").rstrip(".")


def wordmark(font_path):
    """Return the SVG path data for 'astro' and 'infer', and the right edge of the ink."""
    data = Path(font_path).read_bytes()
    face = hb.Face(hb.Blob(data))
    hb_font = hb.Font(face)
    buf = hb.Buffer()
    buf.add_str("astroinfer")
    buf.guess_segment_properties()
    hb.shape(hb_font, buf, {"kern": True, "liga": True, "calt": True})

    tt = TTFont(font_path)
    glyphs = tt.getGlyphSet()
    order = tt.getGlyphOrder()
    scale = FONT_SIZE / tt["head"].unitsPerEm
    tracking = TRACKING / scale  # in font units

    runs = {"astro": SVGPathPen(glyphs, ntos=number), "infer": SVGPathPen(glyphs, ntos=number)}
    x, right = 0.0, -math.inf
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        name = order[info.codepoint]
        run = runs["astro" if info.cluster < 5 else "infer"]
        to_page = Transform(scale, 0, 0, -scale, X0, BASELINE).translate(x + pos.x_offset, pos.y_offset)
        glyphs[name].draw(TransformPen(run, to_page))
        bounds = BoundsPen(glyphs)
        glyphs[name].draw(TransformPen(bounds, to_page))
        if bounds.bounds:
            right = max(right, bounds.bounds[2])
        x += pos.x_advance + tracking
    return runs["astro"].getCommands(), runs["infer"].getCommands(), right


def logo(astro_path, infer_path, width, astro_fill, note):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{HEIGHT}" viewBox="0 0 {width} {HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">astroinfer</title>
  <desc id="desc">Three tilted elliptical posterior contours with a golden central point, followed by the lowercase astroinfer wordmark{note}.</desc>
{MARK}
  <path fill="{astro_fill}" d="{astro_path}"/>
  <path fill="{BLUE}" d="{infer_path}"/>
</svg>
"""


def main():
    astro, infer, right = wordmark(sys.argv[1])
    width = math.ceil(right + RIGHT_MARGIN)
    here = Path(__file__).parent
    (here / "astroinfer-logo.svg").write_text(logo(astro, infer, width, NAVY, ""))
    (here / "astroinfer-logo-dark.svg").write_text(
        logo(astro, infer, width, PALE, ", for dark backgrounds"))
    print(f"wrote astroinfer-logo.svg and astroinfer-logo-dark.svg, {width} x {HEIGHT}")


if __name__ == "__main__":
    main()
