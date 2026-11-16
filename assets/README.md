# astroinfer brand assets

The logo is three nested elliptical contours rotated by -38 degrees, in the brand blue, around a
small amber point, followed by the lowercase wordmark with `astro` in navy and `infer` in blue.

| file | use |
|---|---|
| `astroinfer-logo.svg`, `astroinfer-logo.png` | horizontal logo for light backgrounds (PNG is 1200 px wide) |
| `astroinfer-logo-dark.svg`, `astroinfer-logo-dark.png` | the same, with a pale `astro`, for dark backgrounds |
| `astroinfer-icon.svg`, `astroinfer-icon-512.png` | the mark on its own |
| `favicon.svg`, `favicon-16.png`, `favicon-32.png`, `favicon.ico` | site icon, with two heavier contours so it reads at 16 px |
| `social-preview.png` | 1280 x 640 image for the GitHub social preview, with the tagline |

The documentation uses copies of the four SVG files and `favicon.ico` in `docs/assets/`. None of these files are
part of the Python packages.

## Colours

| role | hex | use |
|---|---|---|
| navy | `#17243D` | `astro` in the wordmark, headings and ordinary text |
| blue | `#418FC5` | the contours, `infer` in the wordmark, accents |
| amber | `#E0A348` | the central point, and rare highlights |
| pale gray | `#E8EDF3` | `astro` on dark backgrounds, subtle backgrounds and rules |

The blue is too light for small text on white, so text and buttons use navy. In the tutorial
plots (`tutorials/_style.py`) the posterior is drawn in the brand blue and true values in navy.

## Regenerating the files

The wordmark in the two logo SVG files is drawn as vector paths, so it looks the same on every
machine. `make_logos.py` writes both files from Inter SemiBold (the 28pt optical size, under the
SIL Open Font License), with the font's own kerning. The font file is not stored in the
repository; pass its path:

```bash
uv run --with fonttools --with uharfbuzz python assets/make_logos.py path/to/Inter_28pt-SemiBold.ttf
cp assets/astroinfer-logo.svg assets/astroinfer-logo-dark.svg docs/assets/
INTER_FONT=path/to/Inter_28pt-SemiBold.ttf assets/export_pngs.sh
```

`export_pngs.sh` renders the PNG files and `favicon.ico` with headless Chrome and Pillow.
`INTER_FONT` is only used for the tagline in `social-preview.png`.
