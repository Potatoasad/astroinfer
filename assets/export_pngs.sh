#!/usr/bin/env bash
# Render the PNG exports and favicon.ico from the SVGs in this folder, with headless Chrome
# (transparent background) and Pillow (downscaling and the .ico file). Run from the repo root:
#
#     assets/export_pngs.sh
#
# The logo SVGs draw the wordmark as paths (see make_logos.py), so the PNGs do not depend on
# installed fonts. Set INTER_FONT to an Inter .ttf file to set the social-preview tagline in Inter;
# otherwise it uses the system sans-serif.
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# render <svg> <width> <height> <out.png> [background]
render() {
    local bg="${5:-transparent}" out="$4"
    [[ "$out" = /* ]] || out="$PWD/$out"
    cat > "$tmp/page.html" <<EOF
<html><body style="margin:0;background:$bg"><img src="file://$PWD/$1" style="display:block;width:$2px;height:$3px"></body></html>
EOF
    "$CHROME" --headless=new --disable-gpu --hide-scrollbars --default-background-color=00000000 \
        --window-size="$2,$3" --screenshot="$out" "file://$tmp/page.html" >/dev/null 2>&1
}

# The logo is 160 units high; read its width from the SVG to keep the aspect ratio at 1200 px.
logo_w="$(sed -n 's/.*viewBox="0 0 \([0-9]*\) 160".*/\1/p' astroinfer-logo.svg | head -1)"
logo_h=$(( 1200 * 160 / logo_w ))
render astroinfer-logo.svg 1200 "$logo_h" astroinfer-logo.png
render astroinfer-logo-dark.svg 1200 "$logo_h" astroinfer-logo-dark.png
render astroinfer-icon.svg 512 512 astroinfer-icon-512.png
render favicon.svg 256 256 "$tmp/favicon-256.png"

# Social preview: the logo centred on white, with the tagline below.
font_face=""
if [[ -n "${INTER_FONT:-}" ]]; then
    font_face="<style>@font-face{font-family:Inter;font-weight:600;src:url('file://$INTER_FONT')}</style>"
fi
cat > "$tmp/social.html" <<EOF
<html><head>$font_face</head><body style="margin:0;width:1280px;height:640px;background:#fff;display:flex;flex-direction:column;
align-items:center;justify-content:center;font-family:Inter,'IBM Plex Sans',system-ui,-apple-system,sans-serif">
<img src="file://$PWD/astroinfer-logo.svg" style="width:680px">
<div style="color:#17243D;font-size:34px;font-weight:600;margin-top:18px;letter-spacing:-0.3px">
Generative modeling and Bayesian inference for astrophysics</div></body></html>
EOF
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --window-size=1280,640 \
    --screenshot="$PWD/social-preview.png" "file://$tmp/social.html" >/dev/null 2>&1

uv run --with pillow python - "$tmp/favicon-256.png" <<'EOF'
import sys
from PIL import Image
big = Image.open(sys.argv[1]).convert("RGBA")
for size in (16, 32):
    big.resize((size, size), Image.LANCZOS).save(f"favicon-{size}.png")
big.save("favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
EOF
ls -la *.png favicon.ico
