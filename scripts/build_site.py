#!/usr/bin/env python3
"""Build static site: copies assets and embeds all prediction data into public/."""
import json, shutil, os
from pathlib import Path

ROOT   = Path(__file__).parent.parent
PUBLIC = ROOT / "public"
PUBLIC.mkdir(exist_ok=True)

# GitHub Pages: prevent Jekyll processing
(PUBLIC / ".nojekyll").touch()

# Copy background image
shutil.copy(ROOT / "data" / "fifa-world-cup.png", PUBLIC / "background.png")

# Load logos
logos = json.loads((ROOT / "data" / "logos.json").read_text())

# Gather all prediction files, keyed by date string
pred_dir = ROOT / "outputs" / "predictions"
all_data = {}
for f in sorted(pred_dir.glob("*_predictions.json")):
    date = f.stem.replace("_predictions", "")
    try:
        all_data[date] = json.loads(f.read_text())
    except Exception as e:
        print(f"  Warning: could not parse {f.name}: {e}")

print(f"Loaded {len(all_data)} prediction dates.")

# Write combined data.js (loaded by index.html at runtime)
data_js = f"const PREDICTIONS = {json.dumps(all_data, ensure_ascii=False)};\n"
data_js += f"const LOGOS = {json.dumps(logos, ensure_ascii=False)};\n"
(PUBLIC / "data.js").write_text(data_js)

# Copy HTML template
src_html = ROOT / "public_src" / "index.html"
if src_html.exists():
    shutil.copy(src_html, PUBLIC / "index.html")
    print("Site built successfully → ./public/")
else:
    print("ERROR: public_src/index.html not found!")
    raise SystemExit(1)
