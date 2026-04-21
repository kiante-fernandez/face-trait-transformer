#!/bin/bash
# Build the schematic PDF (vector) and a high-DPI PNG alongside it.
# Requires: TeX Live (pdflatex/latexmk) and poppler's pdftocairo.
set -euo pipefail
cd "$(dirname "$0")"

# 1. Compile schematic.tex -> schematic.pdf
latexmk -pdf -interaction=nonstopmode -halt-on-error schematic.tex

# 2. High-DPI PNG for README / slides. 300 DPI is standard for papers.
pdftocairo -png -r 300 -singlefile schematic.pdf schematic

# 3. Duplicate the PNG at the figures/ root so the README's existing path works.
cp schematic.png ../schematic.png

# 4. Tidy the latex aux litter.
latexmk -c >/dev/null 2>&1 || true

echo "Built: $(pwd)/schematic.pdf (vector) + $(pwd)/schematic.png ($(pdftocairo -l 1 -png -r 72 schematic.pdf /tmp/probe >/dev/null 2>&1; sips -g pixelWidth ../schematic.png 2>/dev/null | tail -1 | awk '{print $2}')px wide @ 300dpi)"
