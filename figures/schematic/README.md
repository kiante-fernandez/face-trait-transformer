# Workflow schematic

A vector (TikZ / LaTeX) source for the model schematic used in the top-level
README. Both the PDF (vector, for papers) and a 300-DPI PNG (for README / HF
rendering) are built from this folder.

## Build

```bash
cd figures/schematic
./build.sh
```

Requires TeX Live (`pdflatex` / `latexmk`) and poppler (`pdftocairo`). On macOS:

```bash
brew install --cask mactex-no-gui          # or the full mactex / basictex
brew install poppler                        # gives you pdftocairo
```

On Linux (Debian/Ubuntu):

```bash
sudo apt-get install texlive-full poppler-utils
```

Outputs:

- `schematic.pdf` — vector; use this for papers / slide decks.
- `schematic.png` — 300-DPI raster; auto-copied to `../schematic.png` so the
  top-level README's image reference works.

## Files

- `schematic.tex` — the source. Self-contained `standalone` LaTeX document
  using only TikZ libraries shipped with any standard TeX Live install
  (positioning, shapes, arrows.meta, calc, fit, backgrounds).
- `build.sh` — compiles + converts PDF → PNG.
- `face_input.jpg` — OMI stimulus 1 used as the input-image thumbnail
  (CC BY-NC-SA 4.0, inherited from the OMI dataset).

## Editing

The schematic is one ~200-line `.tex` file. Common edits:

- **Palette** (lines near the top): `\definecolor{frozen}{HTML}{…}` etc.
  The defaults match matplotlib's `tab:*` ML-paper palette.
- **Add a stage**: add a `\node[stage, right=…pt of previous] (newname) {…}`
  and an arrow.
- **Change the test-performance table** (bottom-right panel): edit the
  monospace-font `\node` near `\colCx`.
- **Swap the input image**: drop a new JPG/PNG here, rename to `face_input.jpg`
  (or update the `\includegraphics` path in the `.tex`).

Then rerun `./build.sh`.
