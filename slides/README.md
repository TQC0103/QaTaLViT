# Slides

This folder contains Beamer sources and compiled PDFs for the project presentation material.

- `qatalvit_high_level_slides.tex`: concise presentation deck.
- `qatalvit_report_summary_slides.tex`: extended report-summary deck.
- `*_script.md`: speaker notes/scripts.
- `assets/`: figures used by the decks.

Build from this folder:

```powershell
xelatex -interaction=nonstopmode qatalvit_high_level_slides.tex
xelatex -interaction=nonstopmode qatalvit_report_summary_slides.tex
```
