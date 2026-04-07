# Proposal Report Template

Template nay duoc khoi tao tu format LaTeX cua `Lab2-LT/report` va da duoc lam sach de dung cho proposal hien tai.

## Cau truc

```text
report/
|-- main.tex
|-- cover.tex
|-- preface.tex
|-- info.tex
|-- Glossary.tex
|-- references.bib
|-- assets/
|   `-- HCMUS.png
|-- sections/
|   `-- content_main.tex
|-- figures/
|-- tables/
`-- appendix/
```

## Cach dung

1. Sua metadata trong `main.tex`.
2. Viet noi dung trong `sections/content_main.tex`.
3. Them tai lieu tham khao vao `references.bib`.
4. Neu can bang thuat ngu, bo comment dong `\\include{Glossary}` trong `main.tex`.

## Bien dich

Chay trong thu muc `report/`:

```powershell
xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build main.tex
biber --input-directory build --output-directory build main
xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build main.tex
```
