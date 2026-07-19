# Paper package — H9DLGA Repeat (x24153362)

**Student:** Marco Aurélio Minozzo (`x24153362`)

## Submission PDF (sem cover sheet)

`H9DLGA_Repeat_x24153362_Marco_Aurelio_Minozzo_Report.pdf`

Conteúdo atual:
1. Relatório IEEE (`main.pdf`, IEEEtran)
2. Question 2 — Paper Review (`q2_review.pdf`)

## Para você fazer depois
- **Cover sheet** NCI (assinatura/data) — ficheiro `cover.tex` / `cover.pdf` está pronto se quiser usar, mas **não** está incluído neste PDF
- Slides da apresentação (~7)
- Anexar a cover ao PDF final antes do Turnitin, se a school exigir

## Compilar bundle sem capa
```bash
cd paper
pdflatex main.tex && pdflatex main.tex
pdflatex q2_review.tex && pdflatex q2_review.tex
pdfunite main.pdf q2_review.pdf H9DLGA_Repeat_x24153362_Marco_Aurelio_Minozzo_Report.pdf
```

## Compilar com capa (quando estiver pronta)
```bash
pdfunite cover.pdf main.pdf q2_review.pdf H9DLGA_Repeat_x24153362_Marco_Aurelio_Minozzo_Report.pdf
```
