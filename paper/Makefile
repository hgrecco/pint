all: paper.pdf

paper.pdf: paper.md paper.bib latex.template
	pandoc --filter pandoc-citeproc --bibliography paper.bib  paper.md \
  --template latex.template -o paper.pdf \
  --pdf-engine=xelatex

clean:
	rm paper.pdf

.PHONY: clean
