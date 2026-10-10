# Regenerate everything downstream of the committed per-utterance scores.
#   make reproduce   recompute analysis CSVs, tables and figures, then build the paper
#   make check       reproduce and fail if any committed number changes
#   make test        unit tests
PYTHON ?= python

.PHONY: reproduce analysis report paper check test

reproduce: analysis report paper

analysis:
	$(PYTHON) -m condsites.analysis --jobs jobs/pathway.jsonl --scores results/scores/pathway_scores.csv --output-dir results/analysis > /dev/null

report:
	$(PYTHON) -m condsites.report --results results --output paper
	$(PYTHON) -m condsites.figure_pathway --contrasts results/analysis/pathway_contrasts.csv --output paper/figures/pathway_headline.pdf

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex > /dev/null && bibtex main > /dev/null && pdflatex -interaction=nonstopmode main.tex > /dev/null && pdflatex -interaction=nonstopmode main.tex > /dev/null

check: analysis report
	git diff --exit-code -- results/analysis paper/tables
	@echo "All committed analysis outputs and paper tables reproduce from the score files."

test:
	$(PYTHON) -m pytest -q
