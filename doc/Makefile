TEXFILE=report
SRC_DIR=../src
FIG_DIR=./images
TAB_DIR=./tables
RES_DIR=../results

TABLES = $(TAB_DIR)/gmrf.tex $(TAB_DIR)/hybrid.tex
FIGS = $(FIG_DIR)/gmrf_bic.pdf $(FIG_DIR)/gmrf_con.pdf $(FIG_DIR)/comp_eo6.pdf $(FIG_DIR)/comp_q8.pdf $(FIG_DIR)/mrf_ex.pdf $(FIG_DIR)/bn_ex.pdf $(FIG_DIR)/adj_static.pdf $(FIG_DIR)/static_bic.pdf $(FIG_DIR)/gmrf_res_eo6.pdf $(FIG_DIR)/gmrf_res_q8.pdf $(FIG_DIR)/gmrf_qqplot.pdf $(FIG_DIR)/hrf_qqplot_eo6.pdf $(FIG_DIR)/hrf_qqplot_q8.pdf $(FIG_DIR)/hrf_model.pdf $(FIG_DIR)/nutshell.pdf $(FIG_DIR)/nonparam_qqplot.pdf

all: $(PDF_FIGS) $(TABLES) $(TEXFILE).pdf

$(TABLES): %.tex: $(SRC_DIR)/tables.py
	python $< $@

$(FIGS): %.pdf: $(SRC_DIR)/figures.py $(SRC_DIR)/plot.py
	python $< $@

$(TEXFILE).pdf: $(FIGS) $(TABLES) $(TEXFILE).ltx
	latexmk -pdf -quiet $(TEXFILE).ltx

view:
	evince $(TEXFILE).pdf &

# Clean up stray files
clean:
	rm -fv *.aux *.log *.toc *.blg *.bbl *.synctex.gz
	rm -fv *.out *.bcf *blx.bib *.run.xml
	rm -fv *.fdb_latexmk *.fls
	rm -fv $(TEXFILE).pdf

.PHONY: all clean
