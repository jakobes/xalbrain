default:
	pdflatex adjointbidomain
	bibtex adjointbidomain
	pdflatex adjointbidomain
	pdflatex adjointbidomain

diff:
	hg cat adjointbidomain.tex > /tmp/HEAD.tex || true
	latexdiff /tmp/HEAD.tex adjointbidomain.tex > diff.tex
	pdflatex diff
	bibtex diff
	pdflatex diff
	pdflatex diff
