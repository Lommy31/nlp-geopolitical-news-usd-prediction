English IEEE-style report template

Build from this folder with:
    bash build.sh

The PDF is written to main.pdf only after a successful build. The complete
report, including title, authors, figures, tables, and references, must
fit within five pages. An overlength report fails compilation; it is not
automatically shortened. A failed build leaves any previous main.pdf in
place, so check the build result before using the PDF.

Layout: IEEEtran conference, two columns, 10 pt text, A4 paper. The
assignment's one-inch margins override IEEE's default conference margins.

main.tex              Title, authors, and chapter input commands
preamble.tex          Packages, margins, and five-page limit
chapters/             Sections and references
figures/              Figure environments loaded by the relevant chapter
figures/assets/       Image files (PDF, PNG, or JPG)
tables/               Table environments loaded by the relevant chapter

The chapter files are sections of one short paper, not separate pages.
Use input commands, not include commands, to avoid forced page breaks.
Figure and table templates are commented out until real content is ready.
Start directly with the Introduction after the authors.
Do not add an Abstract or Index Terms.

Suggested space budget (not forced page breaks):
    Title, authors, introduction: approximately 1 page
    Methodology: approximately 1.5 pages
    Results and discussion: approximately 1.5 pages
    Conclusion and references: approximately 1 page

References count toward the maximum. Keep only essential figures and
tables; shorten prose rather than shrinking the font or the margins.

No separate Related Work section. Integrate essential cited prior studies
briefly into the Introduction.
