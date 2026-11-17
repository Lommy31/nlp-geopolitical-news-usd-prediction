#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

pages=$(LC_ALL=C pdfinfo build/main.pdf | awk '/^Pages:/ {print $2}')
if [[ ! "$pages" =~ ^[0-9]+$ ]]; then
    echo "Could not determine the PDF page count." >&2
    exit 1
fi
if (( pages > 5 )); then
    echo "Report has $pages pages; the maximum is 5. Shorten it and rebuild." >&2
    exit 1
fi

cp build/main.pdf main.pdf
echo "Build successful: main.pdf ($pages/5 pages)."
