# GeoTrace-Agent paper

A NeurIPS-style preprint covering the system design of [GeoTrace-Agent](../README.md). Self-contained: no external `.bib`, no PNG figures (TikZ only), so the source compiles on stock TeX Live and is ready for arXiv submission.

## Build

```bash
make pdf          # produces geotrace_agent_neurips.pdf
make lint         # chktex if installed
make clean        # remove .aux/.log/.out
make distclean    # also remove the PDF
make arxiv        # build arxiv-build/ and a single-file tarball
```

## arXiv submission checklist

1. `make pdf` produces a clean PDF with no `Undefined references` or `Citation undefined` warnings.
2. `pdffonts geotrace_agent_neurips.pdf` shows only Type 1 / TrueType fonts (arXiv's policy).
3. `make arxiv` produces `geotrace_agent_neurips-arxiv.tar.gz`. Upload at <https://arxiv.org/submit>.
4. arXiv categories (suggested, in priority order): `cs.AI`, `cs.MA`, `cs.LG`, `cs.SI`.
5. Co-authors. The current draft is solo (Arun Sharma) with an Acknowledgments section thanking advisors. To convert to co-authored, edit the `\author{}` block in `geotrace_agent_neurips.tex` to use the `\And` separator pattern.
6. Title-line typo check, abstract length under 250 words.

## Layout

- `geotrace_agent_neurips.tex` — single-file source, NeurIPS-style preamble, inline `thebibliography`.
- `Makefile` — build targets.

## Citation

Once an arXiv ID is assigned, update the BibTeX entry in [`../CITATION.cff`](../CITATION.cff) and the `arXiv:xxxx.xxxxx` badge in [`../README.md`](../README.md).
