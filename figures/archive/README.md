# Archived figure snapshot — not current evidence

I preserve the four original SVGs and their generator from commit `7e0d60c` here without altering their contents. Their hard-coded arrays and labels disagree with the recorded results table; these files are retained for provenance, not presented as validated measurements. **Do not use them as current result figures.**

The historical histogram has within-scene `n=890` rather than the results table's `889`. The legacy range plot gives within-scene median/max `302/3379`, rather than `308/3376`, and false cross-scene maximum `83`, rather than `85`. Its extra cross-site UAV range and revisit quantiles cannot be traced to the current table. It also draws a 100-inlier threshold that should not be taken as a validated general filter.

[`results/reported/legacy-figure-data.json`](../../results/reported/legacy-figure-data.json) preserves these numbers in machine-readable form. The active generator, [`tools/make_figures.py`](../../tools/make_figures.py), excludes them. The archived Python file is an exact source copy for inspection; its relative output path belongs to its former location and it should not be run here.

New diagrams and result plots live one directory above and document their source data in [`figures/README.md`](../README.md). Legacy licensing is recorded in the repository's [license history](../../LICENSES/MIT-legacy.txt).
