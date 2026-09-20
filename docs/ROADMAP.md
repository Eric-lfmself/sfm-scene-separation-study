# Research roadmap

I separate repository work from experiments that still need to be performed. A checked item means the implementation or documentation exists, not that the historical benchmark was reproduced.

## Repository and evaluation

- [x] Organize the overview, method, experiment guide, result data, and figure sources.
- [x] Publish original framework and explanatory diagrams with explicit schematic labels.
- [x] Preserve recorded values with their evidence limits and incompatible old figure inputs.
- [x] Correct shortlist self-selection/final-query behavior and retain a legacy option.
- [x] Separate physical scenes from acquisition/reference frames for revisits.
- [x] Distinguish all-pair and verified-pair inlier ratios.
- [x] Add lightweight regression checks and per-run provenance reports.
- [ ] Complete an end-to-end learned and SIFT reconstruction validation in the intended GPU environment.

## First priority: recover reproducible evidence

- [ ] Recover earlier logs, pair manifests, databases, and reconstructed poses, if copies still exist.
- [ ] Otherwise rerun each table configuration with a fixed input manifest and archived environment.
- [ ] Retain per-pair errors, correspondence counts, camera intrinsics, and component membership.
- [ ] Repeat identical configurations across declared seeds; report all runs and coverage.
- [ ] Regenerate figures directly from those reports, replacing historical-summary provenance only after verification.

## Controls that matter

- [ ] Match realized feature budgets as well as configured caps.
- [ ] Hold camera models, focal priors, and verification settings constant where possible.
- [ ] Vary resolution, feature cap, and pair selection separately.
- [ ] Cross-check pose conversion and alignment with independent implementation/image projections.
- [ ] Measure optical-axis errors separately from full rotation errors.
- [ ] Compare detector/matcher combinations before assigning a failure to one component.
- [ ] Evaluate post-verification edge thresholds for both false merges and lost true links.

## Broader research

- [ ] Test additional sites, wide baselines, lower overlap, and illumination changes.
- [ ] Evaluate a classical-first pipeline with learned matching reserved for unresolved cases.
- [ ] Replace dense retrieval distances with a scalable search strategy and measure retrieval recall.
- [ ] Benchmark larger surveys with end-to-end runtime, memory, and independently justified pose metrics.
- [ ] Resolve the earlier IMC baseline's code provenance.

I do not currently claim production deployment, a completed hybrid pipeline, or measured 12,000-image performance from the evidence in this repository.
