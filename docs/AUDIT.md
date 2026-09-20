# Repository audit and evidence status

I reviewed this individual experiment on 20 September 2026, starting from revision `7e0d60c70d523228e33d9175b0f66e1ed1e6eddc`. I checked every tracked file and inventoried the supplied local working material. I used the PromptMask repository and framework illustration to study information hierarchy, then designed original SfM diagrams. No PromptMask artwork or experimental result was imported.

This is a repository and implementation review. I did **not** rerun the historical GPU experiments or independently confirm the original measurements.

## Scope of the file review

| Area | Review and action |
| :--- | :--- |
| `README.md` | Rebuilt the overview around the research question, pipeline, compact results, setup, documentation, and ongoing status; retained first-person author voice |
| `RESULTS.md` | Preserved all 11 original numerical tables; corrected definitions, evidence claims, units, and causal interpretation |
| `docs/porting-notes.md` | Retained the seven migration sections and historical snippets; scoped version/build observations to the recorded environment |
| Original four `src/*.py` files | Reviewed experiment assembly, pair selection, database import, mapping, metrics, and pose conversion; implemented corrections below |
| Original figure generator and four SVGs | Found conflicting summaries; preserved them in `figures/archive/` with their original values |
| New diagrams and result figures | Illustrated method/failure plates plus editable overview and result charts, numerical source CSVs, descriptions, and provenance hashes; all final exports receive visual review |
| `LICENSE`, `.gitignore` | Reserved rights for new original material, preserved the earlier MIT grant, excluded manuscripts and local machine/runtime material |
| Local drafts and study guide | Reviewed for context only; manuscript files remain local; synthetic teaching diagrams are not treated as measured results |
| Separate local tooling | Kept the Colab connection-tool checkout, machine configuration, and earlier baseline scratch work outside this repository |

The source-folder inventory contained 82 regular files after excluding Git/runtime caches, including unrelated tooling, duplicate exports, and build files. This is not a claim that all 82 files are experimental artifacts or that separate third-party software received a line-by-line audit.

## Evidence I have, and evidence I still need

| Material | Status |
| :--- | :--- |
| Historical Markdown tables | Preserved; source snapshot identified by Git revision |
| Local narrative notes | Available; record that hosted runtimes were reclaimed and raw logs were lost |
| Per-run logs, databases, poses, per-pair arrays | Not located in the supplied source tree |
| New plot inputs | Transcriptions of historical tables; not newly measured results |
| Old histogram bins | Preserved separately; unknown run identity and conflicting summary statistics |
| Fresh corrected-pipeline benchmarks | Pending |

I removed the claim that every table cell can currently be traced to an available log line. The figures re-express recorded summaries; they do not fill the missing raw-evidence gap.

## Corrections to interpretation

1. **Pipeline comparison, not matcher-only causation.** Detector, descriptor, matcher, realized feature counts, and camera initialization differ. The learned importer uses `SIMPLE_PINHOLE`; the native SIFT path uses COLMAP reader defaults. A shared mapper does not isolate LightGlue.
2. **The stock arm changes several variables.** Its recovery on `pipes` cannot establish a resolution-only explanation.
3. **Mill 19 has dataset reference poses.** I do not claim they are independent survey measurements or interpret normalized coordinates as metres.
4. **Full rotation error is not optical-axis reversal.** The >170° count can include a large roll. The `relief_2` count covers 20 evaluated cameras in the dominant component.
5. **Absolute errors are conditional on registration and component selection.** I keep coverage beside the medians.
6. **The old inlier ratio has a conditional denominator.** It excludes failed-verification pairs and is not ground-truth correspondence precision. Historical values keep that definition; new runs report both ratios.
7. **Revisit sessions share one physical scene.** Their reference coordinate frames remain separate for pose evaluation.
8. **The three courtyard values are not a controlled repeat set.** In particular, 849 mm comes from the mixed-scene run.
9. **Continuity is not a pose-convention proof.** A fixed axis flip can preserve orthonormality and adjacent view-angle changes.
10. **Timing and scale claims need limits.** Historical front ends use different devices. One mapping-time comparison does not establish a cost law or large-survey performance.

## Implementation corrections for new runs

| Issue | Current behavior | Compatibility consequence |
| :--- | :--- | :--- |
| Self counted toward the neighbour floor; final query omitted | Corrected pair selection, with a `legacy` option | Candidate pairs may change; old counts remain historical |
| ALIKED resize was passed to its constructor rather than extraction preprocessing | Pass `resize` explicitly to `extract` | Nondefault image-size experiments now use the requested value; default 1024 is unchanged |
| Empty shortlist was reported as the exhaustive pair count | Report zero pairs and skip native matching for empty lists | Reported candidate count reflects actual selection |
| Empty COLMAP observation lines broke pose parsing | Parse complete two-line records, including empty second lines | Pose association is corrected |
| Session labels used as scene labels | Distinct physical-scene and reference-frame identities | Revisit scene agreement changes to the intended definition |
| Failed-verification pairs excluded from pooled ratio | Report all-pair and legacy verified-pair denominators separately | New `inlier_ratio` is not directly comparable with old rows |
| Geometric verification implicit in a matching call | Explicit verification of imported learned correspondences | Fresh runs need validation; this is a behavior change |
| Shared or reused output directories | Unique run and image-staging directories | Prior run artifacts are retained |
| Filtered ablation reported original DB diagnostics | Evaluate filtered DB, retain original result, separate timing | Subsequent ablation reports reflect the graph actually mapped |
| Broad imports blocked `--help` | Load ML dependencies only for relevant operations | CLI inspection needs no model packages |
| Insufficient error checks and unsafe path handling | Validate inputs, mark degenerate alignment metrics unavailable, preserve unmanaged files, use argument-list subprocesses | Invalid inputs fail explicitly; missing orientation counts are null, not measured zero |
| Only console summaries retained | Strict JSON reports with configuration, hashes, runtime, poses, metrics, and camera/feature metadata | Reports start after input preparation; source data must remain available |

I retain a legacy shortlisting option for diagnosis, but it is not a switch that reconstructs the entire old environment. Other corrections above still apply. A fresh result should be identified by its report, code hash, input hashes, and configuration.

## Validation and limits

I use `python -m unittest discover -s tests -v` for numerical, parsing, dataset-staging, reporting, and optional native COLMAP checks. `python tools/validate_repository.py` checks local links, Python syntax, retained numerical tables, CSV consistency, figure provenance, and manuscript exclusion. The figure generator runs without model or dataset downloads. GitHub Actions repeats the lightweight checks.

I visually inspected all eight current figure exports. The implementation tests use synthetic fixtures; optional native checks use small CPU-generated inputs. They do not verify the historical scene reconstructions. Full learned inference, dataset downloads, GPU runtime/memory, and end-to-end experiment reproduction remain untested in this revision.

## Unresolved work

- Recover the missing original artifacts or rerun the benchmark with retained evidence.
- Resolve the earlier Kaggle baseline's author, URL, and license before making stronger source-ownership claims.
- Match camera models, realized feature counts, and verification settings in further controls.
- Validate physical pose conventions against image observations and independent evaluation code.
- Keep manuscript sources, PDFs, and build files out of this update.

[Next experiments](ROADMAP.md) · [Rights and earlier MIT scope](LICENSING.md)
