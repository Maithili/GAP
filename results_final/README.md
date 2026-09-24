# Final benchmark results

Human–robot co-adaptation benchmark, retrain setting, 3 households (A/B/C), two
human models (scripted and LLM `redirect`). Built from the full run artifacts by
`scripts/build_public_results.sh`.

## Headline (3-HH mean; Net = actions saved, Perturbed = actions made harder)

| Method | LLM: Net / Perturbed / F1 | Scripted: Net / Perturbed / F1 |
| ------ | ------------------------- | ------------------------------ |
| SLaTe-PRO | 127 (2.1%) / 991 (18.1%) / 0.216 | 888 (16.6%) / 634 (11.8%) / 0.264 |
| STREAK¹ | −8461 (−159.9%) / 538 (10.6%) / 0.002 | 1160 (20.3%) / 2670 (48.3%) / 0.296 |
| Plain LLM | −8522 (−174.5%) / 3010 (62.1%) / 0.076 | 919 (19.5%) / 1208 (25.2%) / 0.310 |
| **Ours** | **964 (17.7%) / 625 (11.7%) / 0.388** | **1637 (30.8%) / 532 (10.1%) / 0.452** |
| Oracle | 2392 (44.4%) / 444 (8.2%) / 0.617 | 2752 (51.3%) / 482 (8.8%) / 0.651 |

¹ STREAK's LLM cell caps actions to 2/timestep to avoid catastrophic collapse
(uncapped: −578% net, 112.5% perturbed). See DESIGN.md §14 A′.

## Layout — `<method>_<scripted|llm_human>/`

- `results_<HH>.json` — the metrics block (per-run totals + per-5-day-window
  trajectory `finetune_by_ckpt`). Every reported number is computed from these.
  (The raw per-tick trace jsons, 1–25 GB each, are not shipped.)
- `checkpoints_<HH>/weights_day*.pt` — trained inference weights per retrain round.
- `memory_<HH>/` — *(ours only)* LLM-curator artifacts: `task_graphs.json`,
  `activities.json`, `tagged_days/`, `gamma_log.jsonl`, `policy_diag.jsonl`, …
- `heatmaps/` — subsampled next-relocation prediction heatmaps (HH-A, one per
  training checkpoint).
- `llm_audit_<HH>.txt` — *(LLM-human runs only)* full human-LLM decision transcripts
  (prompt + `undo`/`allow`/`redirect` + rationale for every robot action).

## Metric definitions

TP = skipped+accepted; FP = corrected+cdc+undo(vigilance)+redirected; FN = done_as_is;
precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2·TP/(2·TP+FP+FN);
Net = savings−(undo+redirected), Net% = Net/solo; Perturbed = corrected+cdc+redirected(+corr_rep), % of solo.
