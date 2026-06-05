# PCK Pseudo-Ground-Truth Evaluation

Metric: `PCK@0.05`

Thunder is used as pseudo ground truth and Lightning is compared against it.
This does not measure absolute human-pose accuracy. It measures the
localization drift introduced when the system switches to Lightning.

## Conditions

- Reference model: `thunder`
- Candidate model: `lightning`
- Threshold: `0.05` of the normalized image diagonal
- Normalized distance threshold: `0.0707`
- Min reference confidence: `0.3`
- JSON output: `metrics/pck_pseudo_gt.json`

## Summary

| clip | frames | evaluated | eligible keypoints | PCK | mean distance | Thunder conf | Lightning conf |
|---|---:|---:|---:|---:|---:|---:|---:|
| clip_still.mp4 | 897 | 180 | 2956 | 0.976 | 0.0126 | 0.725 | 0.667 |
| clip_slow.mp4 | 897 | 180 | 3016 | 0.964 | 0.0141 | 0.775 | 0.692 |
| clip_fast.mp4 | 896 | 180 | 2954 | 0.983 | 0.0123 | 0.746 | 0.672 |
| **aggregate** | 2690 | 540 | 8926 | 0.974 | 0.0130 | 0.749 | 0.677 |

## Notes

- `PCK` counts keypoints whose Lightning coordinate is within the threshold from Thunder.
- Keypoints with Thunder confidence below the configured threshold are excluded.
- The raw reference clips and JSON output are local benchmark artifacts and should not be committed.
