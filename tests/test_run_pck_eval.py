from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from benchmark.run_pck_eval import (
    ClipPckSummary,
    aggregate_summary,
    evaluate_keypoints,
    fmt,
    normalized_distances,
    pck_distance_threshold,
    summarize_frames,
    write_markdown,
)


def keypoints_with_conf(confidence: float = 0.9) -> np.ndarray:
    points = np.zeros((17, 3), dtype=np.float32)
    points[:, 0] = 0.5
    points[:, 1] = 0.5
    points[:, 2] = confidence
    return points


def test_pck_distance_threshold_uses_normalized_diagonal() -> None:
    assert pck_distance_threshold(0.05) == math.sqrt(2.0) * 0.05


def test_normalized_distances_requires_movenet_shape() -> None:
    reference = keypoints_with_conf()
    candidate = keypoints_with_conf()
    candidate[0, 0] += 0.03
    candidate[0, 1] += 0.04

    distances = normalized_distances(reference, candidate)

    assert distances.shape == (17,)
    assert round(float(distances[0]), 6) == 0.05


def test_evaluate_keypoints_counts_only_visible_reference_points() -> None:
    reference = keypoints_with_conf(0.9)
    candidate = keypoints_with_conf(0.9)
    candidate[0, 0] += 0.01
    candidate[1, 0] += 0.2
    reference[2, 2] = 0.1
    candidate[2, 0] += 0.2

    result = evaluate_keypoints(
        reference,
        candidate,
        threshold_ratio=0.05,
        min_reference_confidence=0.3,
    )

    assert result.eligible_keypoints == 16
    assert result.correct_keypoints == 15
    assert len(result.distances) == 16
    assert result.per_keypoint[2][1] is False
    assert result.per_keypoint[2][2] is False
    assert result.per_keypoint[2][3] is None


def test_summarize_frames_computes_clip_pck() -> None:
    reference = keypoints_with_conf(0.9)
    candidate = keypoints_with_conf(0.9)
    good = evaluate_keypoints(
        reference,
        candidate,
        threshold_ratio=0.05,
        min_reference_confidence=0.3,
    )
    bad_candidate = keypoints_with_conf(0.9)
    bad_candidate[:, 0] += 0.2
    bad = evaluate_keypoints(
        reference,
        bad_candidate,
        threshold_ratio=0.05,
        min_reference_confidence=0.3,
    )

    summary = summarize_frames(
        clip="clip.mp4",
        total_frames=10,
        evaluated_frames=2,
        source_fps=30.0,
        frames=[good, bad],
    )

    assert summary.clip == "clip.mp4"
    assert summary.eligible_keypoints == 34
    assert summary.correct_keypoints == 17
    assert summary.pck == 0.5
    assert summary.per_keypoint["nose"]["eligible"] == 2
    assert summary.per_keypoint["nose"]["correct"] == 1


def test_aggregate_summary_combines_clips() -> None:
    clips = [
        ClipPckSummary(
            clip="a.mp4",
            total_frames=10,
            evaluated_frames=2,
            source_fps=30.0,
            duration_s=0.3,
            eligible_keypoints=10,
            correct_keypoints=8,
            pck=0.8,
            mean_distance=0.01,
            median_distance=0.01,
            reference_avg_confidence=0.9,
            candidate_avg_confidence=0.8,
            per_keypoint={},
        ),
        ClipPckSummary(
            clip="b.mp4",
            total_frames=20,
            evaluated_frames=4,
            source_fps=30.0,
            duration_s=0.6,
            eligible_keypoints=30,
            correct_keypoints=27,
            pck=0.9,
            mean_distance=0.02,
            median_distance=0.02,
            reference_avg_confidence=0.7,
            candidate_avg_confidence=0.6,
            per_keypoint={},
        ),
    ]

    aggregate = aggregate_summary(clips)

    assert aggregate["total_frames"] == 30
    assert aggregate["evaluated_frames"] == 6
    assert aggregate["eligible_keypoints"] == 40
    assert aggregate["correct_keypoints"] == 35
    assert aggregate["pck"] == 0.875


def test_write_markdown_includes_summary_table(tmp_path: Path) -> None:
    summary = ClipPckSummary(
        clip="clip.mp4",
        total_frames=10,
        evaluated_frames=2,
        source_fps=30.0,
        duration_s=0.3,
        eligible_keypoints=10,
        correct_keypoints=9,
        pck=0.9,
        mean_distance=0.01,
        median_distance=0.01,
        reference_avg_confidence=0.8,
        candidate_avg_confidence=0.7,
        per_keypoint={},
    )
    output = tmp_path / "pck.md"

    write_markdown(
        output,
        clips=[summary],
        threshold_ratio=0.05,
        min_reference_confidence=0.3,
        json_output=Path("metrics/pck.json"),
    )

    text = output.read_text(encoding="utf-8")
    assert "PCK@0.05" in text
    assert "clip.mp4" in text
    assert "**aggregate**" in text


def test_fmt_handles_none() -> None:
    assert fmt(None) == ""
    assert fmt(0.123456, 3) == "0.123"
