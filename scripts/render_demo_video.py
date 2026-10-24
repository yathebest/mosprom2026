"""Render a genuine production-cascade demonstration, without database writes.

Inference receives unmodified decoded source pixels. Intermediate source frames
are retained at source FPS; projected boxes are display-only and never vote in
Rules or ScenarioRules. This is an inspectable demo, not an accuracy benchmark.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from app.core import Rules, RunConfig
from app.inference import PeopleTracker, create_backend
from app.preview_motion import PreviewMotion
from app.scenarios import ScenarioRules, annotate_objects, assessment_rows, base_config
from app.video import draw, state_snapshot, transcode

PATTERNS = {"helmet": "no_helmet", "vest": "no_vest", "smoking": "possible_smoking"}
FLAGS = ("helmet", "vest", "mask", "harness", "phone", "fire", "smoke", "smoking", "proximity", "load_posture")


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def save_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    temporary.replace(path)


def scaled_people(people, scale):
    result = []
    for person in people:
        points = person.keypoints.copy() if person.keypoints is not None else None
        if points is not None:
            points[:, :2] *= scale
        result.append(replace(person, box=tuple(v * scale for v in person.box), keypoints=points))
    return result


def summarize_states(states, kind):
    result = []
    for tid, patterns in states.items():
        for key, row in patterns.items():
            if key.split(":", 1)[0] == kind:
                result.append((tid, row))
    return result


def demo_legend(frame, pattern, now, age, measured, states, events):
    """Use ASCII to avoid OpenCV's missing Cyrillic glyphs; labels are factual."""
    height, width = frame.shape[:2]
    rows = summarize_states(states, PATTERNS[pattern])
    open_count = sum(e.get("episode_state") == "open" for e in events)
    lines = [
        f"LINQ SAFETY | {pattern.upper()} | SOURCE {now:.2f}s | NN AGE {age:.2f}s",
        "FRESH NN MEASUREMENT" if measured else "DISPLAY ONLY: projected boxes; no rule votes",
        f"Confirmed open episodes: {open_count} | PPE/object labels = measurements",
    ]
    for tid, row in sorted(rows, key=lambda item: (item[1].get("state") != "violation", item[0]))[:4]:
        current = row.get("observed_state", "unknown")
        lines.append(f"ID {tid}: confirmed={row.get('state', 'unknown')}  latest={current}")
    lines.append("Demo excerpt, not a precision benchmark. Operator review required.")
    line_height = 18 if width < 1000 else 21
    box_height = min(height, 10 + line_height * len(lines))
    shade = frame.copy()
    cv2.rectangle(shade, (0, height - box_height), (width, height), (12, 25, 28), -1)
    cv2.addWeighted(shade, .88, frame, .12, 0, frame)
    for index, line in enumerate(lines):
        y = height - box_height + 17 + index * line_height
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .43 if width < 1000 else .49,
                    (225, 240, 234), 1, cv2.LINE_AA)
    return frame


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Annotated MP4; matching JSON report is written beside it")
    parser.add_argument("--pattern", required=True, choices=PATTERNS)
    parser.add_argument("--sample-fps", type=float, default=5.)
    parser.add_argument("--max-seconds", type=float, default=None, help="Limit source time from the start; no speed changes")
    parser.add_argument("--config-json", type=Path, help="Optional saved RunConfig JSON (or report with config); no threshold tuning is hidden")
    parser.add_argument("--model", type=Path, default=ROOT / "data/models/yolov8n-pose.pt")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    source, output = args.input.resolve(), args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output or output.suffix.lower() != ".mp4":
        raise ValueError("Output must be a separate .mp4 file; the source is never modified")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output}")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError("--max-seconds must be positive")
    values = {"camera": "Offline demo " + args.pattern, "fast_motion": False, "fall_detection": False}
    values.update({flag + "_detection": False for flag in FLAGS})
    if args.config_json:
        saved = json.loads(args.config_json.read_text(encoding="utf-8-sig"))
        values.update(saved.get("config", saved))
    values.update({"sample_fps": args.sample_fps, args.pattern + "_detection": True})
    config = RunConfig(**values)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".json")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists; use --overwrite: {report_path}")
    if report_path == source:
        raise ValueError("Report would overwrite the source")
    run_id = uuid.uuid4().hex[:12]
    evidence = output.parent / (output.stem + "_evidence") / run_id
    evidence.mkdir(parents=True, exist_ok=True)
    temporary_video = output.parent / ("." + output.stem + "." + run_id + ".raw.mp4")
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or not math.isfinite(fps) or not 1 <= fps <= 240 or not count or min(w, h) <= 0:
        cap.release()
        raise ValueError("Unreadable source or unsupported constant-frame-rate timebase")
    stride = max(1, round(fps / config.sample_fps))
    limit = count if args.max_seconds is None else min(count, math.ceil(args.max_seconds * fps))
    scale = min(1., 1280 / max(w, h))
    size = tuple(max(2, int(round(value * scale)) // 2 * 2) for value in (w, h))
    # Drawing coordinates use the actual resize scale; aspect rounding is <=1 px.
    scale = size[0] / w
    with source.open("rb") as source_file:
        source_hash = hashlib.file_digest(source_file, "sha256").hexdigest()
    report = {
        "purpose": "Production cascade demo excerpt; not an independent precision/recall benchmark",
        "source": {"path": str(source), "sha256": source_hash,
                   "width": w, "height": h, "fps": fps, "frames": count, "duration_s": count / fps},
        "output": {"path": str(output), "width": size[0], "height": size[1], "fps": fps,
                   "silent_export": True, "source_frames_interpolated": False},
        "pattern": args.pattern, "config": config.model_dump(), "model": str(args.model.resolve()),
        "sampling": {"requested_fps": config.sample_fps, "effective_fps": fps / stride, "stride": stride,
                     "nn_input": "original decoded source pixels; backend performs its production preprocessing/ROI crops",
                     "between_measurements": "PreviewMotion display-only; never passed back into tracker/rules",
                     "object_overlay_max_age_s": .5},
        "events": [], "lifecycle_updates": [], "measurements": [], "status": "starting",
        "limitations": ["No labelled ground truth or accuracy metric is produced.",
                        "Source end/limit interrupts open episodes; it never proves recovery.",
                        "Projected boxes and cached labels are display-only, not fresh neural measurements."]}
    writer = None
    rules = scenarios = None
    last_measure_time = 0.
    started = time.perf_counter()
    try:
        save_report(report_path, report)
        backend = create_backend(args.model.resolve(), config.confidence, config.fall_detection, config=config)
        report["backend"] = {"name": backend.name, "model_id": getattr(backend, "model_id", None)}
        tracker = PeopleTracker(fps / stride, config.confidence)
        rules, scenarios = Rules(base_config(config)), ScenarioRules(config)
        motion = PreviewMotion()
        writer = cv2.VideoWriter(str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            raise RuntimeError("MP4V output codec is unavailable")
        visual = None
        objects, people, details, latest_states = [], [], {}, {}
        frames, measured_count = 0, 0
        report["status"] = "running"

        def proof(frame, label):
            path = evidence / f"{label}_{uuid.uuid4().hex[:10]}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise IOError(f"Cannot save evidence: {path}")
            return path.relative_to(output.parent).as_posix()

        def apply_updates(updates, annotated=None):
            for original in updates:
                patch = deepcopy(original)
                if patch.get("episode_state") == "closed" and annotated is not None:
                    patch["end_image"] = proof(annotated, "confirmed_end")
                report["lifecycle_updates"].append(patch)
                for event in report["events"]:
                    if event.get("episode_id") == patch.get("episode_id"):
                        event.update(patch)
                        break

        for index in range(limit):
            ok, frame = cap.read()
            if not ok:
                break
            now = index / fps
            fresh = index % stride == 0
            preview = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if (w, h) != size else frame.copy()
            updates, new_events = [], []
            if fresh:
                tick = time.perf_counter()
                # Same production path as app.video.run: scenarios/ROI pose are
                # computed on source pixels before ByteTrack and temporal rules.
                people = backend.predict(frame)
                temporal = getattr(backend, "predict_temporal", None)
                if callable(temporal):
                    temporal(frame, now)
                people = tracker.update(people, frame.shape, now)
                new_events, details = rules.step(people, now, w, h, getattr(backend, "scene_state", None))
                objects = getattr(backend, "objects_state", [])
                extra, active = scenarios.step(people, objects, now, w, h, details)
                new_events.extend(extra)
                updates = list(rules.event_updates) + list(scenarios.event_updates)
                for tid, kinds in active.items():
                    if tid in details:
                        details[tid]["active"].extend(kinds)
                scene_alerts = [key[1] for key, gate in rules.gates.items() if key[0] == 0 and gate.active]
                latest_states = state_snapshot(rules, scenarios)
                last_measure_time = now
                visual = {"time": now, "people": people, "details": deepcopy(details),
                          "scene": deepcopy(getattr(backend, "scene_state", None)), "scene_alerts": scene_alerts,
                          "gray": cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY), "width": w}
                report["measurements"].append({"source_time_s": now, "source_frame": index,
                    "inference_and_rules_ms": round((time.perf_counter() - tick) * 1000, 3),
                    "people": [asdict(p) for p in people], "objects": [asdict(o) for o in objects],
                    "assessments": assessment_rows(people, scenarios.ppe_details), "states": deepcopy(latest_states)})
                display_people = scaled_people(people, scale)
                display_details = details
                # Initialize display flow on this measured frame. Its returned
                # projections are discarded here; measured boxes stay measured.
                motion.project(visual, preview, now)
                measured_count += 1
            else:
                display_people = motion.project(visual, preview, now) if visual else []
                display_details = {tid: {**row, "tracking": "preview flow"} for tid, row in details.items()}
            annotated = draw(preview, display_people, display_details, config, now,
                             visual["scene"] if visual else None, visual["scene_alerts"] if visual else ())
            age = max(0., now - last_measure_time)
            visible_objects = [replace(o, box=tuple(v * scale for v in o.box)) for o in objects] if age <= .5 else []
            display_ppe = scenarios.ppe_details if fresh else {
                tid: {key: {**value, "fresh": False, "reason": "recent_observation"} for key, value in rows.items()}
                for tid, rows in scenarios.ppe_details.items()}
            if age <= 2.5:
                annotated = annotate_objects(annotated, visible_objects, scenarios.ppe_status, display_people, display_ppe)
            for original in new_events:
                event = deepcopy(original)
                event.update(start_image=proof(annotated, "activation"), clip_scope="incident_excerpt")
                event["start_state_confirmation"] = deepcopy(event.get("state_confirmation"))
                event["start_state_from"] = event.get("state_from")
                event["start_state_to"] = event.get("state_to")
                report["events"].append(event)
            apply_updates(updates, annotated)
            for patterns in latest_states.values():
                for state in patterns.values():
                    episode_id = state.get("episode_id")
                    if episode_id:
                        for event in report["events"]:
                            if event.get("episode_id") == episode_id and event.get("episode_state") == "open":
                                event["observed_until_s"] = state.get("last_observed_s")
            annotated = demo_legend(annotated, args.pattern, now, age, fresh, latest_states, report["events"])
            writer.write(annotated)
            frames += 1
            if fresh and measured_count % 10 == 0:
                report.update(decoded_frames=frames, measured_frames=measured_count,
                              current_states=latest_states, current_states_updated_s=last_measure_time,
                              wall_seconds=round(time.perf_counter() - started, 3))
                save_report(report_path, report)
                print(f"source={now:.2f}s measured={measured_count} frames={frames} episodes={len(report['events'])}", flush=True)
        if frames == 0:
            raise RuntimeError("No source frames were decoded; no demonstration was exported")
        report["source_decode_ended_early"] = frames < limit
        reason = "source_limit_reached" if limit < count and frames >= limit else "video_ended"
        before_finish = state_snapshot(rules, scenarios)
        apply_updates(list(rules.finish(last_measure_time, reason)) + list(scenarios.finish(last_measure_time, reason)))
        report.update(status="rendered", decoded_frames=frames, measured_frames=measured_count,
                      current_states=state_snapshot(rules, scenarios), last_observed_states=before_finish,
                      current_states_updated_s=last_measure_time, output_duration_s=frames / fps,
                      wall_seconds=round(time.perf_counter() - started, 3))
        writer.release()
        writer = None
        cap.release()
        report["output"]["h264_transcoded"] = transcode(temporary_video, output)
        if report["output"]["h264_transcoded"]:
            temporary_video.unlink()
        else:
            temporary_video.replace(output)
            report["output"]["codec_fallback"] = "mp4v"
        report["status"] = "completed"
        save_report(report_path, report)
        print(json.dumps({"video": str(output), "report": str(report_path), "events": len(report["events"]),
                          "frames": frames, "measured_frames": measured_count}, ensure_ascii=False), flush=True)
    except BaseException as error:
        if rules is not None and scenarios is not None:
            updates = list(rules.finish(last_measure_time, "processing_failed")) + list(scenarios.finish(last_measure_time, "processing_failed"))
            for patch in updates:
                report["lifecycle_updates"].append(deepcopy(patch))
                for event in report["events"]:
                    if event.get("episode_id") == patch.get("episode_id"):
                        event.update(deepcopy(patch))
            report["current_states"] = state_snapshot(rules, scenarios)
        report.update(status="failed", error=f"{type(error).__name__}: {error}", partial_video=str(temporary_video),
                      wall_seconds=round(time.perf_counter() - started, 3))
        save_report(report_path, report)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
