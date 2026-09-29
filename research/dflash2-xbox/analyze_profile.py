"""Join host monotonic trace, DFlash JSONL, and Xbox request logs."""

import csv
import json
import pathlib
import re
import statistics


ROOT = pathlib.Path(__file__).with_name("profiling-2026-09-27")
MODES = ("real", "xbox", "local")
TRACE = re.compile(r"^DFTRACE stage=(\S+) start_us=(\d+) end_us=(\d+) n=(\d+)$")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("{")]


def percentile(values, percent):
    values = sorted(values)
    pos = (len(values) - 1) * percent / 100
    low = int(pos)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)


def stats(values, total_mean=None):
    result = {
        "mean_ms": statistics.mean(values) / 1000,
        "median_ms": statistics.median(values) / 1000,
        "p95_ms": percentile(values, 95) / 1000,
        "min_ms": min(values) / 1000,
        "max_ms": max(values) / 1000,
    }
    if total_mean is not None:
        result["percent_cycle"] = statistics.mean(values) / total_mean * 100
    return result


def trace_events(mode):
    events = []
    for line in (ROOT / f"dflash-profile-{mode}-server.log").read_text(
            encoding="utf-8", errors="replace").splitlines():
        match = TRACE.match(line)
        if match:
            stage, start, end, n = match.groups()
            events.append({"stage": stage, "start": int(start), "end": int(end), "n": int(n)})
    return events


def xbox_segments():
    requests = [row for row in read_jsonl(ROOT / "xbox-dflash-dedicated.jsonl")
                if row.get("phase") == "request"]
    marks = [i for i, row in enumerate(requests) if row.get("op") == 6]
    if len(marks) < 3:
        raise RuntimeError("missing real/mock/local session markers on Xbox")
    marks = marks[-3:]
    return {
        mode: requests[marks[i] + 1: marks[i + 1] if i + 1 < 3 else len(requests)]
        for i, mode in enumerate(MODES)
    }


def main():
    rows_by_mode = {mode: read_jsonl(ROOT / f"dflash-profile-{mode}.jsonl") for mode in MODES}
    summaries = {mode: json.loads((ROOT / f"dflash-profile-{mode}-summary.json").read_text(encoding="utf-8"))
                 for mode in MODES}
    trajectories = [[(r["target_position"], r["n_proposed"], r["n_accepted"])
                     for r in rows_by_mode[mode]] for mode in MODES]
    if not (trajectories[0] == trajectories[1] == trajectories[2]):
        raise RuntimeError("A/B/C draft or acceptance trajectories differ")
    if len({summary["content_sha256"] for summary in summaries.values()}) != 1:
        raise RuntimeError("A/B/C output text differs")
    segments = xbox_segments()
    by_mode = {}
    csv_rows = []
    for mode in MODES:
        rows = rows_by_mode[mode]
        events = trace_events(mode)
        xbox_drafts = {row["cycleId"]: row for row in segments[mode] if row["op"] == 4}
        if mode != "local" and len(xbox_drafts) != len(rows):
            raise RuntimeError(f"{mode}: {len(xbox_drafts)} Xbox drafts, {len(rows)} host drafts")
        measures = []
        # The first row includes prompt prefill. The final row has no following DRAFT start.
        for i in range(1, len(rows) - 1):
            row = rows[i]
            start = row["draft_start_tick_us"]
            end = rows[i + 1]["draft_start_tick_us"]
            tr = {}
            for event in events:
                if start <= event["start"] < end:
                    tr[event["stage"]] = tr.get(event["stage"], 0) + event["end"] - event["start"]
            required = ("target_submit", "target_sync", "sample_accept", "commit_prompt",
                        "kv_remove", "emit_tokens", "batch_from_sample")
            if any(stage not in tr for stage in required):
                raise RuntimeError(f"{mode} cycle {i}: missing {set(required) - set(tr)}")
            xbox = xbox_drafts.get(row["cycle_id"], {})
            draft_post = xbox.get("postUs", 0)
            draft_send = xbox.get("responseUs", 0)
            stage = {
                "request_prepare": row["prepare_request_us"],
                "draft_serialize": row["serialize_us"],
                "draft_rpc_transport": row["draft_rpc_us"] - row["xbox_receive_us"]
                    - row["xbox_prepare_us"] - row["xbox_compute_us"] - draft_post - draft_send,
                "xbox_receive": row["xbox_receive_us"],
                "xbox_prepare": row["xbox_prepare_us"],
                "xbox_compute": row["xbox_compute_us"],
                "xbox_postprocess": draft_post,
                "xbox_send": draft_send,
                "host_response_copy": row["selector_copy_us"],
                "host_selector": row["selector_us"],
                "host_proposals": row["proposal_us"],
                "batch_prepare": tr["batch_from_sample"],
                "target_submit": tr["target_submit"],
                "target_sync": tr["target_sync"],
                "sample": tr["sample_accept"],
                "commit": tr["commit_prompt"],
                "kv_remove": tr["kv_remove"],
                "emit_tokens": tr["emit_tokens"],
                "hidden_capture": row["capture_hidden_us"],
                "sync_serialize": row["sync_serialize_us"],
                "sync_rpc_transport": row["sync_rpc_us"]
                    - row["sync_xbox_prepare_us"] - row["sync_xbox_compute_us"],
                "sync_xbox_prepare": row["sync_xbox_prepare_us"],
                "sync_xbox_compute": row["sync_xbox_compute_us"],
                "trim_rpc": row["trim_rpc_us"],
                "rewind_rpc": row["rewind_rpc_us"],
            }
            stage["other_host_wait"] = end - start - sum(stage.values())
            if min(stage.values()) < -100:
                raise RuntimeError(f"{mode} cycle {i}: negative stage {stage}")
            stage.update({
                "mode": mode, "cycle_index": i, "position": row["target_position"],
                "proposed": row["n_proposed"], "accepted": row["n_accepted"],
                "full_cycle_us": end - start,
                "draft_rpc_us": row["draft_rpc_us"],
                "sync_rpc_us": row["sync_rpc_us"],
                "target_verify_us": tr["target_submit"] + tr["target_sync"],
                "non_draft_us": end - start - row["draft_rpc_us"],
                "xbox_compute_total_us": row["xbox_compute_us"] + row["sync_xbox_compute_us"],
                "rpc_transport_total_us": stage["draft_rpc_transport"] + stage["sync_rpc_transport"]
                    + row["trim_rpc_us"] + row["rewind_rpc_us"],
                "control_rpc_us": row["trim_rpc_us"] + row["rewind_rpc_us"],
                "commit_kv_us": tr["commit_prompt"] + tr["kv_remove"],
                "draft_host_send_us": row["send_us"],
                "draft_host_payload_read_us": row["payload_read_us"],
                "draft_header_wait_including_compute_us": row["header_wait_us"],
            })
            measures.append(stage)
            csv_rows.append(stage)
        total_mean = statistics.mean(row["full_cycle_us"] for row in measures)
        phases = {name: stats([row[name] for row in measures], total_mean)
                  for name in measures[0] if name not in
                  ("mode", "cycle_index", "position", "proposed", "accepted")}
        all_rows = rows
        by_mode[mode] = {
            "request": summaries[mode],
            "cycles_all": len(all_rows),
            "cycles_steady": len(measures),
            "proposed": sum(row["n_proposed"] for row in all_rows),
            "accepted": sum(row["n_accepted"] for row in all_rows),
            "rpc_failures": sum(row["rpc_failed"] for row in all_rows),
            "rewind_calls_steady": sum(row["rewind_rpc"] > 0 for row in measures),
            "xbox_trim_payload_receive": stats([row["receiveUs"] for row in segments[mode]
                                                 if row["op"] == 5]) if mode != "local" else None,
            "phases": phases,
        }
    with (ROOT / "cycles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    derived = {
        "tokens_per_cycle": 512 / len(rows_by_mode["real"]),
        "baseline_tps_previous": 34.99,
        "break_even_cycle_ms": (512 / len(rows_by_mode["real"])) / 34.99 * 1000,
        "real_minus_mock_xbox_ms_per_cycle":
            (summaries["real"]["predicted_ms"] - summaries["xbox"]["predicted_ms"])
            / len(rows_by_mode["real"]),
        "mock_xbox_minus_local_ms_per_cycle":
            (summaries["xbox"]["predicted_ms"] - summaries["local"]["predicted_ms"])
            / len(rows_by_mode["real"]),
        "same_trajectory_and_text": True,
    }
    report = {"modes": by_mode, "derived": derived}
    (ROOT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": {mode: {
        "tps": data["request"]["tps"],
        "cycles": data["cycles_all"],
        "proposed": data["proposed"],
        "accepted": data["accepted"],
        "full_cycle_ms": data["phases"]["full_cycle_us"]["mean_ms"],
        "top_phases": sorted(((name, s["mean_ms"]) for name, s in data["phases"].items()
                              if name not in ("full_cycle_us", "draft_rpc_us", "sync_rpc_us", "target_verify_us",
                                              "non_draft_us", "xbox_compute_total_us", "rpc_transport_total_us",
                                              "control_rpc_us", "commit_kv_us", "draft_host_send_us",
                                              "draft_host_payload_read_us", "draft_header_wait_including_compute_us")),
                             key=lambda item: -item[1])[:8],
    } for mode, data in by_mode.items()}, "derived": derived}, indent=2))


if __name__ == "__main__":
    main()
