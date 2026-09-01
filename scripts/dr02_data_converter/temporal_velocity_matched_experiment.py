#!/usr/bin/env python3
"""Reproducible matched DR02 temporal/velocity experiment and independent audit.

The tool deliberately keeps all generated trajectories below ``output_data``
(which is ignored).  Small provenance, manifests, resolved overlays, and the
per-motion metric table are copied to ``docs/data`` by ``archive``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from robot_retarget import RobotRetarget  # noqa: E402

FPS = 120.0
FOUR_EE = {
    "left_hand": ("left_fore_arm", "left_hand_link"),
    "right_hand": ("right_fore_arm", "right_hand_link"),
    # These are fixed foot markers.  Their orientations inherit the ankle
    # frame, so their rotational score is a marker-frame orientation score.
    "left_foot": ("left_foot_end_link", "left_foot_end_link"),
    "right_foot": ("right_foot_end_link", "right_foot_end_link"),
}
UPPER_LIMB = {
    "left_shoulder": ("left_shoulder", "left_shoulder_x_link"),
    "right_shoulder": ("right_shoulder", "right_shoulder_x_link"),
    "left_elbow": ("left_arm", "left_elbow_link"),
    "right_elbow": ("right_arm", "right_elbow_link"),
    "left_hand": ("left_fore_arm", "left_hand_link"),
    "right_hand": ("right_fore_arm", "right_hand_link"),
}
CANDIDATES = {
    "B0": {"temporal": {"enabled": False}, "velocity": {"enabled": False}},
    "B1": {"temporal": {"enabled": True, "cost": 6.5, "joint_name_substrings": ["shoulder", "elbow", "wrist"]}, "velocity": {"enabled": True, "joint_velocity_limits": {"shoulder": 4.0, "elbow": 4.0}}},
    "B2": {"temporal": {"enabled": True, "cost": 10.0, "joint_name_substrings": ["shoulder", "elbow", "wrist"]}, "velocity": {"enabled": True, "joint_velocity_limits": {"shoulder": 8.0, "elbow": 8.0}}},
    "B3": {"temporal": {"enabled": True, "cost": 10.0, "joint_name_substrings": ["shoulder", "elbow", "wrist"]}, "velocity": {"enabled": True, "joint_velocity_limits": {"shoulder": 8.0, "elbow": 6.0}}},
    "B4": {"temporal": {"enabled": True, "cost": 10.0, "joint_name_substrings": ["shoulder", "elbow", "wrist"]}, "velocity": {"enabled": True, "joint_velocity_limits": {"shoulder": 8.0, "elbow": 10.0}}},
    "B5": {"temporal": {"enabled": True, "cost": 12.0, "joint_name_substrings": ["shoulder", "elbow", "wrist"]}, "velocity": {"enabled": True, "joint_velocity_limits": {"shoulder": 8.0, "elbow": 8.0}}},
}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def stable_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def experiment_dir(args) -> Path:
    return Path(args.work_dir).resolve() if args.work_dir else ROOT / "output_data/experiments/dr02_temporal_velocity_matched"

def manifest_from_command_md() -> list[dict]:
    text = (ROOT / "command.md").read_text(encoding="utf-8")
    pairs = re.findall(r"# \[(\d{2})\][\s\S]*?--source-csv\s+['\\\"]?([^'\"\\\n ]+\.csv)", text)
    seen, rows = set(), []
    for number, source in pairs:
        if number in seen or int(number) > 25:
            continue
        seen.add(number)
        path = ROOT / source
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"id": int(number), "source": source, "source_sha256": sha(path)})
    if len(rows) != 25 or len({x["source"] for x in rows}) != 25:
        raise RuntimeError(f"expected 25 unique QA sources, found {len(rows)}")
    return sorted(rows, key=lambda x: x["id"])

def load_base() -> dict:
    return yaml.safe_load((ROOT / "config/robot/DR02_temporal_constraints_test.yaml").read_text())

def resolved_config(candidate: str) -> dict:
    cfg = load_base()
    # Experiment-only overlay: never use checked-in active values as a candidate source.
    cfg["render_debug"] = False
    cfg["verbose"] = False
    cfg["two_stage_optimization"] = {"enabled": False}
    cfg["temporal_continuity_regularization"] = CANDIDATES[candidate]["temporal"]
    cfg["inter_frame_velocity_limit"] = CANDIDATES[candidate]["velocity"]
    return cfg

def prepare(args) -> None:
    wd = experiment_dir(args); (wd / "pkl").mkdir(parents=True, exist_ok=True)
    rows = manifest_from_command_md()
    existing_path = wd / "manifest.json"
    existing = {x["id"]: x for x in json.loads(existing_path.read_text())} if existing_path.exists() else {}
    for row in rows:
        row.update({k: v for k, v in existing.get(row["id"], {}).items() if k in {"pkl", "pkl_sha256"}})
        stem = f"{row['id']:02d}_{Path(row['source']).stem}"
        discovered = wd / "pkl" / f"{stem}.pkl"
        if discovered.exists():
            row["pkl"] = str(discovered.relative_to(wd)); row["pkl_sha256"] = sha(discovered)
    (wd / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        stem = f"{row['id']:02d}_{Path(row['source']).stem}"
        normalized = wd / "normalized" / f"{stem}.csv"; normalized.parent.mkdir(parents=True, exist_ok=True)
        pkl = wd / "pkl" / f"{stem}.pkl"
        if not normalized.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts/dr02_data_converter/motiondecode_adapter.py"), "--input", str(ROOT / row["source"]), "--output", str(normalized)], check=True)
        if not pkl.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts/robot_replay.py"), "--no-viewer", "--motion-file", str(normalized), "--source-robot-config", str(ROOT / "config/robot/g1.yaml"), "--target-robot-config", str(ROOT / "config/robot/DR02.yaml"), "--fps", str(FPS), "--output-path", str(pkl)], check=True)
        row["pkl"] = str(pkl.relative_to(wd)); row["pkl_sha256"] = sha(pkl)
    (wd / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"prepared {len(rows)} frozen PKLs in {wd}")

def make_retargeter(cfg: dict, pkl: Path) -> RobotRetarget:
    return RobotRetarget(model_path=str(ROOT / cfg["robot_xml_path"]), keypoint_path=str(pkl), ik_match_table=cfg["ik_match_table"], solver="daqp", verbose=False, render_debug=False, joints_limit_offset_degrees=cfg.get("joints_limit_offset_degrees"), contact_body_names=cfg.get("contact_links"), contact_position_cost=cfg.get("contact_pos_fixed_factor", 10.0), temporal_continuity_regularization=cfg.get("temporal_continuity_regularization"), inter_frame_velocity_limit=cfg.get("inter_frame_velocity_limit"), two_stage_optimization={"enabled": False})

def run(args) -> None:
    wd = experiment_dir(args); rows = json.loads((wd / "manifest.json").read_text())
    candidates = [args.candidate] if args.candidate else list(CANDIDATES)
    for candidate in candidates:
        cfg = resolved_config(candidate); cdir = wd / "configs"; cdir.mkdir(parents=True, exist_ok=True)
        (cdir / f"{candidate}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
        config_hash = stable_sha(cfg)
        status = []
        for row in rows:
            pkl = wd / row["pkl"]; out = wd / "qpos" / candidate / f"{row['id']:02d}_{Path(row['source']).stem}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
            start = time.time()
            if out.exists() and not args.force:
                status.append({"id": row["id"], "status": "skipped", "seconds": 0, "qpos_sha256": sha(out), "config_sha256": config_hash}); continue
            try:
                r = make_retargeter(cfg, pkl); r.retarget(); r.save_results_as_csv(str(out))
                status.append({"id": row["id"], "status": "success", "seconds": time.time()-start, "qpos_sha256": sha(out), "config_sha256": config_hash, "frames": r.num_frames})
            except Exception as exc:
                status.append({"id": row["id"], "status": "failed", "seconds": time.time()-start, "error": repr(exc), "config_sha256": config_hash})
                (wd / "status" / candidate).parent.mkdir(parents=True, exist_ok=True)
                (wd / "status" / f"{candidate}.json").write_text(json.dumps(status, indent=2)+"\n")
                raise
        (wd / "status").mkdir(parents=True, exist_ok=True)
        (wd / "status" / f"{candidate}.json").write_text(json.dumps(status, indent=2)+"\n")
        print(candidate, len(status), "completed")

def read_qpos(path: Path, nq: int) -> np.ndarray:
    x = np.atleast_2d(np.loadtxt(path, delimiter=",", dtype=float))
    if x.shape[1] != nq or not np.isfinite(x).all(): raise ValueError(f"invalid qpos CSV {path}")
    x[:, 3:7] = x[:, [6,3,4,5]]
    n = np.linalg.norm(x[:,3:7], axis=1)
    if np.any(n < 1e-12): raise ValueError(f"zero quaternion {path}")
    x[:,3:7] /= n[:,None]
    return x

def geodesic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(2*np.arccos(np.clip(np.abs(np.sum(a*b, axis=-1)), -1, 1)))

def longest_run(flags: np.ndarray) -> int:
    best = run = 0
    for x in flags:
        run = run+1 if x else 0; best = max(best, run)
    return best

def selected_joint_ids(model, needle: str) -> list[int]:
    return [j for j in range(model.njnt) if needle in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or "") and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]

def evaluate_one(cfg: dict, pkl: Path, qpos_path: Path, candidate: str, motion: dict) -> tuple[dict, dict]:
    r = make_retargeter(cfg, pkl); q = read_qpos(qpos_path, r.model.nq); T = len(q)
    if T != r.num_frames: raise ValueError("frame mismatch")
    groups = {"four_ee": FOUR_EE, "upper_limb": UPPER_LIMB}; per = {}
    for group, pairs in groups.items():
        for label, (src, body) in pairs.items():
            if src not in r.keypoint_name_to_idx or mujoco.mj_name2id(r.model,mujoco.mjtObj.mjOBJ_BODY,body) < 0: raise ValueError(f"invalid explicit pair {src}->{body}")
            pos=[]; rot=[]
            for f, qq in enumerate(q):
                r.configuration.update(qq); bid=mujoco.mj_name2id(r.model,mujoco.mjtObj.mjOBJ_BODY,body); idx=r.keypoint_name_to_idx[src]
                pos.append(np.linalg.norm(r.configuration.data.xpos[bid]-r.keypoints_pos[f,idx])*100); rot.append(geodesic(r.configuration.data.xquat[bid],r.keypoints_quat[f,idx]))
            if len(pos) != T: raise AssertionError("sample-count failure")
            per[f"{group}_{label}_pos_p95_cm"] = float(np.percentile(pos,95,method="linear")); per[f"{group}_{label}_rot_p95_deg"] = float(np.percentile(rot,95,method="linear")); per[f"{group}_{label}_arc_m"] = float(np.linalg.norm(np.diff(np.array([r.configuration.data.xpos[mujoco.mj_name2id(r.model,mujoco.mjtObj.mjOBJ_BODY,body)] for _ in []]),axis=0)) if False else 0)
    # arcs recomputed in one FK sweep to keep semantics explicit.
    positions={name:[] for name in FOUR_EE}
    for qq in q:
        r.configuration.update(qq)
        for name, (_,body) in FOUR_EE.items(): positions[name].append(r.configuration.data.xpos[mujoco.mj_name2id(r.model,mujoco.mjtObj.mjOBJ_BODY,body)].copy())
    for name,x in positions.items(): per[f"four_ee_{name}_arc_m"] = float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum())
    for group, pairs in groups.items():
        per[f"{group}_pos_p95_cm"] = float(np.mean([per[f"{group}_{x}_pos_p95_cm"] for x in pairs])); per[f"{group}_rot_p95_deg"] = float(np.mean([per[f"{group}_{x}_rot_p95_deg"] for x in pairs]))
    shoulder=selected_joint_ids(r.model,"shoulder"); elbow=selected_joint_ids(r.model,"elbow")
    for name, ids in (("shoulder",shoulder),("elbow",elbow)):
        flags_near=[]; flags_hard=[]; jointnear=jointhard=0
        for qq in q:
            vals=[]
            for j in ids:
                d=min(abs(qq[r.model.jnt_qposadr[j]]-r.model.jnt_range[j,0]),abs(qq[r.model.jnt_qposadr[j]]-r.model.jnt_range[j,1])); vals.append(d)
            vals=np.asarray(vals); jointnear+=np.count_nonzero(vals<=np.deg2rad(5)); jointhard+=np.count_nonzero(vals<=.001); flags_near.append(np.any(vals<=np.deg2rad(5))); flags_hard.append(np.any(vals<=.001))
        per.update({f"{name}_near_joint_pct":100*jointnear/(T*len(ids)),f"{name}_hard_joint_pct":100*jointhard/(T*len(ids)),f"{name}_near_frame_pct":100*np.mean(flags_near),f"{name}_hard_frame_pct":100*np.mean(flags_hard),f"{name}_near_longest_run":longest_run(flags_near),f"{name}_hard_longest_run":longest_run(flags_hard)})
    se=shoulder+elbow; vel=[]
    for a,b in zip(q[:-1],q[1:]):
        d=np.empty(r.model.nv); mujoco.mj_differentiatePos(r.model,d,1/FPS,a,b); vel.append(d)
    vel=np.asarray(vel); vse=np.abs(vel[:,[r.model.jnt_dofadr[j] for j in se]])
    per["qdot_p95_rad_s"] = float(np.percentile(vse,95,method="linear")); per["qddot_p95_rad_s2"] = float(np.percentile(np.abs(np.diff(vel[:,[r.model.jnt_dofadr[j] for j in se]],axis=0)*FPS),95,method="linear"))
    if candidate == "B0": per["edge_pct"] = None
    else:
        limits=[]
        for j in se:
            name=mujoco.mj_id2name(r.model,mujoco.mjtObj.mjOBJ_JOINT,j); limits.append(cfg["inter_frame_velocity_limit"]["joint_velocity_limits"]["shoulder" if "shoulder" in name else "elbow"])
        per["edge_pct"] = float(100*np.mean(vse >= .99*np.asarray(limits)))
    # Explicit shoulder/elbow/wrist range, normalized against B0 later.
    arm_ids=selected_joint_ids(r.model,"shoulder")+selected_joint_ids(r.model,"elbow")+selected_joint_ids(r.model,"wrist")
    per["arm_range_mean_rad"] = float(np.mean([np.ptp(q[:,r.model.jnt_qposadr[j]]) for j in arm_ids]))
    out={"candidate":candidate,"motion_id":motion["id"],"source":motion["source"],"frames":T,**per}
    return out, {"four_ee":{k:len(FOUR_EE)*T for k in ["samples"]},"upper_limb":{k:len(UPPER_LIMB)*T for k in ["samples"]}}

def evaluate(args) -> None:
    wd=experiment_dir(args); rows=json.loads((wd/"manifest.json").read_text()); metrics=[]; audit=[]
    for candidate in CANDIDATES:
        cfg=resolved_config(candidate)
        for motion in rows:
            qpath=wd/"qpos"/candidate/f"{motion['id']:02d}_{Path(motion['source']).stem}.csv"
            row, proof=evaluate_one(cfg,wd/motion["pkl"],qpath,candidate,motion); metrics.append(row); audit.append({"candidate":candidate,"motion_id":motion["id"],**proof})
    # motion-relative fidelity ratios use matched B0 only.
    b0={(x["motion_id"]):x for x in metrics if x["candidate"]=="B0"}
    for x in metrics:
        base=b0[x["motion_id"]]; x["four_ee_arc_over_B0"] = float(np.mean([x[f"four_ee_{n}_arc_m"]/base[f"four_ee_{n}_arc_m"] for n in FOUR_EE])); x["arm_range_over_B0"] = x["arm_range_mean_rad"]/base["arm_range_mean_rad"]
    out=wd/"metrics.csv"; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=sorted({k for x in metrics for k in x})); w.writeheader(); w.writerows(metrics)
    (wd/"metric_sample_counts.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(out)

def independently_recompute_fidelity(cfg: dict, pkl: Path, qpos_path: Path) -> dict:
    """Recompute the explicit tracking metrics in a separate FK pass.

    This deliberately does not call ``evaluate_one`` nor reuse its error
    collections.  It is the guard against reintroducing the former
    prefix-based ``left_* -> left_arm`` aggregation bug.
    """
    r = make_retargeter(cfg, pkl)
    q = read_qpos(qpos_path, r.model.nq)
    pairs = {"four_ee": FOUR_EE, "upper_limb": UPPER_LIMB}
    values = {group: {name: {"pos": [], "rot": []} for name in mapping}
              for group, mapping in pairs.items()}
    arcs = {name: [] for name in FOUR_EE}
    for frame, qq in enumerate(q):
        r.configuration.update(qq)
        for group, mapping in pairs.items():
            for name, (source, body) in mapping.items():
                keypoint_index = r.keypoint_name_to_idx[source]
                body_id = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_BODY, body)
                body_pos = r.configuration.data.xpos[body_id].copy()
                values[group][name]["pos"].append(
                    float(np.linalg.norm(body_pos - r.keypoints_pos[frame, keypoint_index]) * 100.0)
                )
                values[group][name]["rot"].append(
                    float(geodesic(r.configuration.data.xquat[body_id], r.keypoints_quat[frame, keypoint_index]))
                )
                if group == "four_ee":
                    arcs[name].append(body_pos)
    out = {}
    for group, mapping in pairs.items():
        for name in mapping:
            out[f"{group}_{name}_pos_p95_cm"] = float(np.percentile(values[group][name]["pos"], 95, method="linear"))
            out[f"{group}_{name}_rot_p95_deg"] = float(np.percentile(values[group][name]["rot"], 95, method="linear"))
        out[f"{group}_pos_p95_cm"] = float(np.mean([out[f"{group}_{name}_pos_p95_cm"] for name in mapping]))
        out[f"{group}_rot_p95_deg"] = float(np.mean([out[f"{group}_{name}_rot_p95_deg"] for name in mapping]))
    for name, x in arcs.items():
        out[f"four_ee_{name}_arc_m"] = float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())
    return out

def audit(args) -> None:
    """Independently audit all saved qpos/PKL pairs against metrics.csv."""
    wd = experiment_dir(args)
    rows = json.loads((wd / "manifest.json").read_text())
    with (wd / "metrics.csv").open(newline="") as f:
        reported = {(x["candidate"], int(x["motion_id"])): x for x in csv.DictReader(f)}
    checks, maximum = [], 0.0
    for candidate in CANDIDATES:
        cfg = resolved_config(candidate)
        for motion in rows:
            qpath = wd / "qpos" / candidate / f"{motion['id']:02d}_{Path(motion['source']).stem}.csv"
            independent = independently_recompute_fidelity(cfg, wd / motion["pkl"], qpath)
            prior = reported[(candidate, motion["id"])]
            differences = {name: abs(value - float(prior[name])) for name, value in independent.items()}
            maximum = max(maximum, *differences.values())
            checks.append({"candidate": candidate, "motion_id": motion["id"],
                           "four_ee_samples": 4 * int(prior["frames"]),
                           "upper_limb_samples": 6 * int(prior["frames"]),
                           "max_abs_difference": max(differences.values()),
                           "passed": max(differences.values()) <= 1e-9})
    out = wd / "independent_audit.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0])); writer.writeheader(); writer.writerows(checks)
    summary = {"rows": len(checks), "passed_rows": sum(x["passed"] for x in checks),
               "max_abs_difference": maximum, "tolerance": 1e-9}
    (wd / "independent_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["passed_rows"] != len(checks):
        raise RuntimeError(f"independent audit failed: {summary}")
    print(out)

def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["prepare","run","evaluate","audit"]); p.add_argument("--work-dir"); p.add_argument("--candidate",choices=list(CANDIDATES)); p.add_argument("--force",action="store_true"); a=p.parse_args(); {"prepare":prepare,"run":run,"evaluate":evaluate,"audit":audit}[a.command](a)
if __name__ == "__main__": main()
