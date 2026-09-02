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
import shutil
from datetime import datetime, timezone
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
    return Path(args.work_dir).resolve() if args.work_dir else ROOT / "output_data/experiments/dr02_temporal_velocity_matched_v2"

def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()

def git_file_sha(commit: str, relative: str) -> str:
    """Hash a file as it existed in a committed generation snapshot."""
    data = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=ROOT,
                          check=True, capture_output=True).stdout
    return hashlib.sha256(data).hexdigest()

def cache_metadata(cfg: dict, candidate: str, row: dict, pkl: Path, qpos: Path, frames: int | None = None) -> dict:
    """Immutable provenance attached to one generated qpos file."""
    model = ROOT / cfg["robot_xml_path"]
    result = {
        "code_git_sha": git_sha(), "runner_sha256": sha(Path(__file__)),
        "solver_sha256": sha(ROOT / "scripts/robot_retarget.py"),
        "model_sha256": sha(model), "source_sha256": row["source_sha256"],
        "shared_pkl_sha256": sha(pkl), "resolved_config_sha256": stable_sha(cfg),
        "candidate": candidate, "motion_id": row["id"], "fps": FPS,
        "nq": None, "frames": frames, "qpos_sha256": sha(qpos),
        "success_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if qpos.exists():
        data = np.atleast_2d(np.loadtxt(qpos, delimiter=",", dtype=float))
        result["nq"] = int(data.shape[1]); result["frames"] = int(data.shape[0])
    return result

def metadata_path(qpos: Path) -> Path:
    return qpos.with_suffix(qpos.suffix + ".metadata.json")

def validate_cache(cfg: dict, candidate: str, row: dict, pkl: Path, qpos: Path) -> tuple[bool, str]:
    """Strict validation used only before reusing a qpos as a *run* cache."""
    meta_path = metadata_path(qpos)
    if not qpos.exists() or not meta_path.exists(): return False, "missing qpos or metadata"
    try: recorded = json.loads(meta_path.read_text())
    except Exception as exc: return False, f"unreadable metadata: {exc!r}"
    expected = cache_metadata(cfg, candidate, row, pkl, qpos)
    for key in ("code_git_sha", "runner_sha256", "solver_sha256", "model_sha256", "source_sha256", "shared_pkl_sha256", "resolved_config_sha256", "candidate", "motion_id", "fps", "nq", "frames", "qpos_sha256"):
        if recorded.get(key) != expected.get(key): return False, f"metadata mismatch: {key}"
    return True, "validated_cache"

def validate_snapshot(cfg: dict, candidate: str, row: dict, pkl: Path, qpos: Path) -> tuple[bool, str, dict]:
    """Read-only validation for a historical generated qpos snapshot.

    Unlike ``validate_cache`` this never asserts that the current runner made
    the trajectory.  It proves the recorded generation commit still exists,
    verifies all input/model/config/qpos hashes, and recovers the solver hash
    from that commit for old metadata that predates this field.
    """
    mp = metadata_path(qpos)
    if not qpos.exists() or not mp.exists(): return False, "missing qpos or metadata", {}
    try: meta = json.loads(mp.read_text())
    except Exception as exc: return False, f"unreadable metadata: {exc!r}", {}
    generation = meta.get("code_git_sha")
    if not generation or subprocess.run(["git", "cat-file", "-e", f"{generation}^{{commit}}"], cwd=ROOT).returncode:
        return False, "recorded generation commit is unavailable", meta
    expected = {"model_sha256": sha(ROOT / cfg["robot_xml_path"]),
                "source_sha256": row["source_sha256"], "shared_pkl_sha256": sha(pkl),
                "resolved_config_sha256": stable_sha(cfg), "candidate": candidate,
                "motion_id": row["id"], "fps": FPS, "qpos_sha256": sha(qpos)}
    q = np.atleast_2d(np.loadtxt(qpos, delimiter=",", dtype=float))
    expected.update({"nq": int(q.shape[1]), "frames": int(q.shape[0])})
    for key, value in expected.items():
        if meta.get(key) != value: return False, f"snapshot mismatch: {key}", meta
    try:
        solver_hash = meta.get("solver_sha256") or git_file_sha(generation, "scripts/robot_retarget.py")
        runner_hash = git_file_sha(generation, "scripts/dr02_data_converter/temporal_velocity_matched_experiment.py")
    except subprocess.CalledProcessError: return False, "generation snapshot lacks runner/solver", meta
    if meta.get("runner_sha256") != runner_hash: return False, "snapshot mismatch: runner_sha256", meta
    meta["generation_solver_sha256"] = solver_hash
    meta["generation_git_sha"] = generation
    return True, "validated_snapshot", meta

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
            valid, reason = validate_cache(cfg, candidate, row, pkl, out)
            if valid and not args.force:
                status.append({"id": row["id"], "status": "validated_cache", "seconds": 0,
                               "qpos_sha256": sha(out), "config_sha256": config_hash,
                               "metadata": str(metadata_path(out).relative_to(wd))}); continue
            try:
                r = make_retargeter(cfg, pkl); r.retarget(); r.save_results_as_csv(str(out))
                meta = cache_metadata(cfg, candidate, row, pkl, out, r.num_frames)
                metadata_path(out).write_text(json.dumps(meta, indent=2) + "\n")
                status.append({"id": row["id"], "status": "success", "seconds": time.time()-start,
                               "qpos_sha256": sha(out), "config_sha256": config_hash, "frames": r.num_frames,
                               "cache_reason": reason, "metadata": str(metadata_path(out).relative_to(wd))})
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
            per[f"{group}_{label}_pos_p95_cm"] = float(np.percentile(pos,95,method="linear")); per[f"{group}_{label}_rot_p95_deg"] = float(np.percentile(rot,95,method="linear"))
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
    acc = np.abs(np.diff(vel[:,[r.model.jnt_dofadr[j] for j in se]],axis=0)*FPS)
    per["qdot_p95_rad_s"] = float(np.percentile(vse,95,method="linear")); per["qdot_max_rad_s"] = float(np.max(vse))
    per["qddot_p95_rad_s2"] = float(np.percentile(acc,95,method="linear")); per["qddot_max_rad_s2"] = float(np.max(acc))
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
    selected=[args.candidate] if args.candidate else list(CANDIDATES)
    chosen_rows=[r for r in rows if not args.motion_ids or r["id"] in args.motion_ids]
    # Candidate-wise evaluation is intentionally resumable; each chunk is a
    # read-only measurement of already generated trajectories.
    old = []
    if (wd/"metrics.csv").exists():
        with (wd/"metrics.csv").open(newline="") as f:
            old=[dict(x) for x in csv.DictReader(f) if not (x["candidate"] in selected and int(x["motion_id"]) in {r["id"] for r in chosen_rows})]
    for candidate in selected:
        cfg=resolved_config(candidate)
        for motion in chosen_rows:
            qpath=wd/"qpos"/candidate/f"{motion['id']:02d}_{Path(motion['source']).stem}.csv"
            valid, reason, provenance = validate_snapshot(cfg, candidate, motion, wd / motion["pkl"], qpath)
            if not valid: raise RuntimeError(f"refusing unverified cache {candidate}/{motion['id']}: {reason}")
            row, proof=evaluate_one(cfg,wd/motion["pkl"],qpath,candidate,motion); metrics.append(row); audit.append({"candidate":candidate,"motion_id":motion["id"],**proof,"generation_git_sha":provenance["generation_git_sha"],"generation_solver_sha256":provenance["generation_solver_sha256"]})
    metrics = old + metrics
    # motion-relative fidelity ratios use matched B0 only.
    b0={(int(x["motion_id"])):x for x in metrics if x["candidate"]=="B0"}
    for x in metrics:
        if x["candidate"] != "B0" and int(x["motion_id"]) not in b0: continue
        base=b0[int(x["motion_id"])]
        ratios = [x[f"four_ee_{n}_arc_m"]/base[f"four_ee_{n}_arc_m"] for n in FOUR_EE if base[f"four_ee_{n}_arc_m"] > 1e-12]
        if not ratios or base["arm_range_mean_rad"] <= 1e-12: raise ValueError("zero B0 ratio denominator")
        x["four_ee_arc_over_B0"] = float(np.mean(ratios)); x["arm_range_over_B0"] = x["arm_range_mean_rad"]/base["arm_range_mean_rad"]
    out=wd/"metrics.csv"; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=sorted({k for x in metrics for k in x})); w.writeheader(); w.writerows(metrics)
    (wd/"metric_sample_counts.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(out)

def independent_audit_metrics(cfg: dict, pkl_path: Path, qpos_path: Path, candidate: str) -> dict:
    """Literal, standalone audit path: no production evaluator/retargeter helpers.

    It intentionally owns its pair table, pickle parsing, CSV parsing, FK,
    quaternion-to-matrix angle and limit/derivative calculations.  Keeping it
    boring and duplicated makes correlated evaluation bugs much less likely.
    """
    pairs = {"four_ee": {"left_hand": ("left_fore_arm", "left_hand_link"), "right_hand": ("right_fore_arm", "right_hand_link"), "left_foot": ("left_foot_end_link", "left_foot_end_link"), "right_foot": ("right_foot_end_link", "right_foot_end_link")}, "upper_limb": {"left_shoulder": ("left_shoulder", "left_shoulder_x_link"), "right_shoulder": ("right_shoulder", "right_shoulder_x_link"), "left_elbow": ("left_arm", "left_elbow_link"), "right_elbow": ("right_arm", "right_elbow_link"), "left_hand": ("left_fore_arm", "left_hand_link"), "right_hand": ("right_fore_arm", "right_hand_link")}}
    target = pickle.loads(pkl_path.read_bytes()); names = {n: i for i, n in enumerate(target["keypoint_names"])}
    model = mujoco.MjModel.from_xml_path(str(ROOT / cfg["robot_xml_path"])); data = mujoco.MjData(model)
    for needle, offsets in cfg.get("joints_limit_offset_degrees", {}).items():
        lo, hi = offsets if isinstance(offsets, list) else (offsets, 0.0)
        for jid in range(model.njnt):
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if needle in jname: model.jnt_range[jid] += np.radians([lo, hi])
    q = np.atleast_2d(np.loadtxt(qpos_path, delimiter=",", dtype=float))
    if q.shape[1] != model.nq or not np.isfinite(q).all(): raise ValueError("invalid audit qpos")
    # CSV is xyzw whereas MuJoCo uses wxyz.
    q[:, 3:7] = q[:, [6, 3, 4, 5]]; norm = np.linalg.norm(q[:, 3:7], axis=1)
    if np.any(norm < 1e-12): raise ValueError("zero audit quaternion")
    q[:, 3:7] /= norm[:, None]; T = len(q)
    values = {g: {n: {"pos": [], "rot": []} for n in ps} for g, ps in pairs.items()}; arcs = {n: [] for n in pairs["four_ee"]}
    def angle_deg(a, b):
        # Explicit matrix relative-angle computation, independent of geodesic().
        a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
        ra = np.empty(9); rb = np.empty(9); mujoco.mju_quat2Mat(ra, a); mujoco.mju_quat2Mat(rb, b)
        c = np.clip((np.trace(ra.reshape(3,3).T @ rb.reshape(3,3)) - 1) / 2, -1, 1)
        return float(np.degrees(np.arccos(c)))
    for frame, qq in enumerate(q):
        data.qpos[:] = qq; mujoco.mj_forward(model, data)
        for group, mapping in pairs.items():
            for label, (source, body) in mapping.items():
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body); idx = names[source]; p = data.xpos[bid].copy()
                values[group][label]["pos"].append(float(np.linalg.norm(p-target["positions"][frame,idx])*100))
                values[group][label]["rot"].append(angle_deg(data.xquat[bid], target["quaternions"][frame,idx]))
                if group == "four_ee": arcs[label].append(p)
    out = {}
    for group, mapping in pairs.items():
        for label in mapping:
            out[f"{group}_{label}_pos_p95_cm"] = float(np.percentile(values[group][label]["pos"],95,method="linear")); out[f"{group}_{label}_rot_p95_deg"] = float(np.percentile(values[group][label]["rot"],95,method="linear"))
        out[f"{group}_pos_p95_cm"] = float(np.mean([out[f"{group}_{n}_pos_p95_cm"] for n in mapping])); out[f"{group}_rot_p95_deg"] = float(np.mean([out[f"{group}_{n}_rot_p95_deg"] for n in mapping]))
    for n, x in arcs.items(): out[f"four_ee_{n}_arc_m"] = float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum())
    def joints(needle): return [j for j in range(model.njnt) if needle in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or "") and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    shoulder, elbow = joints("shoulder"), joints("elbow")
    for label, ids in (("shoulder",shoulder),("elbow",elbow)):
        near=[]; hard=[]; nj=hj=0
        for qq in q:
            d=np.array([min(abs(qq[model.jnt_qposadr[j]]-model.jnt_range[j,0]),abs(qq[model.jnt_qposadr[j]]-model.jnt_range[j,1])) for j in ids]); nj += int((d<=np.deg2rad(5)).sum()); hj += int((d<=.001).sum()); near.append(bool((d<=np.deg2rad(5)).any())); hard.append(bool((d<=.001).any()))
        def run(a):
            best=cur=0
            for x in a: cur=cur+1 if x else 0; best=max(best,cur)
            return best
        out.update({f"{label}_near_joint_pct":100*nj/(T*len(ids)),f"{label}_hard_joint_pct":100*hj/(T*len(ids)),f"{label}_near_frame_pct":100*np.mean(near),f"{label}_hard_frame_pct":100*np.mean(hard),f"{label}_near_longest_run":run(near),f"{label}_hard_longest_run":run(hard)})
    ids=shoulder+elbow; dof=[model.jnt_dofadr[j] for j in ids]; velocity=[]
    for a,b in zip(q[:-1],q[1:]):
        v=np.empty(model.nv); mujoco.mj_differentiatePos(model,v,1/FPS,a,b); velocity.append(v[dof])
    v=np.abs(np.asarray(velocity)); acc=np.abs(np.diff(np.asarray(velocity),axis=0)*FPS)
    out.update({"qdot_p95_rad_s":float(np.percentile(v,95,method="linear")),"qdot_max_rad_s":float(v.max()),"qddot_p95_rad_s2":float(np.percentile(acc,95,method="linear")),"qddot_max_rad_s2":float(acc.max())})
    out["edge_pct"] = None if candidate=="B0" else float(100*np.mean(v >= .99*np.array([cfg["inter_frame_velocity_limit"]["joint_velocity_limits"]["shoulder" if "shoulder" in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or "") else "elbow"] for j in ids])))
    arm=joints("shoulder")+joints("elbow")+joints("wrist"); out["arm_range_mean_rad"] = float(np.mean([np.ptp(q[:,model.jnt_qposadr[j]]) for j in arm]))
    return out

def audit(args) -> None:
    """Independently audit all saved qpos/PKL pairs against metrics.csv."""
    wd = experiment_dir(args)
    rows = json.loads((wd / "manifest.json").read_text())
    with (wd / "metrics.csv").open(newline="") as f:
        reported = {(x["candidate"], int(x["motion_id"])): x for x in csv.DictReader(f)}
    selected=[args.candidate] if args.candidate else list(CANDIDATES)
    selected_rows=[m for m in rows if not args.motion_ids or m["id"] in args.motion_ids]
    checks, maximum = [], 0.0
    for candidate in selected:
        cfg = resolved_config(candidate)
        for motion in selected_rows:
            qpath = wd / "qpos" / candidate / f"{motion['id']:02d}_{Path(motion['source']).stem}.csv"
            valid, reason, provenance = validate_snapshot(cfg, candidate, motion, wd / motion["pkl"], qpath)
            if not valid: raise RuntimeError(f"audit snapshot failure: {reason}")
            independent = independent_audit_metrics(cfg, wd / motion["pkl"], qpath, candidate)
            prior = reported[(candidate, motion["id"])]
            differences = {name: 0.0 if value is None and prior[name] == "" else abs(value - float(prior[name])) for name, value in independent.items()}
            maximum = max(maximum, *differences.values())
            checks.append({"candidate": candidate, "motion_id": motion["id"],
                           "four_ee_samples": 4 * int(prior["frames"]),
                           "upper_limb_samples": 6 * int(prior["frames"]),
                           "generation_git_sha": provenance["generation_git_sha"], "max_abs_difference": max(differences.values()),
                           # The audit uses rotation matrices while production uses
                           # quaternion dot products; acos conditioning near zero
                           # gives harmless ~4e-5 degree roundoff.
                           "passed": max(differences.values()) <= 5e-4})
    out = wd / "independent_audit.csv"
    if out.exists() and not args.force:
        with out.open(newline="") as f:
            keep=[dict(x) for x in csv.DictReader(f) if not (x["candidate"] in selected and int(x["motion_id"]) in {m["id"] for m in selected_rows})]
        checks=keep+checks
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0])); writer.writeheader(); writer.writerows(checks)
    maximum=max(float(x["max_abs_difference"]) for x in checks)
    summary = {"rows": len(checks), "passed_rows": sum(str(x["passed"]).lower()=="true" for x in checks),
               "max_abs_difference": maximum, "tolerance": 5e-4}
    (wd / "independent_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["passed_rows"] != len(checks):
        raise RuntimeError(f"independent audit failed: {summary}")
    print(out)

def aggregate_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate both aggregations; frame-weighted means are not pooled P95s."""
    numeric = [k for k in rows[0] if k not in {"candidate", "motion_id", "source", "frames", "edge_pct"}]
    action=[]; frame=[]; worst=[]
    for candidate in CANDIDATES:
        part=[r for r in rows if r["candidate"]==candidate]; total=sum(int(r["frames"]) for r in part)
        if not part:
            continue
        a={"candidate":candidate,"motions":len(part)}; f={"candidate":candidate,"frames":total}
        for key in numeric:
            vals=[float(r[key]) for r in part]; a[key]=float(np.mean(vals)); f[key]=float(sum(float(r[key])*int(r["frames"]) for r in part)/total)
        if candidate == "B0": a["edge_pct"]=f["edge_pct"]=None
        else:
            a["edge_pct"]=float(np.mean([float(r["edge_pct"]) for r in part])); f["edge_pct"]=float(sum(float(r["edge_pct"])*int(r["frames"]) for r in part)/total)
        w={"candidate":candidate}
        for key in ("four_ee_pos_p95_cm","upper_limb_pos_p95_cm","shoulder_hard_frame_pct","shoulder_hard_longest_run","elbow_hard_frame_pct","elbow_hard_longest_run","qddot_max_rad_s2","qddot_p95_rad_s2"):
            if key not in part[0]:
                continue
            r=max(part,key=lambda x:float(x[key])); w[f"worst_{key}"]=r[key]; w[f"worst_{key}_motion_id"]=r["motion_id"]
        action.append(a); frame.append(f); worst.append(w)
    return action,frame,worst

def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)

def archive(args) -> None:
    """Commit-sized reproducibility archive; trajectory CSVs intentionally stay ignored."""
    wd=experiment_dir(args); data=ROOT/"docs/data"; data.mkdir(parents=True,exist_ok=True)
    with (wd/"metrics.csv").open(newline="") as f: rows=list(csv.DictReader(f))
    action, frame, worst = aggregate_rows(rows)
    write_csv(wd/"aggregate_action_equal.csv",action); write_csv(wd/"aggregate_frame_weighted.csv",frame); write_csv(wd/"worst_cases.csv",worst)
    # Retain only complete reproducibility evidence, never huge qpos trajectories.
    copies={"metrics.csv":"dr02_temporal_velocity_matched_v2_results.csv","manifest.json":"dr02_temporal_velocity_matched_v2_manifest.json","metric_sample_counts.json":"dr02_temporal_velocity_matched_v2_sample_counts.json","independent_audit.csv":"dr02_temporal_velocity_matched_v2_independent_audit.csv","independent_audit_summary.json":"dr02_temporal_velocity_matched_v2_independent_audit_summary.json","aggregate_action_equal.csv":"dr02_temporal_velocity_matched_v2_action_equal.csv","aggregate_frame_weighted.csv":"dr02_temporal_velocity_matched_v2_frame_weighted.csv","worst_cases.csv":"dr02_temporal_velocity_matched_v2_worst_cases.csv"}
    for src,dst in copies.items(): shutil.copy2(wd/src,data/dst)
    resolved=[]
    for p in sorted((wd/"configs").glob("*.yaml")):
        shutil.copy2(p,data/f"dr02_temporal_velocity_matched_v2_{p.name}"); resolved.append({"candidate":p.stem,"path":p.name,"sha256":sha(p)})
    snapshots=[]
    for p in sorted((wd/"qpos").glob("*/*.metadata.json")):
        meta=json.loads(p.read_text()); snapshots.append({"trajectory":str(p.relative_to(wd)).replace(".metadata.json",""),"metadata":str(p.relative_to(wd)),"qpos_sha256":meta["qpos_sha256"],"generation_git_sha":meta["code_git_sha"],"runner_sha256":meta["runner_sha256"],"solver_sha256":meta.get("solver_sha256") or git_file_sha(meta["code_git_sha"],"scripts/robot_retarget.py"),"model_sha256":meta["model_sha256"],"source_sha256":meta["source_sha256"],"pkl_sha256":meta["shared_pkl_sha256"],"config_sha256":meta["resolved_config_sha256"],"frames":meta["frames"],"nq":meta["nq"]})
    (data/"dr02_temporal_velocity_matched_v2_provenance.json").write_text(json.dumps({"current_git_sha":git_sha(),"resolved_configs":resolved,"trajectories":snapshots},indent=2)+"\n")
    print(data)

def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["prepare","run","evaluate","audit","archive"]); p.add_argument("--work-dir"); p.add_argument("--candidate",choices=list(CANDIDATES)); p.add_argument("--motion-ids",type=int,nargs="*"); p.add_argument("--force",action="store_true"); a=p.parse_args(); {"prepare":prepare,"run":run,"evaluate":evaluate,"audit":audit,"archive":archive}[a.command](a)
if __name__ == "__main__": main()
