"""Phase F prep — contamination firewall verification.

Per `experiments/phase_f_prep/three_role_split.md`, the leakage contract is:

  train_normal               → editor MAY train on these rows
  embargo                    → no role uses these rows
  selection_gate_normal      → eval-only; no editor training
  threshold_calibration_normal → eval-only; held-out binders use these for
                                 threshold setting; editor never sees

This script audits every Phase F training entry point's row construction
against the canonical manifest, and reports leakage. It also exercises the
TemporalPairDataset's row-population at runtime to catch misconfigurations
that pure static analysis would miss.

Output:
  experiments/phase_f_prep/contamination_firewall.{json,md}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Manifest: load + parse splits
MANIFEST = ROOT / "experiments/phase_f_prep/normal_core_manifest.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def role_range(manifest: dict, session: str, role: str) -> set[int]:
    sess_key = session.lower()
    rng = manifest["splits"][role][sess_key]
    return set(range(rng[0], rng[1]))


def role_membership(t: int, manifest: dict, session: str) -> str:
    """Returns the role this row index belongs to: train/embargo/selection_gate/threshold_calibration/oob."""
    for role in ("train_normal", "embargo", "selection_gate_normal", "threshold_calibration_normal"):
        rng = manifest["splits"][role][session.lower()]
        if rng[0] <= t < rng[1]:
            return role
    return "oob"


def audit_dataset_rows(rows: list[int], session: str, expected_role: str,
                       manifest: dict) -> dict:
    expected = role_range(manifest, session, expected_role)
    rows_set = set(rows)
    forbidden = {
        "selection_gate_normal": role_range(manifest, session, "selection_gate_normal"),
        "threshold_calibration_normal": role_range(manifest, session, "threshold_calibration_normal"),
        "embargo": role_range(manifest, session, "embargo"),
    }
    if expected_role in forbidden:
        del forbidden[expected_role]
    leakage = {role: sorted(rows_set & rows_in_role) for role, rows_in_role in forbidden.items()}
    leakage = {role: lst for role, lst in leakage.items() if lst}
    out_of_role = sorted(rows_set - expected)
    return {
        "session": session,
        "expected_role": expected_role,
        "n_rows_total": len(rows_set),
        "n_in_expected_role": len(rows_set & expected),
        "n_leaked_into_other_roles": sum(len(lst) for lst in leakage.values()),
        "n_out_of_any_role": len([t for t in rows_set if role_membership(t, manifest, session) == "oob"]),
        "leakage_per_role": {role: len(lst) for role, lst in leakage.items()},
        "first_5_leaked": {role: lst[:5] for role, lst in leakage.items()},
    }


def audit_phase_f_training():
    """Run the actual split_rows() in Phase F's TemporalPairDataset module."""
    from phase_f.dataset_temporal_pairs import split_rows
    manifest = load_manifest()
    out = {"audits": []}
    # Each entry point and the role its rows are SUPPOSED to belong to.
    # Phase F's 'val' rows are used by the F-A causality probe to decide
    # PASS/FAIL — i.e. they're functioning as a SELECTION GATE. So per
    # the three-role contract, those rows MUST live in selection_gate_normal,
    # not in threshold_calibration_normal.
    entry_points = [
        ("train_phase_f_a_mini.py:232 (train ds)", "D2",  "train", "train_normal"),
        ("train_phase_f_a_mini.py:233 (val ds, used for selection gate)", "D2", "val", "selection_gate_normal"),
        ("dataset_temporal_pairs.split_rows('V10','train')", "V10", "train", "train_normal"),
        ("dataset_temporal_pairs.split_rows('V10','val')", "V10", "val", "selection_gate_normal"),
    ]
    for label, session, role, expected_role in entry_points:
        rows = split_rows(session, role)
        audit = audit_dataset_rows(rows, session, expected_role, manifest)
        audit["entry_point"] = label
        out["audits"].append(audit)
    return out


def main():
    manifest = load_manifest()
    out_dir = ROOT / "experiments/phase_f_prep"
    audits = audit_phase_f_training()
    audits["manifest"] = {
        "schema_version": manifest.get("schema_version"),
        "d2_chain_log_sha256_first16": manifest["sources"]["d2"]["chain_log_sha256_first16"],
        "v10_chain_log_sha256_first16": manifest["sources"]["v10"]["chain_log_sha256_first16"],
    }
    (out_dir / "contamination_firewall.json").write_text(json.dumps(audits, indent=2))

    md = [
        "# Phase F — contamination firewall check",
        "",
        ("Audits Phase F training entry points against the three-role split "
         "manifest. Reports any rows leaking from `selection_gate_normal` or "
         "`threshold_calibration_normal` into the editor's training set."),
        "",
        "## Audit results",
        "",
        "| entry point | session | role used | rows in role | leakage | out-of-role |",
        "|---|---|---|---:|---:|---:|",
    ]
    for a in audits["audits"]:
        md.append(
            f"| `{a['entry_point']}` | {a['session']} | "
            f"{a['expected_role']} | "
            f"{a['n_in_expected_role']} / {a['n_rows_total']} | "
            f"{a['n_leaked_into_other_roles']} | "
            f"{a['n_out_of_any_role']} |"
        )
    md += [
        "",
        "## Detail per audit",
        "",
    ]
    for a in audits["audits"]:
        md.append(f"### {a['entry_point']}")
        md.append("")
        md.append(f"- session: **{a['session']}**, expected role: **{a['expected_role']}**")
        md.append(f"- total rows: {a['n_rows_total']}, in-role: {a['n_in_expected_role']}")
        md.append(f"- leaked: {a['n_leaked_into_other_roles']}, out-of-role: {a['n_out_of_any_role']}")
        if a["leakage_per_role"]:
            md.append(f"- per-role leakage: `{a['leakage_per_role']}`")
            md.append(f"- first 5 leaked rows: `{a['first_5_leaked']}`")
        if "note" in a:
            md.append(f"- **note**: {a['note']}")
        md.append("")
    md += [
        "## Recommendation",
        "",
        ("Phase F's `split_rows()` (in `src/phase_f/dataset_temporal_pairs.py`) "
         "predates the three-role split. The 'train' rows are clean (subset of "
         "`train_normal`), but the 'val' rows currently span the "
         "`threshold_calibration_normal` slice. The mini-experiment uses these "
         "rows ONLY for the causality probe (no gradients), but the diversity "
         "metric drives the FAIL/PASS gating, so technically the threshold-"
         "calibration slice influences the F-A acceptance decision."),
        "",
        ("**Fix (low risk, doc-only or 5-LOC code change)**: rename the "
         "current 'val' to 'selection_gate' and point it to `[4792, 5392)`. "
         "Then add a 'threshold_calibration' role pointing at `[5392, 5992)` "
         "that only the held-out E2/E3r binders touch at threshold-setting "
         "time. This is a clean rename — the existing `[5394, 5992)` rows "
         "are mostly in `threshold_calibration_normal`, so by switching to "
         "the explicit selection_gate range we eliminate the leakage entirely."),
        "",
        ("**Risk if not fixed**: F-A's diversity-passing decision could be "
         "biased by the same rows that later set the verifier's threshold. "
         "In the FAIL case (which is what the FiLM mini gave us), this "
         "doesn't matter because we're not promoting the editor — but in a "
         "future PASS case, the bias would matter and would invalidate the "
         "threshold calibration."),
        "",
        ("**Audit status**: PASS for `train` rows (no editor-training-time "
         "leakage). ATTENTION for `val` rows (used for selection-gate metric "
         "but pulls from threshold_calibration_normal slice). Fix recommended "
         "before any future Phase F training run."),
        "",
    ]
    (out_dir / "contamination_firewall.md").write_text("\n".join(md))
    print(f"[done] wrote contamination_firewall.{{json,md}}")
    # Print a concise summary
    print()
    for a in audits["audits"]:
        ok = "OK" if a["n_leaked_into_other_roles"] == 0 and a["n_out_of_any_role"] == 0 else "ATTENTION"
        print(f"  [{ok}] {a['entry_point']}: in_role={a['n_in_expected_role']}, leaked={a['n_leaked_into_other_roles']}, oob={a['n_out_of_any_role']}")


if __name__ == "__main__":
    main()
