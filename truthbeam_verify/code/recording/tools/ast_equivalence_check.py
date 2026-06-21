#!/usr/bin/env python3
"""ast_equivalence_check.py — prove every moved function is AST-equivalent
to its pre-refactor counterpart.

Stronger than the "434 literals present" static check: that one only showed
no strings were *deleted*. This one shows each moved function's *body* is
the same Python AST as before the refactor, modulo a normalised leading
docstring.

Limitations (read before relying on this):

1. AST equivalence of a function body does NOT prove the surrounding
   module's behaviour is preserved. Module-level side effects (import
   order, top-level `os.environ.setdefault` calls, etc.) are out of
   scope here.
2. A Name("foo") node compares equal regardless of where "foo" resolves.
   This is a feature here because the refactor deliberately kept old
   private names via `from newmod import ... as _old_name`; the class
   bodies still call `_old_name`, resolving now to the aliased import.
3. Functions whose body was rebased (callsite rewrites) — specifically
   `main()` in tb_main.py — are NOT body-equivalent to the pre-refactor
   `main()` by design. They're excluded from the strict check and flagged
   separately.

Run:
    python3 tools/ast_equivalence_check.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OLD = REPO / "_archive" / "pre_refactor_20260421_182758" / "protocol"
NEW = REPO / "truth_beam_recording" / "protocol"


# (old_module, old_name, new_module, new_name). Functions where the body
# was copied verbatim — AST equivalence is expected.
VERBATIM_MOVES = [
    ("tb_loop.py", "_probe_aravis_version",        "camera.py",           "probe_aravis_version"),
    ("tb_loop.py", "_probe_camera_serial",         "camera.py",           "probe_camera_serial"),
    ("tb_loop.py", "_probe_camera_firmware",       "camera.py",           "probe_camera_firmware"),
    ("tb_loop.py", "_probe_edid_fingerprint",      "projector.py",        "probe_edid_fingerprint"),
    ("tb_loop.py", "_probe_host_config",           "host_info.py",        "probe_host_config"),
    ("tb_loop.py", "_load_tile_generator",         "tile_backend.py",     "load_tile_generator"),
    ("tb_loop.py", "_save_tile_png_safe",          "session_finalize.py", "save_tile_png_safe"),
    ("tb_loop.py", "debayer_bayerrg_nn",           "session_finalize.py", "debayer_bayerrg_nn"),
    ("tb_loop.py", "pick_monitor",                 "projector.py",        "pick_monitor"),
    ("tb_loop.py", "_preflight",                   "rsk_integration.py",  "preflight"),
    ("tb_loop.py", "offline_encode_previews",      "session_finalize.py", "offline_encode_previews"),
    ("tb_loop.py", "_post_session_reconcile_capture_log",
                                                   "session_finalize.py", "reconcile_capture_log"),
    ("tb_loop.py", "_anchor_interval_s_type",      "tb_main.py",          "_anchor_interval_s_type"),
    ("tb_loop.py", "_final_tx_wait_s_type",        "tb_main.py",          "_final_tx_wait_s_type"),
    ("tb_loop.py", "parse_args",                   "tb_main.py",          "parse_args"),
]

# main() had its callsites rebased; body is NOT expected to be AST-equal.
# Recorded here for completeness; a human review of tb_main.main vs
# _archive/.../tb_loop.main is the right tool for this one.
REBASED_MOVES = [
    ("tb_loop.py", "main",                         "tb_main.py",          "main"),
]


def _find_function(source: str, name: str) -> ast.FunctionDef | None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
        # Reach inside classes (for methods like _probe_host_config that
        # were actually nested? — not the case here, but be defensive).
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    return sub
    return None


def _strip_leading_docstring(fn: ast.FunctionDef) -> list[ast.stmt]:
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return body


def _normalise(fn: ast.FunctionDef) -> str:
    """Dump the function body AST with position info stripped and any
    leading docstring removed. The result is a string that compares equal
    iff two bodies are structurally identical."""
    body = _strip_leading_docstring(fn)
    # Build a synthetic module containing only the body — dump that.
    module = ast.Module(body=body, type_ignores=[])
    return ast.dump(module, annotate_fields=True, include_attributes=False)


def _check_verbatim(
    old_file: Path, old_name: str, new_file: Path, new_name: str
) -> tuple[bool, str]:
    if not old_file.exists():
        return False, f"old source not found: {old_file}"
    if not new_file.exists():
        return False, f"new source not found: {new_file}"
    old_fn = _find_function(old_file.read_text(), old_name)
    if old_fn is None:
        return False, f"{old_name} not in {old_file}"
    new_fn = _find_function(new_file.read_text(), new_name)
    if new_fn is None:
        return False, f"{new_name} not in {new_file}"
    if _normalise(old_fn) == _normalise(new_fn):
        return True, "OK"
    return False, "AST bodies differ"


def main() -> int:
    if not OLD.exists():
        print(f"FATAL: pre-refactor backup not found at {OLD}")
        return 2

    verbatim_fail = 0
    print("== VERBATIM MOVES (AST-body equivalence expected) ==")
    for old_mod, old_name, new_mod, new_name in VERBATIM_MOVES:
        ok, msg = _check_verbatim(
            OLD / old_mod, old_name, NEW / new_mod, new_name
        )
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] {old_mod}:{old_name} -> {new_mod}:{new_name}  ({msg})")
        if not ok:
            verbatim_fail += 1

    print()
    print("== REBASED MOVES (callsites rewritten; body diff expected) ==")
    for old_mod, old_name, new_mod, new_name in REBASED_MOVES:
        old_fn = _find_function((OLD / old_mod).read_text(), old_name)
        new_fn = _find_function((NEW / new_mod).read_text(), new_name)
        if old_fn is None or new_fn is None:
            print(f"  [?? ] {old_mod}:{old_name} -> {new_mod}:{new_name}  "
                  f"(one side missing)")
            continue
        old_n = sum(1 for _ in ast.walk(old_fn))
        new_n = sum(1 for _ in ast.walk(new_fn))
        print(f"  [NOTE] {old_mod}:{old_name} -> {new_mod}:{new_name}  "
              f"nodes: old={old_n}, new={new_n}  "
              f"(expected: similar order of magnitude)")

    print()
    if verbatim_fail:
        print(f"FAIL: {verbatim_fail} verbatim move(s) do not match.")
        return 1
    print("PASS: every verbatim move is AST-body-equivalent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
