# Make protocol/ importable both as `from protocol import tb_loop` (from the
# project root) and via direct script invocation `python3 protocol/tb_loop.py`.
# The submodules use bare `import tile_gpu` / `import tile_cpu` / etc. for
# sibling imports, which relies on this directory being on sys.path; when
# imported as a package that isn't automatic, so inject it here.
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
