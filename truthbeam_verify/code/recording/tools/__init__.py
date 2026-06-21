# Make tools/ importable both as `from tools import <script>` (from the
# project root) and via direct invocation. The tool scripts themselves
# handle `sys.path.insert(..., '../protocol')` at their top so they can
# reach shared protocol modules.
