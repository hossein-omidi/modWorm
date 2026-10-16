"""Safe, idempotent patch of modWorm's existing Julia PPC bridge.

Run in repo root: python apply_julia_native_arrays.py
The script changes only the Python->Julia boundary, not equations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

HELPER = '''# Native Julia storage for Python-provided NumPy arrays.
# Sundials must not initialize its solution cache with PythonCall.PyArray.
# Arrays are copied ONCE at the inter-language boundary; scalar settings
# and the existing modWorm functions remain unchanged.
Main.eval("""
function modworm_native_ppc_dict(raw)
    native = Dict{String,Any}()
    for (key, value) in raw
        native[string(key)] = value isa AbstractArray ? Array(value) : value
    end
    return native
end
function modworm_native_ppc_dict_list(raws)
    return [modworm_native_ppc_dict(raw) for raw in raws]
end
""")

'''

SINGLE_ORIGINAL = 'Main.run_network_ppc(vars_NN_nv, vars_NN_mb, verbose_period, rtol_nv, atol_nv, rtol_mb, atol_mb)'
SINGLE_UPDATED = ('Main.run_network_ppc(\n'
    '        Main.modworm_native_ppc_dict(vars_NN_nv),\n'
    '        Main.modworm_native_ppc_dict(vars_NN_mb),\n'
    '        verbose_period, rtol_nv, atol_nv, rtol_mb, atol_mb\n'
    '    )')
ENSEMBLE_ORIGINAL = 'Main.run_network_ppc_ensemble(vars_NN_nv_ens, vars_NN_mb_ens, verbose_period, rtol_nv, atol_nv, rtol_mb, atol_mb)'
ENSEMBLE_UPDATED = ('Main.run_network_ppc_ensemble(\n'
    '        Main.modworm_native_ppc_dict_list(vars_NN_nv_ens),\n'
    '        Main.modworm_native_ppc_dict_list(vars_NN_mb_ens),\n'
    '        verbose_period, rtol_nv, atol_nv, rtol_mb, atol_mb\n'
    '    )')
INSERT_ANCHOR = 'def run_network_julia_ensemble('
PATCH_MARKER = 'function modworm_native_ppc_dict(raw)'


def patch_source(source: str) -> tuple[str, bool]:
    """Modify only unambiguous, known upstream call sites. Return source, changed."""
    already = PATCH_MARKER in source
    if already and SINGLE_UPDATED in source and ENSEMBLE_UPDATED in source:
        return source, False
    if already:
        raise RuntimeError('Partially modified native Julia bridge; inspect it manually before retrying.')
    for name, needle in [('Julia section', INSERT_ANCHOR),
                         ('single-worm solver', SINGLE_ORIGINAL),
                         ('ensemble solver', ENSEMBLE_ORIGINAL)]:
        if source.count(needle) != 1:
            raise RuntimeError(f'Expected exactly one {name} anchor; found {source.count(needle)}. No files changed.')
    new = source.replace(INSERT_ANCHOR, HELPER + INSERT_ANCHOR, 1)
    new = new.replace(SINGLE_ORIGINAL, SINGLE_UPDATED, 1)
    new = new.replace(ENSEMBLE_ORIGINAL, ENSEMBLE_UPDATED, 1)
    # Check that Python source is syntactically valid before writing anything.
    compile(new, 'modWorm/proprioception_simulation.py', 'exec')
    return new, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd(),
                        help='Root directory of modWorm checkout')
    args = parser.parse_args()
    target = args.repo / 'modWorm' / 'proprioception_simulation.py'
    if not target.is_file():
        parser.error(f'Not a modWorm checkout or file missing: {target}')
    old = target.read_text(encoding='utf-8')
    new, changed = patch_source(old)
    if not changed:
        print(f'Already patched; left unchanged: {target}')
        return
    backup = target.with_suffix(target.suffix + '.before_julia_arrays.bak')
    if backup.exists():
        raise RuntimeError(f'Backup already exists: {backup}; refusing to overwrite it.')
    shutil.copy2(target, backup)
    try:
        target.write_text(new, encoding='utf-8')
    except Exception:
        shutil.copy2(backup, target)
        raise
    print(f'Patched: {target}')
    print(f'Original backup: {backup}')
    print('Restart the Jupyter kernel before re-running notebook 01.')


if __name__ == '__main__':
    main()
