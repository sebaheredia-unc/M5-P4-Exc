"""
concat_multipass.py
-------------------
Opens every pass file in a directory and concatenates them into a single
.mat file along a chosen axis.

Equivalent to: concatmultipass1v1.m
Author (original): Sergio Masuelli
Traduccion a Python: Sebastian Heredia
"""

import os
import glob
import time

import numpy as np
import scipy.io as sio


def concat_multipass(
    path:    str,
    pattern: str,
    oufile:  str,
    axis:    int = 1,
) -> np.ndarray:
    """
    Concatenate all matching .mat files in *path* along *axis*.

    The variable name is auto-detected from the first file (mirrors the
    ``whos('-file', ...)`` MATLAB trick).  The output variable is saved
    with an ``'a'`` prefix, e.g. ``CI`` → ``aCI``.

    Parameters
    ----------
    path    : Directory containing the pass files.
    pattern : Glob pattern, e.g. ``'CIPGQ2011242*_MWR_L1_V2.1.mat'``.
    oufile  : Full path of the output .mat file.
    axis    : Concatenation axis (0-based).  MATLAB ``dim=2`` → ``axis=1``.

    Returns
    -------
    Concatenated array.
    """
    files = sorted(glob.glob(os.path.join(path, pattern)))
    n     = len(files)
    if n == 0:
        raise FileNotFoundError(
            f'No files found: {os.path.join(path, pattern)}'
        )

    t0 = time.perf_counter()

    # --- First file: detect variable name and initialise accumulator ---
    first   = files[0]
    meta    = sio.whosmat(first)
    # Filter out MATLAB metadata keys (start with '__')
    user_vars = [m for m in meta if not m[0].startswith('__')]
    if not user_vars:
        raise ValueError(f'No user variables found in {first}')
    var_name    = user_vars[0][0]
    accumulator = sio.loadmat(first)[var_name]
    print(f'[1/{n}]  {os.path.basename(first)}'
          f'  shape={accumulator.shape}'
          f'  var="{var_name}"'
          f'  elapsed={time.perf_counter()-t0:.1f}s')

    # --- Remaining files ---
    for i, fpath in enumerate(files[1:], start=2):
        arr         = sio.loadmat(fpath)[var_name]
        accumulator = np.concatenate([accumulator, arr], axis=axis)
        print(f'[{i}/{n}]  {os.path.basename(fpath)}'
              f'  shape={accumulator.shape}'
              f'  elapsed={time.perf_counter()-t0:.1f}s')

    # --- Save ---
    out_var = 'a' + var_name
    sio.savemat(oufile, {out_var: accumulator})
    total = time.perf_counter() - t0
    print(f'\nSaved "{out_var}" {accumulator.shape} → {oufile}  ({total:.1f}s)')
    return accumulator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    BASE   = os.path.dirname(os.path.abspath(__file__))
    ROOT   = os.path.normpath(os.path.join(BASE, '..', '..'))  # .../M5-P4/
    os.chdir(ROOT)

    DPG_DIR = os.path.join(ROOT, 'data', 'Temp', 'DeltaP_G50S1v1')
    CI_DIR  = os.path.join(ROOT, 'data', 'Temp', 'CI_50S')

    # mode=1 → concatena archivos dtemp (ΔPG)  →  MultiPG.mat  (para ice_scatter)
    # mode=2 → concatena archivos CI           →  MultiCI.mat  (para grafic_ci)
    mode = 1

    if mode == 1:
        # dtemp shape: (8, 5, N)  →  concatenar sobre eje 2 (registros)
        concat_multipass(
            path    = DPG_DIR,
            pattern = 'PGEO_*.mat',
            oufile  = os.path.join(DPG_DIR, 'MultiPG.mat'),
            axis    = 2,
        )

    elif mode == 2:
        # CI shape: (7, M)  →  concatenar sobre eje 1 (columnas)
        concat_multipass(
            path    = CI_DIR,
            pattern = 'CIPGEO_*.mat',
            oufile  = os.path.join(CI_DIR, 'MultiCI.mat'),
            axis    = 1,
        )