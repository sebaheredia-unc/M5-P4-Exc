"""
ice_scatter.py
--------------
Generates PR vs GR density scatter plots for each beam and each
ascending/descending pass type from the multi-pass concatenated data.

Equivalent to: ICE_scatter2v0.m
Author (original): Sergio Masuelli
Traduccion a Python: Sebastian Heredia
"""

import os

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# Default parameters (mirror the MATLAB script)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    dim_PR  = 50,
    dim_GR  = 50,
    PR_min  = 10,
    PR_max  = 90,
    GR_min  = -40,
    GR_max  = 40,
    numbeam = 8,
)

PASS_LABELS = {1: 'D', 2: 'A'}   # 1=descending, 2=ascending (MATLAB convention)
FILL        = -999.0


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def ice_scatter(
    path:    str,
    infile:  str,
    prefix:  str  = 'ScatterPG_',
    cmap           = None,
    **kwargs,
) -> None:
    """
    Build PR-GR density matrices per beam/pass, save them as .mat files
    and generate PNG figures (linear + log scale).

    Parameters
    ----------
    path    : Output directory for .mat and .png files.
    infile  : Multi-pass .mat file containing ``adtemp`` (8 × 5 × N).
    prefix  : Filename prefix for scatter output files.
    cmap    : Matplotlib colormap (default: 'plasma').
    **kwargs: Override any key in DEFAULTS (dim_PR, PR_min, ...).
    """
    os.makedirs(path, exist_ok=True)

    p = {**DEFAULTS, **kwargs}

    delta_PR = (p['PR_max'] - p['PR_min']) / p['dim_PR']
    delta_GR = (p['GR_max'] - p['GR_min']) / p['dim_GR']

    if cmap is None:
        cmap = plt.cm.plasma

    # --- Load data ---
    data   = sio.loadmat(infile)
    adtemp = data['adtemp']          # shape (8, 5, N)  — 0-based in Python
    N      = adtemp.shape[2]
    NB     = p['numbeam']

    # Axis tick centres
    xPR = np.arange(p['dim_PR']) * delta_PR + p['PR_min'] + delta_PR / 2
    yGR = np.arange(p['dim_GR']) * delta_GR + p['GR_min'] + delta_GR / 2

    # --- Separate records by beam and pass type ---
    vPR = np.full((NB, 2, N), FILL)
    vGR = np.full((NB, 2, N), FILL)
    count = np.zeros((NB, 2), dtype=int)

    for i in range(N):
        for j in range(NB):
            dTp  = adtemp[j, 2, i]
            dTg  = adtemp[j, 3, i]
            desc = int(adtemp[j, 4, i])   # 1=desc, 2=asc

            if dTp == FILL or dTg == FILL:
                continue
            if desc not in (1, 2):
                continue

            k = desc - 1   # 0=desc, 1=asc
            c = count[j, k]
            vPR[j, k, c] = dTp
            vGR[j, k, c] = dTg
            count[j, k] += 1

    # --- Build density matrices and plot ---
    for j in range(NB):
        for k in range(2):
            nc  = count[j, k]
            PR  = vPR[j, k, :nc]
            GR  = vGR[j, k, :nc]
            lbl = PASS_LABELS[k + 1]     # 'D' or 'A'

            density = _build_density(PR, GR, p, delta_PR, delta_GR)

            # Save density matrix
            tag    = f'B{j+1}{lbl}'
            matout = os.path.join(path, f'{prefix}{tag}.mat')
            sio.savemat(matout, {'densidadPRGR': density})

            title_str = f'Beam {j+1} {lbl}'
            _plot_scatter(xPR, yGR, density, title_str, cmap,
                          os.path.join(path, f'{prefix}{tag}.png'),
                          log=False)
            _plot_scatter(xPR, yGR, density, f'log {title_str}', cmap,
                          os.path.join(path, f'{prefix}{tag}_log.png'),
                          log=True)
            plt.close('all')
            print(f'  {tag}  n={nc}  max_density={density.max():.0f}')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_density(
    PR:       np.ndarray,
    GR:       np.ndarray,
    p:        dict,
    delta_PR: float,
    delta_GR: float,
) -> np.ndarray:
    density = np.zeros((p['dim_GR'], p['dim_PR']), dtype=float)
    for pr, gr in zip(PR, GR):
        if (p['PR_min'] <= pr <= p['PR_max'] and
                p['GR_min'] <= gr <= p['GR_max']):
            i_pr = min(int((pr - p['PR_min']) / delta_PR), p['dim_PR'] - 1)
            i_gr = min(int((gr - p['GR_min']) / delta_GR), p['dim_GR'] - 1)
            density[i_gr, i_pr] += 1
    return density


def _plot_scatter(
    xPR:   np.ndarray,
    yGR:   np.ndarray,
    data:  np.ndarray,
    title: str,
    cmap,
    outpath: str,
    log:   bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_data = np.log(data + 1) if log else data
    mesh = ax.pcolormesh(xPR, yGR, plot_data, cmap=cmap, shading='auto')
    ax.set_title(title)
    ax.set_xlabel('PR')
    ax.set_ylabel('GR')
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.colorbar(mesh, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.normpath(os.path.join(BASE, '..', '..'))  # .../M5-P4/
    DPG_DIR = os.path.join(ROOT, 'data', 'Temp', 'DeltaP_G50S1v1')
    OUTPUT_DIR  = os.path.join(DPG_DIR, 'Scatter')
    INFILE = os.path.join(DPG_DIR, 'MultiPG.mat')

    # Optionally load a custom colormap
    # cmap_data = sio.loadmat('MyColormaps1.mat')
    # mycmap = mcolors.ListedColormap(cmap_data['mycmap1'])
    mycmap = None

    ice_scatter(OUTPUT_DIR, INFILE, prefix='ScatterPG_', cmap=mycmap)
