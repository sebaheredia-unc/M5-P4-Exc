"""
grafic_ci.py
------------
Generates Ice Concentration maps from a CI data file using several
projections: Transverse Mercator (global), Stereographic North Pole,
Stereographic South Pole.  Each projection is produced with all beams
and with beam-2 excluded.

Requires: cartopy, matplotlib

Equivalent to: grafic_CI_1v2.m
Author (original): Sergio Masuelli
Traduccion a Python: Sebastian Heredia
"""

import os

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print('[grafic_ci] WARNING: cartopy not found. '
          'Stereographic projections will be skipped.')


# ---------------------------------------------------------------------------
# CI discretisation (0.1 bins, mirrors MATLAB binning)
# ---------------------------------------------------------------------------
_BIN_EDGES  = np.arange(0.0, 1.05, 0.1)
_BIN_CENTRES = (_BIN_EDGES[:-1] + _BIN_EDGES[1:]) / 2


def _discretise(ci: np.ndarray) -> np.ndarray:
    """Snap CI values to the nearest 0.1-spaced bin centre."""
    idx = np.digitize(ci, _BIN_EDGES, right=False) - 1
    idx = np.clip(idx, 0, len(_BIN_CENTRES) - 1)
    return _BIN_CENTRES[idx]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_ci(infile: str, outpath: str, multipass: bool = False) -> None:
    """
    Load a CI file and produce six PNG maps.

    Parameters
    ----------
    infile    : Path to a .mat file that contains a variable ``CI`` or
                ``aCI`` (shape 7 × N).
    outpath   : Directory where PNG files are written.
    multipass : Set True when *infile* is the concatenated MultiCI.mat
                (variable name is ``aCI``); False for a single-pass file
                (variable name is ``CI``).
    """
    os.makedirs(outpath, exist_ok=True)

    data = sio.loadmat(infile)
    raw  = data.get('aCI', data.get('CI'))
    if raw is None:
        raise KeyError('No "CI" or "aCI" variable found in ' + infile)

    aCI = raw.copy()
    # Row 5 (0-indexed) = CI value; discretise
    aCI[5, :] = _discretise(aCI[5, :])

    lat   = aCI[0, :]
    lon   = aCI[1, :]
    ci    = aCI[5, :]
    beam  = aCI[6, :]

    # Mask for beam-2 exclusion
    mask_b2 = beam == 2

    # ---------- Transverse Mercator (global) ----------
    _plot_mercator(lon, lat, ci, mask_b2, outpath)

    # ---------- Stereographic projections ----------
    if HAS_CARTOPY:
        _plot_stereo_np(lon, lat, ci, mask_b2, outpath)
        _plot_stereo_sp(lon, lat, ci, mask_b2, outpath)
    else:
        print('[grafic_ci] Skipping stereographic plots (cartopy missing).')


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def _scatter_ci(ax, lon, lat, ci, title):
    sc = ax.scatter(lon, lat, c=ci, s=2, cmap='Blues_r',
                    vmin=0, vmax=1, marker='.')
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label='Ice Concentration')


def _plot_mercator(lon, lat, ci, mask_b2, outpath):
    cmap = plt.cm.Blues_r

    # All beams
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('lightgrey')
    sc = ax.scatter(lon, lat, c=ci, s=2, cmap=cmap, vmin=0, vmax=1, marker='.')
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_title('Ice Concentration – Transverse Mercator (all beams)')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.grid(True, linewidth=0.3)
    plt.colorbar(sc, ax=ax, label='Ice Conc.')
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, 'IC_all_TM.png'), dpi=150)
    plt.close(fig)
    print('  Saved: IC_all_TM.png')

    # Without beam 2
    lon_nb2 = np.where(mask_b2, np.nan, lon)
    lat_nb2 = np.where(mask_b2, np.nan, lat)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('lightgrey')
    sc = ax.scatter(lon_nb2, lat_nb2, c=ci, s=2, cmap=cmap,
                    vmin=0, vmax=1, marker='.')
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_title('Ice Concentration – Transverse Mercator (without beam 2)')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.grid(True, linewidth=0.3)
    plt.colorbar(sc, ax=ax, label='Ice Conc.')
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, 'IC_wB2_TM.png'), dpi=150)
    plt.close(fig)
    print('  Saved: IC_wB2_TM.png')


def _stereo_fig(origin_lat, extent_lat):
    """Create a cartopy Stereographic figure + axis."""
    proj = ccrs.Stereographic(central_latitude=origin_lat,
                               central_longitude=0.0)
    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw={'projection': proj},
    )
    ax.set_extent([-180, 180, *extent_lat], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,  facecolor='grey', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)
    ax.gridlines(draw_labels=True, linewidth=0.5, linestyle='--', alpha=0.6)
    return fig, ax


def _stereo_scatter(ax, lon, lat, ci, title, transform):
    cmap = plt.cm.Blues_r
    sc   = ax.scatter(lon, lat, c=ci, s=4, cmap=cmap,
                      vmin=0, vmax=1, marker='.',
                      transform=transform, zorder=2)
    ax.set_title(title)
    return sc


def _plot_stereo_np(lon, lat, ci, mask_b2, outpath):
    geo = ccrs.PlateCarree()
    lat_lim = 50.0
    mask_np = lat >= lat_lim

    # All beams
    fig, ax = _stereo_fig(90, [lat_lim, 90])
    sc = _stereo_scatter(ax,
                         np.where(mask_np, lon, np.nan),
                         np.where(mask_np, lat, np.nan),
                         np.where(mask_np, ci,  np.nan),
                         'North Pole – all beams', geo)
    plt.colorbar(sc, ax=ax, label='Ice Conc.', shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, 'IC_all_NP.png'), dpi=150)
    plt.close(fig)
    print('  Saved: IC_all_NP.png')

    # Without beam 2
    mask_np_nb2 = mask_np & ~mask_b2
    fig, ax = _stereo_fig(90, [lat_lim, 90])
    sc = _stereo_scatter(ax,
                         np.where(mask_np_nb2, lon, np.nan),
                         np.where(mask_np_nb2, lat, np.nan),
                         np.where(mask_np_nb2, ci,  np.nan),
                         'North Pole – without beam 2', geo)
    plt.colorbar(sc, ax=ax, label='Ice Conc.', shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, 'IC_wB2_NP.png'), dpi=150)
    plt.close(fig)
    print('  Saved: IC_wB2_NP.png')


def _plot_stereo_sp(lon, lat, ci, mask_b2, outpath):
    geo = ccrs.PlateCarree()
    lat_lim = -50.0
    mask_sp = lat <= lat_lim

    # All beams
    fig, ax = _stereo_fig(-90, [-90, lat_lim])
    sc = _stereo_scatter(ax,
                         np.where(mask_sp, lon, np.nan),
                         np.where(mask_sp, lat, np.nan),
                         np.where(mask_sp, ci,  np.nan),
                         'South Pole – all beams', geo)
    plt.colorbar(sc, ax=ax, label='Ice Conc.', shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, 'IC_all_SP.png'), dpi=150)
    plt.close(fig)
    print('  Saved: IC_all_SP.png')

    # Without beam 2
    mask_sp_nb2 = mask_sp & ~mask_b2
    fig, ax = _stereo_fig(-90, [-90, lat_lim])
    sc = _stereo_scatter(ax,
                         np.where(mask_sp_nb2, lon, np.nan),
                         np.where(mask_sp_nb2, lat, np.nan),
                         np.where(mask_sp_nb2, ci,  np.nan),
                         'South Pole – without beam 2', geo)
    plt.colorbar(sc, ax=ax, label='Ice Conc.', shrink=0.7)
    fig.tight_layout()
    fig.savefig(os.path.join(outpath, 'IC_wB2_SP.png'), dpi=150)
    plt.close(fig)
    print('  Saved: IC_wB2_SP.png')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    BASE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.normpath(os.path.join(BASE, '..', '..'))  # .../M5-P4/
    CI_DIR   = os.path.join(ROOT, 'data', 'L2', 'CI_50S')

    # # Single pass
    # plot_ci(
    #     infile    = os.path.join(CI_DIR, 'CIPGEO_20130424_000704_CUSS_SACD_MWR_L1B_SCI_078_000_004.mat'),
    #     outpath   = CI_DIR,
    #     multipass = False,
    # )

    # Multi-pass concatenated file
    plot_ci(
        infile    = os.path.join(CI_DIR, 'MultiCI.mat'),
        outpath   = CI_DIR,
        multipass = True,
    )
