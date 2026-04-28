"""
ic_processor.py
---------------
Calculates Ice Concentration (CI) from delta-P and delta-G brightness
temperature differences for each MWR beam.


Equivalent to: IC_processor_1v3.m
Author (original): Sergio Masuelli
Traduccion a Python: Sebastian Heredia
"""

import numpy as np
import scipy.io as sio


# ---------------------------------------------------------------------------
# Tie points  (dTp, dTg)
# ---------------------------------------------------------------------------
# Odd beams  (beams 1, 3, 5, 7  →  0-indexed 0, 2, 4, 6)
dF1 = np.array([21.6,  -4.4])
dM1 = np.array([20.5, -11.2])
dO1 = np.array([62.7,  12.4])

# Even beams (beams 2, 4, 6, 8  →  0-indexed 1, 3, 5, 7)
dF2 = np.array([27.4,  -5.7])
dM2 = np.array([25.0, -10.2])
dO2 = np.array([73.3,  13.9])

FILL     = -999.0
NUMBEAM  = 8
NUMVAR0  = 5   # input  cols: lat, lon, dTp, dTg, asc/des
NUMVAR1  = 7   # output cols: lat, lon, dTp, dTg, asc/des, CI, beam


def _precompute(dF, dM, dO):
    """Return (factor, denominator) for the linear CI interpolation."""
    factor      = (dF[1] - dM[1]) / (dF[0] - dM[0])
    denominator = (dM[1] - dO[1]) - (dM[0] - dO[0]) * factor
    return factor, denominator


def process_ic(dtemp: np.ndarray) -> np.ndarray:
    """
    Compute Ice Concentration from multi-beam brightness temperature deltas.

    Parameters
    ----------
    dtemp : ndarray, shape (8, 5, N)
        Axis 0 → beam index (0-based)
        Axis 1 → [lat, lon, dTp, dTg, asc_des]
        Axis 2 → record index

    Returns
    -------
    CI : ndarray, shape (7, M)
        Rows: [lat, lon, dTp, dTg, asc_des, CI_value, beam_number (1-based)]
        M = total valid records across all beams.
        CI_value is clipped to [0, 1].
    """
    N  = dtemp.shape[2]
    CI = FILL * np.ones((NUMVAR1, NUMBEAM * N))

    factor1, denom1 = _precompute(dF1, dM1, dO1)
    factor2, denom2 = _precompute(dF2, dM2, dO2)

    def _valid_records(beam_idx: int) -> int:
        """Number of non-fill records for a given beam (0-based)."""
        fill_idx = np.where(dtemp[beam_idx, 0, :] == FILL)[0]
        return int(fill_idx[0]) if len(fill_idx) > 1 else N

    def _fill_ci(beam_idx, ini, factor, denom, beam_number):
        nr  = _valid_records(beam_idx)
        fin = ini + nr

        CI[:NUMVAR0, ini:fin] = dtemp[beam_idx, :NUMVAR0, :nr]
        CI[5, ini:fin] = (
            dtemp[beam_idx, 3, :nr] - dO1[1]   # dTg  (row 3, 0-indexed)
            if beam_number % 2 != 0 else
            dtemp[beam_idx, 3, :nr] - dO2[1]
        )
        # Rewrite properly for each parity:
        if beam_number % 2 != 0:   # odd beam (1-based)
            CI[5, ini:fin] = (
                dtemp[beam_idx, 3, :nr] - dO1[1]
                - (dtemp[beam_idx, 2, :nr] - dO1[0]) * factor
            ) / denom
        else:                       # even beam (1-based)
            CI[5, ini:fin] = (
                dtemp[beam_idx, 3, :nr] - dO2[1]
                - (dtemp[beam_idx, 2, :nr] - dO2[0]) * factor
            ) / denom

        CI[6, ini:fin] = beam_number
        return fin

    ini = 0
    # Odd beams first (MATLAB order: 1,3,5,7 → 0-indexed 0,2,4,6)
    for j in range(0, NUMBEAM, 2):
        ini = _fill_ci(j, ini, factor1, denom1, beam_number=j + 1)

    # Even beams next (MATLAB order: 2,4,6,8 → 0-indexed 1,3,5,7)
    for j in range(1, NUMBEAM, 2):
        ini = _fill_ci(j, ini, factor2, denom2, beam_number=j + 1)

    # Clip CI to [0, 1] and trim to valid records
    CI[5, :ini] = np.clip(CI[5, :ini], 0.0, 1.0)
    return CI[:, :ini]


# ---------------------------------------------------------------------------
# Standalone entry point (mirrors IC_processor called from Multi_passesPRO)
# ---------------------------------------------------------------------------
def run(infile: str, oufile: str) -> None:
    data  = sio.loadmat(infile)
    dtemp = data['dtemp']          # expected key from Delta_PG_processor
    CI    = process_ic(dtemp)
    sio.savemat(oufile, {'CI': CI})
    print(f'  Saved CI {CI.shape} → {oufile}')
