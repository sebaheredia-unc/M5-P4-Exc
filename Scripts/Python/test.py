#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:09:52 2026

@author: adaip
"""


import os
import numpy as np
import h5py
import scipy.io as sio
from Scripts.Python.load_l1b import load_l1b


# ---------------------------------------------------------------------------
# Para test
# ---------------------------------------------------------------------------

indir   = './data/L1/EO_20130424_000704_CUSS_SACD_MWR_L1B_SCI_078_000_004/'
oudir   = './data/Temp/DeltaP_G50S1v1'

filename = 'EO_20130424_000704_CUSS_SACD_MWR_L1B_SCI_078_000_004.h5'
h5file = os.path.join(indir, filename)


# leemos y transformamos la estructura
data = load_l1b(h5file)

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------
FILL      = -999.0
NUMBAND   =  3
NUMBEAM   =  8
NUMVAR    =  4      # variables del dato crudo: lat, lon, Tb, asc_des
RECEIVERS = ['RX37H', 'RX37V', 'RX23H']

N_LAT         =  49.0          # límite norte del filtro de latitud
S_LAT         = -49.0          # límite sur
MASK_LAT      = 361            # filas del land mask (180° / 0.5° + 1)
MASK_LON      = 720            # columnas del land mask (360° / 0.5°)
DIST_THRESH   = 25.0 / 108.0  # umbral de vecindad inter-banda en grados



# ---------------------------------------------------------------------------
# Ver contenido del hdf5
f=h5py.File(h5file, 'r')
# con f.keys() veo los grupos
f.keys()
# Out[32]: <KeysViewHDF5 ['Ancillary Data', 'Geolocation Data', 'Global Metadata', 'MWR Calibrated Radiometric Data', 'Quality indicators']>

# leo los grupos que necesito
rad_data_group = f['MWR Calibrated Radiometric Data']
geo_data_group = f['Geolocation Data']

# imprimimos los keys
rad_data_group.keys()
geo_data_group.keys()
# Out[35]: <KeysViewHDF5 ['k_h_antenna_temperature', 'ka_h_antenna_temperature', 'ka_n45_antenna_temperature', 'ka_p45_antenna_temperature', 'ka_v_antenna_temperature']>

# caegamos las temperaturas de brillo
# banda k
kh_temp = rad_data_group['k_h_antenna_temperature'][:]
# banda ka
kah_temp = rad_data_group['ka_h_antenna_temperature'][:]
kav_temp = rad_data_group['ka_v_antenna_temperature'][:]


# cargamos latitud y longitud
kh_lat = geo_data_group['k_h_latitude'][:]
kh_lon = geo_data_group['k_h_longitude'][:]

# banda ka
kah_lat = geo_data_group['ka_h_latitude'][:]
kah_lon = geo_data_group['ka_h_longitude'][:]
kav_lat = geo_data_group['ka_v_latitude'][:]
kav_lon = geo_data_group['ka_v_longitude'][:]

# transformamos la estructura
data = load_l1b(h5file)

# R_37H=data['RX37H']
# R_37H.keys()
# R_37H_B1=R_37H['B1']
# R_37H_B1.keys()

# ===========================================================================
# 1.  FLAG ASCENDENTE / DESCENDENTE
# ===========================================================================

def _find_turning_points(data: dict) -> tuple[int, int]:
    """
    Detecta isouth e inorth (índices 0-based) como el MATLAB:

        auxx = (data.RX37H.B1.Lat > -100) .* data.RX37H.B1.Lat
        [~, inorth] = max(auxx)
        [~, isouth] = min(auxx)

    La máscara > -100 excluye fill values sin alterar los valores válidos.
    """
    lat    = data['RX37H']['B1']['Lat']
    auxx   = np.where(lat > -100.0, lat, 0.0)
    inorth = int(np.argmax(auxx))
    isouth = int(np.argmin(auxx))
    return isouth, inorth


def _asc_des_flag(k: int, isouth: int, inorth: int) -> int:
    """
    Asigna el flag para el índice k (0-based).

    Asume que el paso comienza descendente (igual que el comentario MATLAB).
        k < isouth          →  1  (descendente)
        k == isouth         →  0  (turning point sur)
        isouth < k < inorth →  2  (ascendente)
        k == inorth         →  0  (turning point norte)
        k > inorth          →  1  (descendente de nuevo)
    """
    if   k <  isouth: return 1
    elif k == isouth: return 0
    elif k <  inorth: return 2
    elif k == inorth: return 0
    else:             return 1


# ===========================================================================
# 2.  FILTRO DE LATITUD
# ===========================================================================

def apply_lat_filter(data: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Conserva solo los registros con latitud de referencia (haz 1)
    fuera de la banda ecuatorial: |lat| > 49° y |lat| < 90°.

    Para cada registro válido copia lat, lon, Tb y flag a rawlat,
    y guarda el índice original en indlat.

    Dimensiones (0-based en Python):
        rawlat  (NUMBAND, NUMBEAM, NUMVAR, numrec)
        indlat  (NUMBAND, NUMBEAM, numrec)
        counlat [3]  — número de registros válidos por banda
    """
    isouth, inorth = _find_turning_points(data)
    numrec  = len(data['RX37H']['B1']['Tb'])

    rawlat  = FILL * np.ones((NUMBAND, NUMBEAM, NUMVAR, numrec))
    indlat  = FILL * np.ones((NUMBAND, NUMBEAM, numrec))
    counlat = [0, 0, 0]

    for i, rx in enumerate(RECEIVERS):
        auxlat = data[rx]['B1']['Lat']    # haz 1 como referencia de filtro
        c = 0
        for k in range(numrec):
            lat_k = auxlat[k]
            if (lat_k > N_LAT or lat_k < S_LAT) and abs(lat_k) < 90.0:
                flag = _asc_des_flag(k, isouth, inorth)
                for ji in range(NUMBEAM):
                    bkey = f'B{ji + 1}'
                    rawlat[i, ji, 0, c] = data[rx][bkey]['Lat'][k]
                    rawlat[i, ji, 1, c] = data[rx][bkey]['Lon'][k]
                    rawlat[i, ji, 2, c] = data[rx][bkey]['Tb'][k]
                    rawlat[i, ji, 3, c] = flag
                    indlat[i, ji, c]    = k
                c += 1
        counlat[i] = c

    return rawlat, indlat, counlat


# ===========================================================================
# 3.  FILTRO DE TIERRA (LAND MASK)
# ===========================================================================

def load_landmask(landmask_file: str) -> np.ndarray:
    """Carga landmask2.mat y aplica flipud (idéntico al MATLAB)."""
    mat = sio.loadmat(landmask_file)
    # landmask=mat['landmask_ser'][:]
    return np.flipud(mat['landmask_ser'].astype(np.int8))


def _latlon_to_mask_idx(lat: float, lon: float) -> tuple[int, int]:
    """
    Convierte lat/lon a índices 0-based del land mask de 0.5°.

    MATLAB (1-based):
        alat = round((lat+90)*2) + 1
        alon = round((lon+180)*2) + 1
        if alon > MASK_LON: alon = alon - MASK_LON + 1

    Python (0-based):
        alat = round((lat+90)*2)              rango [0, 360]
        alon = round((lon+180)*2) % MASK_LON  rango [0, 719]
    """
    alat = int(round((lat  +  90.0) * 2))
    alon = int(round((lon  + 180.0) * 2)) % MASK_LON
    alat = max(0, min(alat, MASK_LAT - 1))
    return alat, alon


def apply_land_filter(rawlat: np.ndarray, indlat: np.ndarray,
                      counlat: list[int],
                      landmask: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Elimina los registros cuya posición cae sobre tierra.

    La banda 36.5 GHz H (band 0) se usa como referencia de posición para
    filtrar simultáneamente H y V (bands 0 y 1) — mismo píxel espacial.
    La banda 23.8 GHz (band 2) se filtra independientemente.

    Retorna
    -------
    rawland  (NUMBAND, NUMBEAM, NUMVAR, max_count)
    indland  (NUMBAND-1, NUMBEAM, max_count)   — H y V comparten índice
    counland (NUMBAND-1, NUMBEAM)  int
    """
    max_count = max(max(counlat), 1)
    rawland   = FILL * np.ones((NUMBAND,     NUMBEAM, NUMVAR, max_count))
    indland   = FILL * np.ones((NUMBAND - 1, NUMBEAM, max_count))
    counland  = np.zeros((NUMBAND - 1, NUMBEAM), dtype=int)

    # --- 36.5 GHz H+V  (band index 0 como referencia de posición) ---
    for ji in range(NUMBEAM):
        c = 0
        for k in range(counlat[0]):
            lat = rawlat[0, ji, 0, k]
            lon = rawlat[0, ji, 1, k]
            if abs(lat) >= 90.0:
                continue
            alat, alon = _latlon_to_mask_idx(lat, lon)
            if landmask[alat, alon] == 0:          # 0 = océano
                rawland[0, ji, :, c] = rawlat[0, ji, :, k]   # RX37H
                rawland[1, ji, :, c] = rawlat[1, ji, :, k]   # RX37V
                indland[0, ji, c]    = indlat[0, ji, k]
                c += 1
        counland[0, ji] = c

    # --- 23.8 GHz  (band index 2) ---
    for ji in range(NUMBEAM):
        c = 0
        for k in range(counlat[2]):
            lat = rawlat[2, ji, 0, k]
            lon = rawlat[2, ji, 1, k]
            if abs(lat) >= 90.0:
                continue
            alat, alon = _latlon_to_mask_idx(lat, lon)
            if landmask[alat, alon] == 0:
                rawland[2, ji, :, c] = rawlat[2, ji, :, k]
                indland[1, ji, c]    = indlat[2, ji, k]
                c += 1
        counland[1, ji] = c

    return rawland, indland, counland


# ===========================================================================
# 4.  CO-LOCALIZACIÓN DE HACES
# ===========================================================================

def _great_circle_deg(lat1: float, lon1: float,
                      lat2: float, lon2: float) -> float:
    """
    Distancia de gran círculo en GRADOS — equivalente a la función
    ``distance()`` de la Mapping Toolbox de MATLAB (modo 'gc').
    """
    r = np.radians
    a = (np.sin((r(lat2) - r(lat1)) / 2) ** 2
         + np.cos(r(lat1)) * np.cos(r(lat2))
         * np.sin((r(lon2) - r(lon1)) / 2) ** 2)
    return float(np.degrees(2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))))


def collocate_beams(rawland: np.ndarray, indland: np.ndarray,
                    counland: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Co-localización de los tres receivers en un punto común.

    Etapa A — 36.5 GHz H+V (pivot):
        Agrupa registros en tripletes con test de contigüidad
        (índice original consecutivo: ind[k+2] - ind[k] == 2).
        Tb co-localizado = (Tb[k] + 2·Tb[k+1] + Tb[k+2]) / 4

    Etapa B — 23.8 GHz (búsqueda secuencial):
        Para cada punto co-localizado de 36 GHz avanza un puntero
        compartido sobre el array 23H hasta encontrar el mínimo local
        (dist1 > dist2) y aplica el mismo promedio de tripletes.
        El puntero NO se resetea entre iteraciones de k (igual MATLAB).

    Etapa C — Filtro de vecindad:
        Descarta pares (37,23) con distancia > DIST_THRESH grados.
        rawloc se compacta in-place; counloc[1,j] guarda el conteo final.

    Retorna
    -------
    rawloc   (NUMBAND, NUMBEAM, NUMVAR, M)
    indloc   (NUMBAND-1, NUMBEAM, M)
    counloc  (NUMBAND-1, NUMBEAM)  int
        counloc[0,j] = puntos co-localizados 36 GHz
        counloc[1,j] = puntos válidos tras el filtro de vecindad
    """
    max_coloc = max(1, int(np.max(counland)) // 3)
    rawloc    = FILL * np.ones((NUMBAND,     NUMBEAM, NUMVAR, max_coloc))
    indloc    = FILL * np.ones((NUMBAND - 1, NUMBEAM, max_coloc))
    counloc   = np.zeros((NUMBAND - 1, NUMBEAM), dtype=int)

    # ---- Etapa A: 36.5 GHz H+V -----------------------------------------
    for ji in range(NUMBEAM):
        n36   = counland[0, ji]
        n_tri = n36 // 3
        cnt   = 0
        for k in range(n_tri):
            i0, i1, i2 = 3 * k, 3 * k + 1, 3 * k + 2
            # Test contigüidad: diferencia de índices originales debe ser 2
            if indland[0, ji, i2] - indland[0, ji, i0] == 2:
                # lat/lon del punto central
                rawloc[0:2, ji, 0:2, cnt] = rawland[0:2, ji, 0:2, i1]
                # Tb ponderado: Tb0 + 2·Tb1 + Tb2  (se divide por 4 al final)
                rawloc[0:2, ji, 2, cnt] = (rawland[0:2, ji, 2, i0]
                                           + rawland[0:2, ji, 2, i2]
                                           + 2.0 * rawland[0:2, ji, 2, i1])
                # flag asc/des del punto central
                rawloc[0:2, ji, 3, cnt] = rawland[0:2, ji, 3, i1]
                indloc[0, ji, cnt]       = indland[0, ji, i1]
                cnt += 1

        if cnt > 0:
            rawloc[0:2, ji, 2, 0:cnt] /= 4.0   # normalizar Tb
        counloc[0, ji] = cnt

    # ---- Etapa B: 23.8 GHz (búsqueda secuencial) -----------------------
    # ptr23[ji] es el puntero corriente sobre rawland[2,ji,...]
    # NO se resetea entre iteraciones de k, igual que counloc(2,j) en MATLAB
    ptr23 = np.zeros(NUMBEAM, dtype=int)

    for ji in range(NUMBEAM):
        n23       = counland[1, ji]
        n36_coloc = counloc[0, ji]

        for k in range(n36_coloc):
            ref_lat = rawloc[0, ji, 0, k]
            ref_lon = rawloc[0, ji, 1, k]

            dist1 = 1001.0
            dist2 = 1000.0

            # Avanzar puntero mientras la distancia siga decreciendo
            while dist1 > dist2 and ptr23[ji] < n23 - 1:
                ptr23[ji] += 1
                dist1 = dist2
                dist2 = _great_circle_deg(
                    ref_lat, ref_lon,
                    rawland[2, ji, 0, ptr23[ji]],
                    rawland[2, ji, 1, ptr23[ji]],
                )

            # ptr23[ji] > 1  (0-based)  ≡  counloc(2,j) > 2  (1-based MATLAB)
            if ptr23[ji] > 1:
                ptr23[ji] -= 1      # retroceder al mínimo local
                c = ptr23[ji]

                # Test de contigüidad del triplete 23 GHz
                if (c + 1 < n23 and c - 1 >= 0
                        and indland[1, ji, c + 1] - indland[1, ji, c - 1] == 2):
                    rawloc[2, ji, 0:2, k] = rawland[2, ji, 0:2, c]
                    rawloc[2, ji, 2,   k] = (rawland[2, ji, 2, c - 1]
                                             + rawland[2, ji, 2, c + 1]
                                             + 2.0 * rawland[2, ji, 2, c])
                    rawloc[2, ji, 3, k]   = rawland[2, ji, 3, c]
                    # Nota: el MATLAB original usa indland(1,j,...) aquí
                    # (band 0, no 1) — posible bug replicado fielmente
                    indloc[1, ji, k] = indland[0, ji, c]

    rawloc[2, :, 2, :] /= 4.0   # normalizar Tb de 23 GHz

    # ---- Etapa C: filtro de vecindad -----------------------------------
    # counloc[1,ji] se reutiliza como puntero de escritura (compactación)
    for ji in range(NUMBEAM):
        write_ptr = 0
        for k in range(counloc[0, ji]):
            lat_37 = rawloc[0, ji, 0, k]
            lon_37 = rawloc[0, ji, 1, k]
            lat_23 = rawloc[2, ji, 0, k]
            lon_23 = rawloc[2, ji, 1, k]

            if lat_37 == FILL or lat_23 == FILL:
                continue

            dist = _great_circle_deg(lat_37, lon_37, lat_23, lon_23)
            if dist < DIST_THRESH:
                rawloc[:, ji, :, write_ptr] = rawloc[:, ji, :, k]
                indloc[:, ji, write_ptr]    = indloc[:, ji, k]
                write_ptr += 1

        counloc[1, ji] = write_ptr

    return rawloc, indloc, counloc


# ===========================================================================
# 5.  CÁLCULO DE ΔTp y ΔTg
# ===========================================================================

def compute_dtemp(rawloc: np.ndarray,
                  counloc: np.ndarray) -> np.ndarray:
    """
    Calcula las diferencias de temperatura de brillo y arma la salida final.

        ΔTp = Tb37V − Tb37H
        ΔTg = Tb37H − Tb23H

    Retorna
    -------
    dtemp : ndarray (NUMBEAM, 5, M)
        M = max(counloc[1, :])
    """
    M = int(np.max(counloc[1, :]))
    if M == 0:
        M = 1
    dtemp = FILL * np.ones((NUMBEAM, NUMVAR + 1, M))

    for ji in range(NUMBEAM):
        n = counloc[1, ji]
        if n == 0:
            continue
        dtemp[ji, 0, :n] = rawloc[0, ji, 0, :n]                          # lat
        dtemp[ji, 1, :n] = rawloc[0, ji, 1, :n]                          # lon
        dtemp[ji, 2, :n] = rawloc[1, ji, 2, :n] - rawloc[0, ji, 2, :n]  # ΔTp
        dtemp[ji, 3, :n] = rawloc[0, ji, 2, :n] - rawloc[2, ji, 2, :n]  # ΔTg
        dtemp[ji, 4, :n] = rawloc[0, ji, 3, :n]                          # flag

    return dtemp



# ===========================================================================
# Cuerpo principal
# ===========================================================================
rawlat, indlat, counlat    = apply_lat_filter(data)
print(f'    Filtro lat:    {counlat}')
# landmask_file = './data/landmask/landMask.mat'
landmask_file = './data/landmask/landmask_ser.mat'

# mat = sio.loadmat(landmask_file)
# mat.keys()
# landmask=mat['landmask_ser'][:]
landmask                   = load_landmask(landmask_file)

rawland, indland, counland = apply_land_filter(rawlat, indlat,
                                               counlat, landmask)
print(f'    Filtro tierra: {counland}')

rawloc, indloc, counloc    = collocate_beams(rawland, indland, counland)
print(f'    Co-loc 36GHz:  {counloc[0]}')
print(f'    Válidos final: {counloc[1]}')

dtemp                      = compute_dtemp(rawloc, counloc)
print(f'    dtemp shape:   {dtemp.shape}')

sio.savemat(oufile, {'dtemp': dtemp})


print(f'    Guardado:      {oufile}')

