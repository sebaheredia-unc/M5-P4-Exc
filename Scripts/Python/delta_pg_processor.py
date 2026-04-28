"""
delta_pg_processor.py
=====================
Traducción completa a Python de Delta_PG_processor1v1.m

Calcula las diferencias de temperatura de brillo ΔTp y ΔTg a partir de
archivos L1B del MWR (SAC-D/Aquarius) en formato HDF5.

Etapas (idénticas al MATLAB):
  1. Lectura del HDF5  (receivers: RX37H, RX37V, RX23H — 8 haces)
  2. Detección de puntos de inflexión lat (inorth, isouth) → flag asc/des
  3. Filtro de latitud  (|lat| > 49° Y |lat| < 90°, haz 1 como referencia)
  4. Filtro de tierra   (land mask 0.5°, flipud igual que MATLAB)
  5. Co-localización 36.5 GHz H+V: tripletes con promedio ponderado Tb
  6. Co-localización 23.8 GHz: búsqueda secuencial del vecino más cercano
     + mismo promedio ponderado de tripletes
  7. Filtro de distancia inter-banda  (< 25/108 grados ≈ 13 km)
  8. ΔTp = Tb37V − Tb37H  ,  ΔTg = Tb37H − Tb23H

Salida:
  dtemp : ndarray (numbeam=8, numvar=5, numpixels)
    var 0 (fila 0): lat
    var 1 (fila 1): lon
    var 2 (fila 2): ΔTp
    var 3 (fila 3): ΔTg
    var 4 (fila 4): flag  (2=asc, 1=des, 0=turning point, -999=inválido)

Autor original (MATLAB): Sergio Masuelli, 25/01/2012 v1.1
Author de Traduccion a Python: Sebastian Heredia
"""

import os
import numpy as np
import scipy.io as sio

from Scripts.Python.load_l1b import load_l1b




# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------
FILL      = -999.0
NUMBAND   =  3
NUMBEAM   =  8
NUMVAR    =  4      # variables del dato crudo: lat, lon, Tb, asc_des
RECEIVERS = ['RX37H', 'RX37V', 'RX23H']

N_LAT       =  49.0          # límite norte del filtro de latitud
S_LAT       = -49.0          # límite sur
MASK_LAT    = 361            # filas del land mask (180° / 0.5° + 1)
MASK_LON    = 720            # columnas del land mask (360° / 0.5°)
DIST_THRESH = 25.0 / 108.0  # umbral de vecindad inter-banda en grados


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

       Parámetros
       ----------
       rawlat   : ndarray (NUMBAND, NUMBEAM, NUMVAR, N)
                  Datos filtrados por latitud. NUMVAR = [lat, lon, Tb, flag].
       indlat   : ndarray (NUMBAND, NUMBEAM, N)
                  Índice original en el L1B de cada registro filtrado.
       counlat  : list[int], longitud 3
                  Número de registros válidos por banda tras el filtro de latitud:
                    counlat[0] → RX37H  (36.5 GHz H-pol)
                    counlat[1] → RX37V  (36.5 GHz V-pol)
                    counlat[2] → RX23H  (23.8 GHz H-pol)
                  Define cuántos elementos de rawlat/indlat son válidos por banda.
       landmask : ndarray (MASK_LAT, MASK_LON), int8
                  Máscara de tierra a 0.5° de resolución (0=océano, 1=tierra),
                  cargada con flipud para orientación sur→norte.

       Retorna
       -------
       rawland  : ndarray (NUMBAND, NUMBEAM, NUMVAR, max_count)
                  Registros oceánicos por banda y haz.
       indland  : ndarray (NUMBAND-1, NUMBEAM, max_count)
                  Índices originales en el L1B. NUMBAND-1 porque RX37H y RX37V
                  comparten posición espacial y se indexan juntos.
       counland : ndarray (NUMBAND-1, NUMBEAM), int
                  Número de registros oceánicos válidos por banda y haz:
                    counland[0, j] → RX37H+RX37V, haz j
                    counland[1, j] → RX23H,        haz j
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
# 6.  FUNCIÓN PRINCIPAL
# ===========================================================================

def process_delta_pg(h5file: str, oufile: str,
                     landmask_file: str) -> np.ndarray:
    """
    Procesa un archivo L1B HDF5 y guarda dtemp en un .mat.

    Parámetros
    ----------
    h5file        : Ruta al .h5 del L1B.
    oufile        : Ruta de salida del .mat  (variable ``dtemp``).
    landmask_file : Ruta al ``landmask_ser.mat``.
    plot_dir      : Si se indica, guarda un PNG de control en ese directorio.
                    Si es None, no genera figura.

    Retorna
    -------
    dtemp : ndarray (8, 5, N)
    """
    print(f'  Leyendo: {os.path.basename(h5file)}')

    data                       = load_l1b(h5file)
    rawlat, indlat, counlat    = apply_lat_filter(data)
    print(f'    Filtro lat:    {counlat}')

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

    # Siempre genera el gráfico de control, PNG al lado del .mat
    basename  = os.path.splitext(os.path.basename(h5file))[0]
    plot_file = os.path.join(os.path.dirname(oufile), f'control_{basename}.png')
    plot_dtemp(dtemp, title=basename, lat_max=-50.0, outfile=plot_file)

    return dtemp


# Punto de entrada compatible con multi_passes_pro.py (mode='delta_pg')
# landmask_file NO tiene default: debe venir siempre de multi_passes_pro.
def run(infile: str, oufile: str, landmask_file: str) -> None:
    process_delta_pg(infile, oufile, landmask_file)

# ===========================================================================
# 7.  VISUALIZACIÓN DE CONTROL
# ===========================================================================

def plot_dtemp(dtemp: np.ndarray,
               title: str = '',
               lat_max: float = -50.0,
               outfile: str = None) -> None:
    """
    Grafica los valores de ΔTp y ΔTg en función de lat/lon para todos los
    haces, filtrado a la región polar (lat < lat_max, default Antártida).

    Útil para verificar que los datos calculados son coherentes antes de
    guardar el .mat.

    Parámetros
    ----------
    dtemp   : ndarray (8, 5, N) — salida de compute_dtemp()
    title   : título adicional (ej. nombre del archivo)
    lat_max : latitud de corte (solo se grafican puntos con lat < lat_max)
    outfile : si se indica, guarda la figura en ese path (PNG); si no, muestra
    """
    import matplotlib.pyplot as plt

    # Aplanar todos los haces en arrays 1D, solo puntos válidos y polares
    lats, lons, dtps, dtgs = [], [], [], []
    for ji in range(dtemp.shape[0]):
        for k in range(dtemp.shape[2]):
            lat = dtemp[ji, 0, k]
            lon = dtemp[ji, 1, k]
            dtp = dtemp[ji, 2, k]
            dtg = dtemp[ji, 3, k]
            if lat == FILL or lon == FILL or dtp == FILL or dtg == FILL:
                continue
            if lat > lat_max:          # filtro polar
                continue
            lats.append(lat)
            lons.append(lon)
            dtps.append(dtp)
            dtgs.append(dtg)

    if len(lats) == 0:
        print(f'  [plot_dtemp] Sin datos con lat < {lat_max}°')
        return

    lats = np.array(lats)
    lons = np.array(lons)
    dtps = np.array(dtps)
    dtgs = np.array(dtgs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'ΔT control — {title}  (lat < {lat_max}°, n={len(lats)})',
                 fontsize=11)

    for ax, values, label, cmap in [
        (axes[0], dtps, 'ΔTp = Tb37V − Tb37H  [K]', 'RdBu_r'),
        (axes[1], dtgs, 'ΔTg = Tb37H − Tb23H  [K]', 'RdBu_r'),
    ]:
        sc = ax.scatter(lons, lats, c=values, s=3, cmap=cmap, marker='.')
        ax.set_xlabel('Longitud [°]')
        ax.set_ylabel('Latitud [°]')
        ax.set_title(label)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, lat_max + 2)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        fig.colorbar(sc, ax=ax, label='K')

    plt.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=150)
        print(f'  Figura guardada: {outfile}')
    else:
        plt.show()
    plt.close(fig)


# ===========================================================================
# __main__  —  ejecución directa para test de un solo archivo
# ===========================================================================

if __name__ == '__main__':
    import os as _os

    # Mover CWD a la raíz del proyecto (M5-P4/) independientemente de
    # desde donde se ejecute el script.
    _BASE = _os.path.dirname(_os.path.abspath(__file__))       # Scripts/Python/
    _ROOT = _os.path.normpath(_os.path.join(_BASE, '..', '..'))  # M5-P4/
    _os.chdir(_ROOT)
    print(f'CWD -> {_ROOT}')

    # --- rutas de test ---
    _indir    = './data/L1/EO_20130424_000704_CUSS_SACD_MWR_L1B_SCI_078_000_004/'
    _oudir    = './data/Temp/DeltaP_G50S1v1'
    _filename = 'EO_20130424_000704_CUSS_SACD_MWR_L1B_SCI_078_000_004.h5'
    _h5file   = _os.path.join(_indir, _filename)
    _landmask = './data/landmask/landmask_ser.mat'
    _oufile   = _os.path.join(_oudir, 'PG' + _os.path.splitext(_filename)[0] + '.mat')
    _os.makedirs(_oudir, exist_ok=True)

    # --- procesar y graficar (plot_dir activa la figura automáticamente) ---
    dtemp = process_delta_pg(_h5file, _oufile, _landmask,
                             plot_dir=_oudir)
