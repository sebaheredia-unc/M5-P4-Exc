"""
run_l1b_batch.py
================
Lista todas las subcarpetas L1B en un directorio raíz y aplica el
procesador Delta_PG sobre cada archivo HDF5 encontrado.

Estructura esperada en L1_DIR:
    L1/
    ├── EO_20130424_000704_CUSS_SACD_MWR_L1B_SCI_078_000_004/
    │   └── EO_20130424_000704_CUSS_SACD_MWR_L1B_SCI_078_000_004.h5
    ├── EO_20130424_...../
    │   └── ...
    └── ...

El script busca el .h5 dentro de cada carpeta (mismo nombre que la carpeta).
Si no lo encuentra, intenta cualquier .h5 que haya en esa carpeta.

Uso
---
    python run_l1b_batch.py

Ajustar las rutas en el bloque  ## CONFIGURACIÓN  antes de ejecutar.
"""

import os
import glob
import time
import scipy.io as sio
from delta_pg_processor import process_delta_pg


# ===========================================================================
## CONFIGURACIÓN — modificar según el entorno
# ===========================================================================

L1_DIR        = r'/ruta/a/L1'          # directorio raíz con las carpetas L1B
OUT_DIR       = r'/ruta/a/DeltaPG'     # donde se guardan los .mat de salida
LANDMASK_FILE = r'/ruta/a/landmask2.mat'
PREFIX        = 'PG'                   # prefijo de los archivos de salida

# ===========================================================================


def find_l1b_folders(l1_dir: str) -> list[tuple[str, str]]:
    """
    Lista todas las subcarpetas L1B y devuelve pares (carpeta, h5_path).

    Estrategia de búsqueda del .h5:
      1. Mismo nombre que la carpeta  (ej: EO_20130424_.../EO_20130424_....h5)
      2. Cualquier .h5 dentro de la carpeta (primer resultado)

    Retorna
    -------
    Lista de (nombre_carpeta, ruta_h5), ordenada alfabéticamente.
    """
    pairs = []
    entries = sorted(os.scandir(l1_dir), key=lambda e: e.name)

    for entry in entries:
        if not entry.is_dir():
            continue

        folder_name = entry.name
        folder_path = entry.path

        # 1. H5 con el mismo nombre que la carpeta
        candidate = os.path.join(folder_path, folder_name + '.h5')
        if os.path.isfile(candidate):
            pairs.append((folder_name, candidate))
            continue

        # 2. Cualquier .h5 en la carpeta
        h5_files = glob.glob(os.path.join(folder_path, '*.h5'))
        if h5_files:
            pairs.append((folder_name, sorted(h5_files)[0]))
        else:
            print(f'  [AVISO] No se encontró .h5 en: {folder_path}')

    return pairs


def run_batch(l1_dir:        str,
              out_dir:       str,
              landmask_file: str,
              prefix:        str = 'PG') -> None:
    """
    Procesa todos los archivos L1B encontrados en l1_dir.

    Para cada archivo genera un .mat en out_dir con el nombre:
        <prefix><nombre_carpeta>.mat
    """
    os.makedirs(out_dir, exist_ok=True)

    pairs  = find_l1b_folders(l1_dir)
    n      = len(pairs)

    if n == 0:
        print(f'No se encontraron carpetas L1B en: {l1_dir}')
        return

    # ---- Listar lo que se va a procesar ----
    print(f'\n{"="*60}')
    print(f'  Carpetas L1B encontradas: {n}')
    print(f'  Directorio de entrada:    {l1_dir}')
    print(f'  Directorio de salida:     {out_dir}')
    print(f'  Land mask:                {landmask_file}')
    print(f'{"="*60}')
    for i, (folder, h5path) in enumerate(pairs, 1):
        print(f'  [{i:3d}/{n}]  {folder}')
    print(f'{"="*60}\n')

    # ---- Procesar ----
    t0      = time.perf_counter()
    errors  = []

    for i, (folder_name, h5path) in enumerate(pairs, 1):
        oufile = os.path.join(out_dir, f'{prefix}{folder_name}.mat')

        print(f'[{i}/{n}]  {folder_name}')

        if os.path.isfile(oufile):
            print(f'  Ya existe, saltando: {os.path.basename(oufile)}')
            continue

        try:
            process_delta_pg(h5path, oufile, landmask_file)
        except Exception as exc:
            msg = f'  [ERROR] {folder_name}: {exc}'
            print(msg)
            errors.append(msg)

        elapsed = time.perf_counter() - t0
        print(f'  Tiempo acumulado: {elapsed:.1f} s\n')

    # ---- Resumen final ----
    total = time.perf_counter() - t0
    print(f'\n{"="*60}')
    print(f'  Procesados: {n - len(errors)}/{n}')
    print(f'  Tiempo total: {total/60:.2f} min')
    if errors:
        print(f'  Errores ({len(errors)}):')
        for e in errors:
            print(f'    {e}')
    print(f'{"="*60}')


# ===========================================================================
if __name__ == '__main__':
    run_batch(L1_DIR, OUT_DIR, LANDMASK_FILE, PREFIX)
