"""
multi_passes_pro.py
-------------------
Applies a processing function to every file matching a pattern in a
directory and writes one output file per input.

Equivalent to: Multi_passesPRO1v0.m
Author (original): Sergio Masuelli
Author de Traduccion a Python: Sebastian Heredia

"""

import os
import fnmatch
import time


# ---------------------------------------------------------------------------
# Búsqueda de archivos
# ---------------------------------------------------------------------------

def _find_files_in_subdirs(indir: str, pattern: str) -> list[str]:
    """
    Lista todas las subcarpetas de *indir* y dentro de cada una busca
    archivos cuyo nombre coincida con *pattern* (fnmatch).

    Estructura esperada:
        indir/
            EO_20130424_000704_.../
                EO_20130424_000704_....h5   <- este
            EO_20130424_014452_.../
                EO_20130424_014452_....h5   <- este
            ...

    Retorna lista ordenada de rutas absolutas de los archivos encontrados.
    """
    # Resolver a ruta absoluta para no depender del CWD
    indir_abs = os.path.abspath(indir)

    if not os.path.isdir(indir_abs):
        raise FileNotFoundError(
            f'El directorio de entrada no existe: {indir_abs}\n'
            f'  (indir original: {indir!r})\n'
            f'  CWD: {os.getcwd()}'
        )

    matches = []
    # Listar solo las subcarpetas directas (un nivel)
    for entry in sorted(os.scandir(indir_abs), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        # Dentro de cada subcarpeta, buscar archivos que cumplan el patrón
        for fname in os.listdir(entry.path):
            if fnmatch.fnmatch(fname, pattern):
                matches.append(os.path.join(entry.path, fname))

    return sorted(matches)


def _find_files_flat(indir: str, pattern: str) -> list[str]:
    """Busca archivos en el nivel raíz de indir (para mode='ic', archivos .mat planos)."""
    indir_abs = os.path.abspath(indir)
    if not os.path.isdir(indir_abs):
        raise FileNotFoundError(
            f'El directorio de entrada no existe: {indir_abs}\n'
            f'  (indir original: {indir!r})\n'
            f'  CWD: {os.getcwd()}'
        )
    matches = []
    for fname in os.listdir(indir_abs):
        full = os.path.join(indir_abs, fname)
        if os.path.isfile(full) and fnmatch.fnmatch(fname, pattern):
            matches.append(full)
    return sorted(matches)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def multi_passes_pro(
    indir:         str,
    oudir:         str,
    pattern:       str,
    prefix:        str,
    mode:          str = 'delta_pg',
    landmask_file: str = None,    # requerido cuando mode='delta_pg'
    recursive:     bool = False,  # True: busca en subcarpetas de indir
) -> list[float]:
    """
    Procesa cada archivo encontrado en *indir* y escribe un .mat por archivo.

    Parameters
    ----------
    indir         : Directorio de entrada (absoluto o relativo al CWD).
    oudir         : Directorio de salida (se crea si no existe).
    pattern       : Patrón fnmatch del nombre de archivo, ej. 'EO_*.h5'.
    prefix        : Prefijo del archivo de salida, ej. 'PG'.
    mode          : 'delta_pg' | 'ic'
    landmask_file : Ruta al landmask_ser.mat (obligatorio para mode='delta_pg').
    recursive     : True  -> busca en subcarpetas directas de indir.
                    False -> busca solo en el nivel raíz de indir.
    """
    if mode == 'delta_pg' and landmask_file is None:
        raise ValueError(
            "mode='delta_pg' requiere landmask_file.\n"
            "Ejemplo: landmask_file='./data/landmask/landmask_ser.mat'"
        )

    os.makedirs(os.path.abspath(oudir), exist_ok=True)

    # Selección de estrategia de búsqueda
    if recursive:
        files = _find_files_in_subdirs(indir, pattern)
    else:
        files = _find_files_flat(indir, pattern)

    n = len(files)
    if n == 0:
        raise FileNotFoundError(
            f'No se encontraron archivos con patron "{pattern}"\n'
            f'  indir resuelto: {os.path.abspath(indir)}\n'
            f'  recursive={recursive}\n'
            f'  CWD: {os.getcwd()}'
        )

    print(f'\n{"="*60}')
    print(f'  Archivos encontrados: {n}')
    print(f'  Entrada:  {os.path.abspath(indir)}')
    print(f'  Salida:   {os.path.abspath(oudir)}')
    print(f'  Modo:     {mode}')
    print(f'{"="*60}')

    processor = _get_processor(mode, landmask_file)
    times = []
    t0    = time.perf_counter()

    for i, infile in enumerate(files, start=1):
        fname_noext = os.path.splitext(os.path.basename(infile))[0]
        oufile      = os.path.join(os.path.abspath(oudir),
                                   prefix + fname_noext + '.mat')

        print(f'\n[{i}/{n}] {os.path.basename(infile)}')
        processor(infile, oufile)

        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f'  elapsed: {elapsed:.1f} s')

    print(f'\n{"="*60}')
    print(f'  Completado: {n} archivos en {times[-1]/60:.2f} min')
    print(f'{"="*60}\n')
    return times


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_processor(mode: str, landmask_file: str = None):
    """Devuelve f(infile, oufile), con landmask ligado en closure si aplica."""
    if mode == 'delta_pg':
        try:
            from Scripts.Python.delta_pg_processor import run as _run_dpg
        except ImportError:
            raise ImportError('delta_pg_processor.py no encontrado.')
        def run_dpg(infile: str, oufile: str) -> None:
            _run_dpg(infile, oufile, landmask_file)
        return run_dpg

    if mode == 'ic':
        from Scripts.Python.ic_processor import run as run_ic
        return run_ic

    raise ValueError(f'Unknown mode: {mode!r}. Use "ic" or "delta_pg".')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # Mover el CWD a la raiz del proyecto (M5-P4/) independientemente
    # de desde donde se ejecute el script.
    BASE = os.path.dirname(os.path.abspath(__file__))        # .../Scripts/Python/
    ROOT = os.path.normpath(os.path.join(BASE, '..', '..'))  # .../M5-P4/
    os.chdir(ROOT)
    print(f'CWD -> {ROOT}')

    LANDMASK = os.path.join(ROOT, 'data', 'landmask', 'landmask_ser.mat')
    L1_DIR   = os.path.join(ROOT, 'data', 'L1')
    DPG_DIR  = os.path.join(ROOT, 'data', 'Temp', 'DeltaP_G50S1v1')
    CI_DIR   = os.path.join(ROOT, 'data', 'L2', 'CI_50S')

    mode = 2

    if mode == 1:
        # Paso 1: ΔP/ΔG — H5 dentro de subcarpetas EO_.../ de L1/
        multi_passes_pro(
            indir         = L1_DIR,
            oudir         = DPG_DIR,
            pattern       = 'EO_*.h5',
            prefix        = 'PG',
            mode          = 'delta_pg',
            landmask_file = LANDMASK,
            recursive     = True,
        )

    elif mode == 2:
        # Paso 2: CI a partir de los .mat de ΔPG
        multi_passes_pro(
            indir   = DPG_DIR,
            oudir   = CI_DIR,
            pattern = 'PG*.mat',
            prefix  = 'CI',
            mode    = 'ic',
        )
