"""
load_l1b.py
===========
Lee un archivo HDF5 del MWR L1B (SAC-D/Aquarius) y organiza los datos
en la estructura anidada que espera delta_pg_processor.py:

    data[receiver][beam_number] = {'Lat': array_1D, 'Lon': array_1D, 'Tb': array_1D}

    receiver      beam_number   fuente en el HDF5
    ---------     -----------   ------------------------------------------
    'RX23H'       1 … 8         k_h_antenna_temperature / k_h_lat/lon   [:,j-1]
    'RX37H'       1 … 8         ka_h_antenna_temperature / ka_h_lat/lon [:,j-1]
    'RX37V'       1 … 8         ka_v_antenna_temperature / ka_v_lat/lon [:,j-1]

Mapeo de nombres HDF5 → receiver
---------------------------------
    'k_h_antenna_temperature'  →  RX23H   (23 GHz H-pol)
    'ka_h_antenna_temperature' →  RX37H   (36.5 GHz H-pol)
    'ka_v_antenna_temperature' →  RX37V   (36.5 GHz V-pol)
"""

import h5py
import numpy as np

NUMBEAM = 8

# Mapeo: receiver → (dataset de temperatura, dataset de lat, dataset de lon)
# Todas las claves son del grupo 'MWR Calibrated Radiometric Data' y
# 'Geolocation Data' respectivamente.
CHANNEL_MAP = {
    'RX23H': {
        'tb_key':  'k_h_antenna_temperature',
        'lat_key': 'k_h_latitude',
        'lon_key': 'k_h_longitude',
    },
    'RX37H': {
        'tb_key':  'ka_h_antenna_temperature',
        'lat_key': 'ka_h_latitude',
        'lon_key': 'ka_h_longitude',
    },
    'RX37V': {
        'tb_key':  'ka_v_antenna_temperature',
        'lat_key': 'ka_v_latitude',
        'lon_key': 'ka_v_longitude',
    },
}


def load_l1b(h5file: str) -> dict:
    """
    Carga el HDF5 y devuelve el dict anidado para delta_pg_processor.

    Parámetros
    ----------
    h5file : str
        Ruta al archivo .h5 del L1B.

    Retorna
    -------
    data : dict
        data[receiver][beam_number] = {'Lat': ndarray, 'Lon': ndarray, 'Tb': ndarray}
        Cada array tiene shape (N,) — un valor por registro temporal.
        beam_number va de 1 a 8 (igual que en el MATLAB original).

    Ejemplo de acceso
    -----------------
        data['RX23H'][1]['Tb']    # Tb 23 GHz H-pol, bocina 1  ← kh_temp[:,0]
        data['RX37H'][3]['Lat']   # lat 36.5 GHz H-pol, bocina 3 ← kah_lat[:,2]
        data['RX37V'][8]['Lon']   # lon 36.5 GHz V-pol, bocina 8 ← kav_lon[:,7]
    """
    data = {}

    with h5py.File(h5file, 'r') as f:
        rad_grp = f['MWR Calibrated Radiometric Data']
        geo_grp = f['Geolocation Data']

        for receiver, keys in CHANNEL_MAP.items():
            # Cargar matrices completas (shape: N_registros × 8 bocinas)
            tb_mat  = rad_grp[keys['tb_key']][:]   # (N, 8)
            lat_mat = geo_grp[keys['lat_key']][:]  # (N, 8)
            lon_mat = geo_grp[keys['lon_key']][:]  # (N, 8)

            data[receiver] = {}

            for beam_num in range(1, NUMBEAM + 1):
                col = beam_num - 1    # índice 0-based de la columna
                data[receiver][f'B{beam_num}'] = {
                    'Tb' : tb_mat [:, col].astype(float),
                    'Lat': lat_mat[:, col].astype(float),
                    'Lon': lon_mat[:, col].astype(float),
                }

    return data
