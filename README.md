# TP4 — Procesador de Concentración de Hielo (MWR SAC-D/Aquarius)

**Módulo 5 — Práctica 4**  
**Área:** ADAIP · CONAE  
**Instrumento:** MWR (Microwave Radiometer) — Misión SAC-D/Aquarius

---

## Objetivo

Implementar paso a paso el pipeline de procesamiento de datos del radiómetro
de microondas (MWR) para estimar la **Concentración de Hielo marino (CI)**
a partir de archivos L1B en formato HDF5.

El notebook `TP4_MwrIcePipeline.ipynb` tiene el pipeline completo estructurado
en secciones. En varias de ellas el código está **incompleto** y el alumno
debe completarlo para que el procesamiento funcione de punta a punta.

---

## Estructura del repositorio

```
M5-P4-Exc/
│
├── Scripts/
│   └── Python/
│       └── TP4_MwrIcePipeline.ipynb   ← notebook principal del ejercicio
│
└── data/
    ├── L1/                            ← VACÍO — ver "Datos de entrada"
    ├── L2/                            ← generado automáticamente
    ├── Temp/                          ← generado automáticamente
    └── landmask/
        └── landmask_ser.mat           ← máscara de tierra 0.5° (incluida)
```

> Las carpetas `L1/`, `L2/` y `Temp/` están vacías en el repositorio
> (solo contienen un `.gitkeep`). Los datos L1B deben descargarse por separado.

---

## Datos de entrada

Los archivos L1B del MWR tienen el siguiente formato:

```
data/L1/
├── EO_20130424_000704_CUSS_SACD_MWR_L1B_SCI_078_000_004/
│   └── EO_20130424_000704_CUSS_SACD_MWR_L1B_SCI_078_000_004.h5
├── EO_20130424_014452_CUSS_SACD_MWR_L1B_SCI_071_000_004/
│   └── EO_20130424_014452_CUSS_SACD_MWR_L1B_SCI_071_000_004.h5
└── ...
```

Cada pasada es una carpeta con un único archivo `.h5` de igual nombre.
El docente proveerá los datos o el enlace de descarga por fuera del repositorio.

---

## Instalación

```bash
pip install numpy scipy matplotlib h5py cartopy
```

En Linux (si `cartopy` falla):
```bash
sudo apt install libgeos-dev libproj-dev
pip install cartopy
```

En Windows/Mac se recomienda usar conda:
```bash
conda install cartopy
```

---

## Cómo ejecutar el notebook

1. Clonar el repositorio y ubicar los datos L1B en `data/L1/`
2. Abrir `Scripts/Python/TP4_MwrIcePipeline.ipynb`
3. Ajustar la variable `ROOT` en la celda de configuración:

```python
ROOT = os.path.abspath('../../')   # raíz del proyecto relativa al notebook
```

4. Ejecutar las celdas en orden y completar las partes marcadas

---

## Ejercicios a completar

El notebook tiene **6 puntos** donde el código está incompleto.
Están marcados con comentarios del tipo `# Completar`, `# Hay que`, etc.

---

### Ejercicio 1 — Sección 1: Mapeo de canales HDF5

Completar el diccionario `CHANNEL_MAP` que relaciona cada receiver
con los datasets del HDF5:

```python
CHANNEL_MAP = {
    'RX23H': {
        'tb_key':  ,      # ← nombre del dataset de Tb
        'lat_key': ,      # ← nombre del dataset de lat
        'lon_key': ,      # ← nombre del dataset de lon
    },
    : {                   # ← nombre del receiver 36.5 GHz H-pol
        ...
    },
    : {                   # ← nombre del receiver 36.5 GHz V-pol
        ...
    },
}
```

**Pista:** los grupos del HDF5 son `'MWR Calibrated Radiometric Data'`
y `'Geolocation Data'`. Inspeccionarlos con `f.keys()` en la celda anterior.

---

### Ejercicio 2 — Sección 1: Visualización de Tb

Completar el scatter plot de temperatura de brillo con lat/lon como ejes
y Tb como color. El esqueleto del gráfico ya está dado:

```python
# Concatenar los datos de las 8 bocinas
                                           # ← completar

# Scatter único
sc = ax.scatter()                          # ← completar argumentos
```

---

### Ejercicio 3 — Sección 3: Filtro de latitud

Completar la función `apply_lat_filter` que conserva solo los registros
fuera de la banda ecuatorial (|lat| > 49° y |lat| < 90°).
La función `_asc_des_flag` ya está implementada como referencia.

```python
def apply_lat_filter(data: dict) -> tuple[...]:
    ...
    # ← implementar el loop sobre receivers, registros y haces
```

---

### Ejercicio 4 — Sección 4: Cargar y graficar la land mask

Cargar `landmask_ser.mat`, verificar sus dimensiones y graficarla:

```python
LANDMASK = os.path.join(ROOT, 'data', 'landmask', 'landmask_ser.mat')

# ← cargar con sio.loadmat, aplicar flipud, mostrar shape
# ← graficar con imshow usando lat/lon como ejes
```

---

### Ejercicio 5 — Sección 4: Conversión lat/lon → índice de máscara

Completar `_latlon_to_mask_idx` que convierte coordenadas geográficas
a índices de la máscara de resolución 0.5°:

```python
def _latlon_to_mask_idx(lat: float, lon: float) -> tuple[int, int]:
    # La máscara tiene resolución 0.5°
    # lat ∈ [-90, 90]  →  fila  ∈ [0, 360]
    # lon ∈ [-180,180] →  col   ∈ [0, 719]  con wrap en ±180°
    ...
```

**Pista:** la fórmula es `alat = round((lat + 90) * 2)` y análoga para lon.
El operador `% MASK_LON` resuelve el wrap en la longitud ±180°.

---

### Ejercicio 6 — Sección 8: Cálculo de CI

Completar `process_ic` con la fórmula de interpolación lineal entre
tie points para calcular la Concentración de Hielo:

```
        ΔTg_obs - ΔTg_O - (ΔTp_obs - ΔTp_O) · factor
CI =   ────────────────────────────────────────────────
                       denominator
```

```python
def process_ic(dtemp: np.ndarray) -> np.ndarray:
    ...
    # ← implementar _fill_ci con la fórmula de CI
    # ← usar tie points distintos para haces pares e impares
    # ← recortar CI al intervalo [0, 1]
```

Los tie points y la función `_precompute` ya están dados como ayuda.

---

## Flujo completo del pipeline

```
data/L1/EO_*/EO_*.h5
         │
         ▼
 Ej.1  load_l1b()          →  data[receiver]['B1'..'B8'] = {Lat, Lon, Tb}
         │
         ▼
 Ej.2  Visualización Tb    →  scatter lat/lon/Tb por bocina
         │
         ▼
 Ej.3  apply_lat_filter()  →  rawlat  (|lat| > 49°, con flag asc/des)
         │
         ▼
 Ej.4  cargar landmask     →  ndarray (361, 720)  — 0=océano, 1=tierra
 Ej.5  apply_land_filter() →  rawland (sin píxeles sobre tierra)
         │
         ▼
       collocate_beams()   →  rawloc  (36 GHz ↔ 23 GHz co-localizados)
         │
         ▼
       compute_dtemp()     →  dtemp (8, 5, N)
                               [lat, lon, ΔTp, ΔTg, flag]
         │
         ▼
 Ej.6  process_ic()        →  CI (7, M)
                               [lat, lon, ΔTp, ΔTg, flag, CI, haz]
         │
         ▼
       Mapas de CI         →  IC_all_SP.png, IC_all_NP.png, ...
```

---

## Criterios de evaluación

| Ejercicio | Descripción | Puntos |
|---|---|---|
| 1 | `CHANNEL_MAP` correcto — 3 receivers con sus 3 datasets cada uno | 1 |
| 2 | Scatter plot Tb con lat/lon como ejes y colorbar | 1 |
| 3 | `apply_lat_filter` funcional — `counlat` con valores razonables | 2 |
| 4 | Land mask cargada, dimensiones verificadas y graficada | 1 |
| 5 | `_latlon_to_mask_idx` correcta incluyendo wrap de longitud | 2 |
| 6 | `process_ic` calcula CI ∈ [0,1] con tie points correctos | 3 |
| | **Total** | **10** |

---

## Material de referencia

- **Sección 1 del notebook:** exploración del HDF5 con `h5py` — ver grupos y shapes
- **Sección 2:** `_asc_des_flag` ya resuelta, sirve como ejemplo de estructura
- **Sección 5:** `collocate_beams` completa, sirve como referencia de estilo
- **Docstrings de cada función:** describen los shapes de entrada y salida esperados

---

*Módulo 5 — Práctica 4 · ADAIP · CONAE*
