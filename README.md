# Human Silhouette SVG

Pipeline de visión por computadora para convertir fotografías reales de personas en siluetas vectoriales en formato **SVG** listas para corte/grabado láser, rotulación o edición vectorial.

El problema de fondo no se resuelve bien aplicando un detector de bordes clásico (Canny/Sobel) ni vectorizando una fotografía directamente con Potrace: las texturas de ropa, sombras y fondos complejos generan miles de trazos irrelevantes. Para resolverlo, el proyecto descompone la tarea en cuatro etapas: segmentación de la persona con **SAM (ViT-H)**, traducción imagen-a-trazo mediante una **U-Net** entrenada con pérdida L1 + perceptual (VGG-19), limpieza morfológica con esqueletización y conversión final a rutas SVG.

---

## Arquitectura del pipeline

El flujo principal vive en `src/main.py` (`run_pipeline`) y se reutiliza tanto en el modo de escritorio interactivo como en la API web:

```text
Foto RGB (archivo o buffer)
   │
   ▼
1. Preprocesamiento (src/preprocessing.py)
   ├─ Segmentación del sujeto con SAM (ViT-H): manual por clics o automática
   ├─ Aislamiento sobre fondo negro (bitwise AND)
   ├─ Letterbox proporcional a 512x512
   └─ Normalización a tensor [3, 512, 512] en [-1, 1]
   │
   ▼
2. Inferencia de trazos (UNet en src/training.py)
   ├─ Encoder-Decoder simétrico (64 → 128 → 256 → 512 → 1024) con skip connections
   └─ Salida continua [1, 512, 512] en [-1, 1] (activación Tanh)
   │
   ▼
3. Postprocesamiento (src/postprocessing.py)
   ├─ Estiramiento de contraste por percentiles (p1 - p99)
   ├─ Umbralizado adaptativo gaussiano (block_size=31, C=-5)
   ├─ Filtrado de componentes conexas (área >= 25 px)
   ├─ Cierre morfológico elíptico (3x3)
   └─ Esqueletización / thinning (cv2.ximgproc.thinning)
   │
   ▼
4. Vectorización (src/vectorization.py)
   ├─ Suavizado raster mediante Potrace (--pgm)
   ├─ Extracción de contornos (cv2.findContours) y filtrado (área >= 8.0)
   ├─ Simplificación poligonal Douglas-Peucker (epsilon = 0.0015 * perímetro)
   └─ Exportación de <svg> con trazados (M, L, Z)
```

---

## Decisiones técnicas

- **Separar segmentación y generación de silueta (SAM + U-Net):** entrenar una U-Net directa desde fotos sin recortar exige un volumen de datos alto para que aprenda a ignorar fondos arbitrarios. Al usar SAM ViT-H como etapa previa para eliminar el fondo, la U-Net solo necesita aprender la correspondencia entre persona recortada y trazo, lo que permitió entrenar con un dataset propio reducido (72 pares en entrenamiento).
- **Letterboxing a 512x512 en lugar de resize directo:** mantener la relación de aspecto rellenando con fondo negro evita deformar proporciones corporales entre fotos verticales y horizontales.
- **Pérdida combinada (`L1 + 0.1 * PerceptualLoss`):** entrenar solo con L1 o MSE en tareas de *line-art* tiende a difuminar o romper líneas finas cuando existe un desplazamiento mínimo respecto al target. Agregar una pérdida perceptual sobre las primeras 16 capas congeladas de VGG-19 (pesos `IMAGENET1K_V1`) penaliza diferencias estructurales y mejora la continuidad del trazo.
- **Postprocesamiento clásico antes de vectorizar:** la salida de la U-Net es un mapa continuo en `[-1, 1]`, no una máscara binaria limpia. La combinación de normalización por percentiles, umbral adaptativo, limpieza por área y *thinning* estandariza el grosor del trazo a 1 píxel antes de pasar por Potrace y `approxPolyDP`.
- **Herramienta propia de etiquetado (`src/dataset_builder.py`):** al no contar con un dataset público que encajara con el estilo de trazo buscado, se construyó una utilidad con Tkinter + OpenCV + SAM para generar pares alineados y persistir el progreso en `data/preprocessing_state.json`.

---

## Modelo y entrenamiento

Todo el flujo de entrenamiento está en `src/training.py`.

### Arquitectura U-Net

| Bloque | Operación | Canales | Resolución espacial |
| :--- | :--- | :---: | :---: |
| Encoder (x4) | `2x (Conv3x3 → BatchNorm → ReLU)` + `MaxPool2d(2)` | `3 → 64 → 128 → 256 → 512` | `512 → 32` |
| Bottleneck | `2x (Conv3x3 → BatchNorm → ReLU)` | `512 → 1024` | `32x32` |
| Decoder (x4) | `ConvTranspose2d(2)` + `concat(skip)` + `ConvBlock` | `1024 → 512 → 256 → 128 → 64` | `32 → 512` |
| Salida | `Conv1x1` + `Tanh` | `64 → 1` | `512x512` |

El `state_dict` entrenado (`checkpoints/best_model.pth`) ocupa ~142 MB en `float32`.

### Configuración de entrenamiento

| Parámetro | Valor |
| :--- | :--- |
| Resolución de entrada / salida | `512x512` RGB → `512x512` escala de grises |
| Normalización | `[-1, 1]` (`mean=0.5, std=0.5`) |
| Función de pérdida | `L1Loss + 0.1 * PerceptualLoss(VGG19[:16])` |
| Optimizador | `Adam` (`lr = 1e-3`) |
| Scheduler | `ReduceLROnPlateau(mode="min", factor=0.1, patience=10)` |
| Batch size / Épocas máx. | `8` / `500` |
| Early stopping | `patience = 5` sobre pérdida de validación |
| Checkpoints | Cada `5` épocas (`checkpoints/checkpoint_epoch_<N>.pth`) + `best_model.pth` |

### Data Augmentation

Se aplica de forma sincronizada sobre imagen y máscara usando `albumentations` únicamente en el conjunto de entrenamiento:
- Rotación aleatoria (`±10°`, `p=0.5`).
- Transformación afín: escala `(0.9, 1.1)` y traslación `(-5%, +5%)` (`p=0.5`).
- Variación aleatoria de brillo y contraste (`±15%`, `p=0.5`).

El entrenamiento reanuda automáticamente desde el checkpoint con mayor número de época encontrado en `checkpoints/`.

---

## Estructura del repositorio

```text
human_silhouette_svg/
├── bin/
│   └── potrace.exe               # Binario de Potrace para Windows
├── checkpoints/                  # Pesos de U-Net (best_model.pth, model_final.pth, checkpoints)
├── data/
│   ├── input_train/              # Imágenes segmentadas RGB 512x512 (72 muestras actuales)
│   ├── output_train/             # Trazos objetivo en escala de grises (72 muestras actuales)
│   ├── input_val/                # Imágenes de validación
│   ├── output_val/               # Targets de validación (requerido para ejecutar training.py)
│   ├── input_segmentation/       # Fotos en bruto usadas por dataset_builder.py
│   └── output_segmentation/      # Targets en bruto usados por dataset_builder.py
├── frontend/
│   ├── index.html                # Interfaz web (Tailwind CSS via CDN)
│   └── app.js                    # Lógica de carga drag & drop, llamada a /api/vectorize y descarga
├── model_u2net/                  # Artefacto de pruebas previas de segmentación (no usado en pipeline)
├── outputs/                      # Salidas intermedias (preprocessing, U_Net, postprocessing) y SVGs
├── segment_anything/             # Código fuente de SAM (Meta AI) + checkpoint sam_vit_h_4b8939.pth
├── src/
│   ├── api.py                    # Servidor FastAPI y servicio de archivos del frontend
│   ├── config.py                 # Rutas centralizadas y detección de Potrace
│   ├── dataset_builder.py        # Herramienta GUI para generar pares de entrenamiento con SAM
│   ├── main.py                   # Pipeline unificado (run_pipeline) y entrada de escritorio
│   ├── postprocessing.py         # Contraste, binarización adaptativa, limpieza y thinning
│   ├── preprocessing.py          # Carga de SAM, segmentación (interactiva/auto) y letterboxing
│   ├── training.py               # Dataset, arquitectura U-Net, CombinedLoss y loop de entrenamiento
│   ├── utils.py                  # Transformaciones de entrada/salida y desnormalización
│   └── vectorization.py          # Integración con Potrace, contornos y escritura SVG
├── yolov8n-seg.pt                # Peso de pruebas previas de segmentación (no usado en pipeline)
└── requirements.txt
```

> Nota: `model_u2net/` y `yolov8n-seg.pt` corresponden a experimentos iniciales de segmentación previos a estandarizar en SAM ViT-H; el pipeline activo no los importa.

---

## Instalación

### Requisitos previos

- **Python:** 3.10 – 3.12 (recomendado por compatibilidad con PyTorch y `opencv-contrib-python`).
- **GPU (opcional, recomendada para entrenamiento):** NVIDIA con soporte CUDA 12.1.
- **Potrace:**
  - En Windows se usa `bin/potrace.exe` automáticamente.
  - En Linux/macOS debe estar disponible en el `PATH` (`sudo apt install potrace` / `brew install potrace`); `src/config.py` lo detecta automáticamente con `shutil.which("potrace")`.

### Dependencias

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

Si vas a levantar la interfaz web (`src/api.py`), instala además las dependencias del servidor que no están declaradas en `requirements.txt`:

```bash
pip install fastapi uvicorn python-multipart pillow
```

### Pesos necesarios

Antes de ejecutar inferencia, verifica que existan ambos archivos:

1. **SAM ViT-H:** `segment_anything/sam_vit_h_4b8939.pth` (descargable desde el repositorio oficial de [Segment Anything](https://github.com/facebookresearch/segment-anything#model-checkpoints)).
2. **U-Net entrenada:** `checkpoints/best_model.pth` (generado por `src/training.py`).

---

## Uso

### 1. Servidor web y API REST

Levanta FastAPI cargando U-Net y SAM una sola vez al inicio:

```bash
python src/api.py
```

O con Uvicorn directamente:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

- Interfaz web: `http://127.0.0.1:8000`
- Documentación OpenAPI (Swagger): `http://127.0.0.1:8000/docs`

Ejemplo de llamada directa al endpoint de inferencia:

```bash
curl -X POST "http://127.0.0.1:8000/api/vectorize" \
  -F "file=@ruta/a/foto.jpg"
```

Respuesta:

```json
{
  "status": "success",
  "filename": "silhouette_a1b2c3d4.svg",
  "svg": "<svg xmlns=\"http://www.w3.org/2000/svg\" ...>...</svg>"
}
```

### 2. Modo de escritorio (segmentación interactiva)

Útil cuando la foto tiene varios sujetos o el fondo complica la segmentación automática:

```bash
python src/main.py
```

1. Selecciona la imagen en el diálogo de Tkinter.
2. En la ventana de OpenCV, haz **clic izquierdo** sobre las regiones de la persona para ir sumando máscaras con SAM.
   - `Z`: deshacer la última máscara.
   - `S`: confirmar máscara y continuar el pipeline.
   - `ESC`: cancelar.
3. El resultado final se guarda en `outputs/vectorization/<nombre>.svg` y las imágenes de depuración de cada etapa en `outputs/{preprocessing,U_Net,postprocessing}/`.

### 3. Construcción de pares de entrenamiento

```bash
python src/dataset_builder.py
```

Guarda pares sincronizados en `data/input_train/` y `data/output_train/` con numeración incremental (`imagen_1.png`, `imagen_2.png`, ...) manteniendo el último índice procesado en `data/preprocessing_state.json`.

### 4. Entrenamiento de la U-Net

Asegúrate de tener pares `.png` con nombres idénticos en `data/input_train/`, `data/output_train/`, `data/input_val/` y `data/output_val/` (en `input_val` solo deben existir `.png` válidos con su par correspondiente en `output_val`):

```bash
python src/training.py
```

---

## Limitaciones conocidas y próximos pasos

Puntos concretos identificados en el estado actual del código para la siguiente iteración:

- **Acoplamiento de arquitectura y entrenamiento:** `UNet` está definida dentro de `src/training.py`, lo que obliga a `src/main.py` a importar dependencias de entrenamiento (`albumentations`, `vgg19`) en inferencia. Conviene extraer `UNet` y sus bloques a `src/model.py`.
- **Conflicto entre scheduler y early stopping:** `EARLY_STOPPING_PATIENCE = 5` detiene el entrenamiento antes de que `ReduceLROnPlateau(patience=10)` pueda reducir el learning rate. Hay que invertir esa relación (por ejemplo, `scheduler.patience = 4` y `early_stopping = 10`).
- **Dispositivo de SAM y prompts automáticos:** `load_sam()` en `src/preprocessing.py` no mueve el modelo a GPU (`sam.to(device)`), por lo que corre en CPU incluso con CUDA disponible. Además, el modo automático inyecta 3 puntos fijos sobre el eje central asumiendo una sola persona centrada; integrar un detector de personas (como YOLOv8) para pasar un *bounding box* como prompt a SAM mejoraría la robustez en encuadres descentrados.
- **Concurrencia y calidad vectorial en `src/vectorization.py`:** actualmente Potrace se usa solo como suavizado intermedio (`--pgm`) y los trazados SVG finales son polilíneas (`M`, `L`, `Z`). Exportar curvas Bézier cúbicas directamente desde Potrace (`potrace -s`) reduciría nodos y mejoraría la suavidad para corte láser; además, los archivos temporales deberían crearse con `tempfile` para evitar colisiones ante peticiones concurrentes en la API.
- **Tamaño del dataset y reproducibilidad:** con 72 pares de entrenamiento la generalización en poses inusuales u oclusiones es limitada. Ampliar el dataset, fijar semillas aleatorias, agregar métricas cuantitativas en validación (IoU / Dice sobre trazos binarizados) y completar `requirements.txt` con las dependencias de FastAPI son prioridades directas.
