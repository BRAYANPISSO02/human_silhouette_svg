# AI Vectorizer Pro — Human Silhouette SVG Generator

Sistema integral de visión por computadora y Deep Learning para la extracción y vectorización automática de siluetas humanas en formato **SVG (Scalable Vector Graphics)** a partir de fotografías reales.

El proyecto implementa un pipeline híbrido de múltiples etapas que combina **Segment Anything Model (SAM)** de Meta para la segmentación de sujetos, una **red neuronal convolucional U-Net personalizada** con funciones de pérdida perceptuales para la generación de siluetas y arte de líneas (*line-art*), filtros morfológicos avanzados y vectorización basada en **Potrace** y simplificación de contornos geométricos.

Incluye tanto una **aplicación web moderna (FastAPI + SPA en Tailwind CSS)** como una **herramienta de escritorio interactiva**, un generador asistido de datasets y un entorno completo de entrenamiento en PyTorch.

---

## Tabla de Contenidos
1. [Características Principales](#características-principales)
2. [Arquitectura del Pipeline](#arquitectura-del-pipeline)
   - [Paso 1: Preprocesamiento y Segmentación (SAM)](#paso-1-preprocesamiento-y-segmentación-sam)
   - [Paso 2: Inferencia de Silueta con U-Net](#paso-2-inferencia-de-silueta-con-u-net)
   - [Paso 3: Postprocesamiento y Esqueletización](#paso-3-postprocesamiento-y-esqueletización)
   - [Paso 4: Suavizado y Vectorización SVG](#paso-4-suavizado-y-vectorización-svg)
3. [Módulos y Herramientas del Proyecto](#módulos-y-herramientas-del-proyecto)
   - [Servidor Web y API REST (`src/api.py` + `frontend/`)](#1-servidor-web-y-api-rest-srcapipy--frontend)
   - [Pipeline de Escritorio Interactivo (`src/main.py`)](#2-pipeline-de-escritorio-interactivo-srcmainpy)
   - [Generador Asistido de Dataset (`src/dataset_builder.py`)](#3-generador-asistido-de-dataset-srcdataset_builderpy)
   - [Pipeline de Entrenamiento (`src/training.py`)](#4-pipeline-de-entrenamiento-srctrainingpy)
   - [Modelos Adicionales de Soporte](#5-modelos-adicionales-de-soporte)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Instalación y Requisitos](#instalación-y-requisitos)
   - [Requisitos Previos](#requisitos-previos)
   - [Instalación de Dependencias](#instalación-de-dependencias)
   - [Pesos y Checkpoints Requeridos](#pesos-y-checkpoints-requeridos)
6. [Guía de Ejecución](#guía-de-ejecución)
   - [1. Levantar el Servicio Web](#1-levantar-el-servicio-web)
   - [2. Ejecutar la Aplicación de Escritorio](#2-ejecutar-la-aplicación-de-escritorio)
   - [3. Construir Dataset de Pares](#3-construir-dataset-de-pares)
   - [4. Entrenar la Red U-Net](#4-entrenar-la-red-u-net)
7. [Proyección de Despliegue en la Nube (AWS)](#proyección-de-despliegue-en-la-nube-aws)

---

## Características Principales

- **Segmentación Robusta con SAM**: Aislamiento del sujeto respecto a fondos complejos utilizando Segment Anything (`vit_h`), operable en modo interactivo asistido por el usuario (OpenCV GUI) o en modo totalmente automático (mediante puntos clave heurísticos).
- **Traducción Imagen-a-Silueta mediante U-Net**: Modelo U-Net entrenado para traducir fotos RGB segmentadas (512×512) a mapas continuos de silueta y trazos.
- **Entrenamiento con Pérdida Perceptual (VGG-19)**: Optimización guiada por una función de pérdida combinada (`L1Loss` + `PerceptualLoss` con características profundas de VGG-19 preentrenada), preservando la coherencia estructural y los bordes humanos.
- **Postprocesamiento Morfológico Riguroso**: Normalización adaptativa por percentiles, umbralizado adaptativo gaussiano, filtrado de componentes conexas por área mínima (`MIN_AREA = 25`), cierre morfológico y adelgazamiento de trazo (*skeletonization / thinning*).
- **Salida Vectorial SVG Optimizada**: Curvas suavizadas con Potrace y simplificadas geométricamente con el algoritmo de Douglas-Peucker (`cv2.approxPolyDP`), generando rutas SVG limpias (`M`, `L`, `Z`) ideales para grabado/corte láser, rotulación o ilustración digital.
- **Doble Modo de Operación**:
  - **Web SPA**: Interfaz en Tailwind CSS con drag & drop, previsualización en tiempo real y descarga directa del archivo `.svg`.
  - **Desktop GUI**: Interfaz con Tkinter y OpenCV para control fino manual sobre la máscara de segmentación.

---

## Arquitectura del Pipeline

El flujo de procesamiento unificado (`src/main.py` -> `run_pipeline`) consta de cuatro etapas deterministas y reproducibles:

```
Imagen de Entrada (RGB)
        │
        ▼
[ Paso 1: Preprocesamiento & Segmentación SAM ]
  ├─ Detección / Selección de Persona (SAM ViT-H)
  ├─ Extracción por máscara binaria (Bitwise AND)
  ├─ Redimensionamiento con Letterbox (512×512)
  └─ Normalización a Tensor [-1.0, 1.0]
        │
        ▼
[ Paso 2: Inferencia de Silueta (U-Net) ]
  ├─ Red Neuronal U-Net (Encoder-Decoder con Skip Connections)
  └─ Salida con activación Tanh (tensor 1×512×512 en [-1.0, 1.0])
        │
        ▼
[ Paso 3: Postprocesamiento & Morfología ]
  ├─ Estiramiento de contraste por percentiles (p1 - p99)
  ├─ Binarización adaptativa gaussiana
  ├─ Filtrado de ruido por componentes conexas (área >= 25 px)
  ├─ Cierre morfológico elíptico (3×3)
  └─ Esqueletización / Adelgazamiento (cv2.ximgproc.thinning)
        │
        ▼
[ Paso 4: Vectorización SVG ]
  ├─ Pre-suavizado con Potrace (binario externo)
  ├─ Extracción de contornos (cv2.findContours)
  ├─ Filtrado por área mínima (área >= 8.0)
  ├─ Simplificación poligonal Douglas-Peucker (approxPolyDP)
  └─ Generación y serialización del archivo SVG (<svg><path d="..."/></svg>)
```

### Paso 1: Preprocesamiento y Segmentación (SAM)
- **Carga de imagen**: Lee desde ruta en disco o desde un buffer en memoria (`cv2.imdecode`).
- **Segmentación**:
  - *Modo interactivo*: Abre ventana OpenCV permitiendo clics para incluir regiones, tecla `Z` para deshacer (*undo*), tecla `S` para confirmar y `ESC` para cancelar.
  - *Modo automático*: Inyecta puntos guía estratégicos (centro y tercios verticales de la imagen) y selecciona la máscara con el mayor puntaje de confianza.
- **Letterboxing**: Escala proporcionalmente la imagen segmentada manteniendo la relación de aspecto dentro de un lienzo de 512×512 píxeles con fondo neutro.
- **Transformación**: Convierte a RGB, tensor PyTorch y normaliza a valores entre -1 y 1 (`mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`).

### Paso 2: Inferencia de Silueta con U-Net
- **Arquitectura U-Net**:
  - **Encoder**: 4 bloques descendentes dobles (Conv3x3 + BatchNorm + ReLU) seguidos de MaxPooling (512×512 → 256×256 → 128×128 → 64×64 → 32×32) con canales progresivos (64 → 128 → 256 → 512).
  - **Cuello de Botella (Bottleneck)**: Bloque convolucional profundo de 1024 canales en resolución 32×32.
  - **Decoder**: 4 bloques ascendentes con convoluciones transpuestas (`ConvTranspose2d`), concatenación de *skip connections* del encoder y bloques de convolución doble (canales 512 → 256 → 128 → 64).
  - **Capa de Salida**: Convolución 1×1 y activación `Tanh`, entregando un tensor de 1 canal en escala `[-1, 1]` que define la silueta.
- Los pesos se cargan automáticamente desde `checkpoints/best_model.pth`.

### Paso 3: Postprocesamiento y Esqueletización
- **Normalización**: Escala el tensor a NumPy `uint8` en el rango `[0, 255]`.
- **Contraste Dinámico**: Ajuste automático de brillo/contraste usando los percentiles 1% y 99%.
- **Binarización**: Umbralizado adaptativo gaussiano con tamaño de bloque 31 y constante C = -5.
- **Limpieza de Ruido**: Análisis de componentes conexas de 8 vecinos para eliminar artefactos aislados menores a 25 píxeles de área.
- **Refinamiento Morfológico**: Cierre morfológico con kernel elíptico para consolidar líneas interrumpidas.
- **Thinning**: Aplicación de algoritmo de adelgazamiento (esqueletización) de OpenCV para garantizar trazos de un solo píxel de ancho.

### Paso 4: Suavizado y Vectorización SVG
- **Potrace**: Invoca el binario de Potrace para suavizar curvas continuas a partir del mapa de bits procesado.
- **Extracción de Contornos**: Detección jerárquica con `cv2.findContours`.
- **Filtrado y Simplificación**:
  - Se descartan contornos con área menor a 8.0.
  - Se simplifican las polilíneas mediante `cv2.approxPolyDP` con épsilon adaptativo relativo al perímetro ($\epsilon = 0.0015 \times \text{perímetro}$).
- **Escritura SVG**: Construcción directa de instrucciones vectoriales (`M`, `L`, `Z`) dentro de un elemento `<svg>` con `viewBox` acorde a la resolución original y estilos listos para renderizado o corte láser.

---

## Módulos y Herramientas del Proyecto

### 1. Servidor Web y API REST (`src/api.py` + `frontend/`)
Backend desarrollado con **FastAPI** que aloja el servicio de inferencia en tiempo real y sirve la interfaz web:
- **Carga de Modelos en Memoria**: Inicializa SAM (`vit_h`) y U-Net en GPU (si CUDA está disponible) o CPU al arrancar el servidor, evitando latencia por recarga de modelos.
- **Endpoint `POST /api/vectorize`**:
  - Recibe un archivo de imagen en formato multiparte (`UploadFile`).
  - Ejecuta el pipeline en modo automático (sin GUI de escritorio).
  - Devuelve una respuesta JSON con el estado, el nombre del archivo y el código SVG serializado:
    ```json
    {
      "status": "success",
      "filename": "silhouette_a1b2c3d4.svg",
      "svg": "<svg xmlns=\"http://www.w3.org/2000/svg\" ... </svg>"
    }
    ```
- **Frontend SPA (`frontend/index.html`, `frontend/app.js`)**:
  - Diseño responsivo y moderno basado en Tailwind CSS.
  - Zona de arrastrar y soltar (*drag & drop*) de imágenes (JPG, PNG, WEBP).
  - Vista comparativa en dos columnas: previsualización de la foto original vs. visualizador SVG interactivo.
  - Botón de descarga directa del SVG (`silueta_vectorial.svg`).
  - Botón para reiniciar la interfaz y procesar nuevas imágenes.

### 2. Pipeline de Escritorio Interactivo (`src/main.py`)
Punto de entrada para uso local interactivo:
- Diálogo gráfico con **Tkinter** para seleccionar la imagen de entrada desde el explorador de archivos.
- Permite la segmentación manual asistida por SAM en ventana OpenCV.
- Guarda opcionalmente las etapas intermedias de depuración en:
  - `outputs/preprocessing/`
  - `outputs/U_Net/`
  - `outputs/postprocessing/`
  - `outputs/vectorization/` (archivo SVG final)

### 3. Generador Asistido de Dataset (`src/dataset_builder.py`)
Herramienta diseñada para acelerar la construcción y curación de pares de imágenes para el entrenamiento de la U-Net:
- Diálogo interactivo para continuar desde el último índice guardado (persistido en `data/preprocessing_state.json`) o elegir un número personalizado.
- Carga de imágenes de entrada desde `data/input_segmentation/` o explorador de archivos.
- Segmentación con SAM por clics (`clic izquierdo`: agregar región, `Z`: deshacer, `S`: guardar par procesado, `ESC`: salir guardando estado).
- Exporta de forma organizada a:
  - `data/input_train/imagen_X.png` (imagen recortada y con letterbox 512×512)
  - `data/output_train/imagen_X.png` (máscara / arte de línea objetivo)

### 4. Pipeline de Entrenamiento (`src/training.py`)
Módulo de entrenamiento completo con PyTorch:
- **Dataset (`SilhouetteDataset`)**: Empareja imágenes RGB con sus respectivas imágenes objetivo en escala de grises.
- **Data Augmentation (`albumentations`)**:
  - Rotaciones aleatorias (`limit=10, p=0.5`).
  - Transformaciones afines (`scale=(0.9, 1.1)`, `translate_percent=(-0.05, 0.05)`).
  - Variaciones aleatorias de brillo y contraste (`limit=0.15, p=0.5`).
- **Función de Pérdida (`CombinedLoss`)**:
  - Pérdida L1 a nivel de píxeles: $\mathcal{L}_{\text{L1}} = \| y_{\text{pred}} - y_{\text{true}} \|_1$
  - Pérdida Perceptual basada en VGG-19 (primeras 16 capas de features):
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{L1}} + 0.1 \times \mathcal{L}_{\text{perceptual}}$$
- **Optimizador y Regularización**:
  - Optimizador Adam ($lr = 0.001$).
  - `ReduceLROnPlateau` con factor 0.1 y paciencia de 10 épocas.
  - Early Stopping con paciencia de 5 épocas sin mejora en validación.
  - Guardado de checkpoints periódicos cada 5 épocas en `checkpoints/checkpoint_epoch_*.pth`, guardado del mejor modelo en `checkpoints/best_model.pth` y guardado final en `checkpoints/model_final.pth`.

### 5. Modelos Adicionales de Soporte
- **U2-Net (`model_u2net/`)**: Implementación completa de la arquitectura U2-Net y checkpoint `u2net_human_seg.pth` orientado a la segmentación de figuras humanas.
- **YOLOv8-Seg (`yolov8n-seg.pt`)**: Modelo preentrenado de Ultralytics disponible en la raíz del proyecto para tareas alternativas de segmentación por instancias.
- **Potrace (`bin/potrace.exe`)**: Utilidad de línea de comandos para transformar bitmaps en trazados de curvas bezier de alta fidelidad.

---

## Estructura del Proyecto

```text
human_silhouette_svg/
├── bin/
│   └── potrace.exe             # Binario ejecutable de Potrace para vectorización
├── checkpoints/                # Modelos y pesos entrenados de la U-Net
│   ├── best_model.pth          # Checkpoint con mejor loss de validación
│   ├── checkpoint_epoch_*.pth  # Checkpoints periódicos de entrenamiento
│   └── model_final.pth         # Estado del modelo al finalizar las épocas
├── data/                       # Almacenamiento local de datasets
│   ├── input_segmentation/     # Imágenes en bruto a segmentar
│   ├── output_segmentation/    # Máscaras brutas de segmentación
│   ├── input_train/            # Imágenes de entrenamiento (RGB 512×512)
│   ├── output_train/           # Siluetas objetivo de entrenamiento
│   ├── input_val/              # Imágenes de validación
│   └── preprocessing_state.json # Persistencia del último índice etiquetado
├── frontend/                   # Interfaz de usuario Web (Single Page App)
│   ├── index.html              # Estructura HTML5 con Tailwind CSS
│   └── app.js                  # Lógica cliente (drag & drop, peticiones fetch, render SVG)
├── model_u2net/                # Arquitectura y pesos de U2-Net
│   ├── u2net.py                # Definición de capas y bloques RSU de U2-Net
│   └── u2net_human_seg/
│       └── u2net_human_seg.pth # Checkpoint de U2-Net para segmentación humana
├── outputs/                    # Artefactos generados por el pipeline
│   ├── preprocessing/          # Imágenes intermedias segmentadas y escaladas
│   ├── U_Net/                  # Predicciones intermedias de la U-Net
│   ├── postprocessing/         # Imágenes binarizadas y esqueletizadas
│   └── vectorization/          # Archivos vectoriales finales (.svg)
├── segment_anything/           # Módulo local de Segment Anything (Meta AI)
│   ├── modeling/               # Componentes de SAM (Image Encoder, Prompt Encoder, Mask Decoder)
│   ├── utils/                  # Utilidades y transformaciones geométricas de SAM
│   ├── build_sam.py            # Constructores de modelos SAM (ViT-H, ViT-L, ViT-B)
│   ├── predictor.py            # Clase SamPredictor para inferencia
│   └── sam_vit_h_4b8939.pth    # Checkpoint preentrenado de SAM ViT-H (~2.4 GB)
├── src/                        # Código fuente principal
│   ├── config.py               # Rutas base, constantes del proyecto y directorios
│   ├── utils.py                # Transformaciones torchvision y desnormalización de tensores
│   ├── preprocessing.py        # Carga de SAM, segmentación (interactiva/auto) y letterboxing
│   ├── training.py             # Arquitectura U-Net, Dataset, PerceptualLoss y bucle de train
│   ├── dataset_builder.py      # GUI para etiquetado y construcción asistida de dataset
│   ├── postprocessing.py       # Limpieza morfológica, umbralizado adaptativo y thinning
│   ├── vectorization.py        # Enlace con Potrace, detección y simplificación de contornos SVG
│   ├── main.py                 # Pipeline unificado y punto de entrada de escritorio (CLI/GUI)
│   └── api.py                  # Servidor backend FastAPI y endpoints REST
├── yolov8n-seg.pt              # Checkpoint de YOLOv8 Nano Segmentación
├── requirements.txt            # Dependencias del entorno Python
└── README.md                   # Documentación detallada del proyecto
```

---

## Instalación y Requisitos

### Requisitos Previos
- **Sistema Operativo**: Windows 10/11 o Linux.
- **Python**: 3.10 a 3.12 recomendado.
- **Aceleración GPU (Opcional pero muy recomendada)**: Tarjeta gráfica NVIDIA con controladores actualizados y soporte para CUDA 12.1.
- **Potrace**: En Windows el binario se incluye en `bin/potrace.exe`. En entornos Linux, asegurarse de disponer de Potrace en el PATH del sistema o configurar su ruta en `src/config.py`.

### Instalación de Dependencias

1. **Crear y activar un entorno virtual**:
   ```bash
   python -m venv venv

   # En Windows:
   venv\Scripts\activate

   # En Linux / macOS:
   source venv/bin/activate
   ```

2. **Instalar los paquetes requeridos**:
   ```bash
   pip install -r requirements.txt
   ```

   *Nota de dependencias clave*:
   - PyTorch con soporte CUDA (`torch`, `torchvision`, `torchaudio`).
   - OpenCV (`opencv-contrib-python==4.10.0.84`) para operaciones de ventana y módulo `ximgproc`.
   - `albumentations` para aumento de datos durante el entrenamiento.
   - `fastapi` y `uvicorn` para el servicio web.
   - `numpy<2.0.0` para garantizar compatibilidad con librerías de visión y aprendizaje profundo.
   - `ultralytics` para modelos complementarios YOLO.

### Pesos y Checkpoints Requeridos
Asegurarse de que los archivos de pesos estén ubicados en sus rutas correspondientes:
- **SAM ViT-H**: `segment_anything/sam_vit_h_4b8939.pth`
- **U-Net Best Model**: `checkpoints/best_model.pth`
- **U2-Net (Opcional)**: `model_u2net/u2net_human_seg/u2net_human_seg.pth`

---

## Guía de Ejecución

### 1. Levantar el Servicio Web
Para iniciar el servidor FastAPI y acceder a la interfaz web:

```bash
python src/api.py
```
o mediante Uvicorn directamente:
```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Una vez iniciado:
- Abre el navegador en: `http://127.0.0.1:8000`
- Documentación interactiva de la API (Swagger UI): `http://127.0.0.1:8000/docs`

En la web podés arrastrar una foto; el backend la segmentará automáticamente con SAM, la procesará con U-Net, realizará el postprocesamiento morfológico y devolverá el SVG en pantalla listo para descargar.

### 2. Ejecutar la Aplicación de Escritorio
Para utilizar el pipeline local con selector de archivos y segmentación manual interactiva:

```bash
python src/main.py
```
1. Se abrirá una ventana para seleccionar la imagen deseada.
2. Si la segmentación es interactiva, hacé clic sobre el cuerpo de la persona en la ventana de OpenCV.
3. Presioná `S` para confirmar la máscara.
4. El proceso generará el archivo SVG en `outputs/vectorization/` y guardará los pasos intermedios en `outputs/`.

### 3. Construir Dataset de Pares
Para preparar nuevas muestras de entrenamiento de manera asistida:

```bash
python src/dataset_builder.py
```
- Permite seleccionar la imagen de partida, aplicar SAM iterativamente y almacenar de manera sincronizada la imagen de entrada y su máscara correspondiente en `data/input_train/` y `data/output_train/`.

### 4. Entrenar la Red U-Net
Para entrenar o ajustar la U-Net sobre el conjunto de datos preparado:

```bash
python src/training.py
```
- Lee las imágenes desde `data/input_train/` y `data/output_train/`.
- Aplica las técnicas de Data Augmentation configuradas.
- Valida con `data/input_val/` y `data/output_val/`.
- Guarda checkpoints automáticos y almacena el mejor modelo en `checkpoints/best_model.pth`.

---

## Proyección de Despliegue en la Nube (AWS)

El diseño desacoplado del proyecto permite una migración y escalabilidad directa sobre la infraestructura de Amazon Web Services:

1. **Amazon S3 (`s3://silhouette-project/`)**:
   - `dataset/`: Almacenamiento centralizado y versionado de pares de entrenamiento (`input_train/`, `output_train/`).
   - `models/`: Repositorio de artefactos (`sam_vit_h_4b8939.pth`, `best_model.pth`).
   - `results/`: Archivo histórico de resultados vectoriales y analíticos.

2. **AWS SageMaker**:
   - Ejecución de trabajos de entrenamiento distribuidos (*SageMaker Training Jobs*) aprovechando instancias GPU dedicadas (`ml.g4dn` / `ml.g5`).
   - Registro y linaje de modelos mediante *Model Registry*.

3. **Amazon EC2 / ECS**:
   - Despliegue del contenedor con el servidor **FastAPI** (`src/api.py`), modelos pre-cargados en GPU y Potrace para atención de solicitudes web de baja latencia.
