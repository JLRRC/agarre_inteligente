# Agarre inteligente por brazos robóticos en entornos no estructurados

Repositorio del Trabajo Fin de Máster (TFM) **“Agarre inteligente por brazos robóticos en entornos no estructurados”**.  
El objetivo es construir un **pipeline reproducible** de detección de poses de agarre en imágenes **RGB-D** utilizando **deep learning** y el **Cornell Grasping Dataset**.

El repositorio contiene:

- Código en **PyTorch** para:
  - Carga y preprocesado del dataset Cornell.
  - Modelos de detección de agarres basados en **CNN simple** y **ResNet-18**, tanto en **RGB** como en **RGB-D** (fusionando profundidad).
  - Entrenamiento, validación y cálculo de métricas tipo Cornell (IoU, error angular, *grasp success*).
- Ficheros de configuración **YAML** para lanzar experimentos controlados.
- Resultados experimentales (métricas en CSV) y comparativas **A/B** entre modelos.

> **Importante:**  
> El dataset Cornell **no se incluye** en el repositorio.  
> Cada usuario debe descargarlo y colocarlo localmente en la carpeta `data/` (ver sección correspondiente).

---

## 1. Estructura del repositorio

```text
agarre_inteligente/
├── .venv/                          # Entorno virtual local (NO se versiona)
├── config/                         # Configuraciones YAML de modelos y experimentos
│   ├── cornell_simple.yaml                 # Config base SimpleGraspCNN (RGB)
│   ├── cornell_resnet18.yaml               # Config base ResNet18Grasp (RGB)
│   ├── exp1_simple_rgb.yaml                # EXP1: SimpleGraspCNN RGB sin augment
│   ├── exp2_simple_rgb_augment.yaml        # EXP2: SimpleGraspCNN RGB + augment
│   └── exp3_resnet18_rgbd.yaml             # EXP3: ResNet18Grasp RGB-D (RGB + Depth)
├── data/                           # Datasets locales (IGNORADOS por git)
│   ├── cornell_raw/                # Cornell original (ficheros .png/.tiff/.txt)
│   └── cornell_processed/          # Versión re-escalada/preprocesada
├── experiments/                    # Resultados experimentales
│   ├── cornell_simple_baseline/            # Métricas baseline SimpleGraspCNN
│   ├── cornell_resnet18_baseline/          # Métricas baseline ResNet18Grasp
│   ├── exp1_simple_rgb/                    # EXP1: metrics.csv + config_used.yaml
│   ├── exp2_simple_rgb_augment/            # EXP2
│   ├── EXP3_RESNET18_RGBD_seed0/           # EXP3: ResNet18Grasp RGB-D (RGB + Depth)
│   ├── ab_exp1_vs_exp2_val_success.csv     # Comparativa A/B (EXP1 vs EXP2)
│   ├── ab_exp2_vs_exp3_val_success.csv     # Comparativa A/B (EXP2 vs EXP3)
│   ├── plan_experimentos_base.md           # Plan de experimentos base
│   └── summary_base.csv                    # Resumen tabular de resultados
├── src/
│   ├── graspnet/
│   │   ├── datasets/
│   │   │   └── cornell_dataset.py          # Dataloader y preprocesado Cornell (RGB / RGB-D)
│   │   ├── models/
│   │   │   ├── simple_cnn.py               # Modelo SimpleGraspCNN
│   │   │   └── resnet18_grasp.py           # Modelo ResNet18Grasp (RGB / RGB-D)
│   │   ├── train/
│   │   │   ├── train_cornell.py            # Script principal de entrenamiento
│   │   │   ├── debug_dataloader.py         # Script de depuración del dataloader
│   │   │   ├── build_summary_base.py       # Construcción de summary_base.csv
│   │   │   └── compare_ab.py               # Comparativas A/B entre experimentos
│   │   └── utils/
│   │       ├── metrics.py                  # IoU, ángulo, grasp success, etc.
│   │       └── debug_metrics.py            # Utilidades de depuración de métricas
│   └── __init__.py
├── requirements.txt                # Dependencias Python del proyecto
└── README.md                       # Este documento

⸻

## 2. Requisitos
•	Sistema operativo: Ubuntu 22.04 / 24.04 (probado en MeLE Overclock 4C N300).
•	Python: 3.10+ (en el TFM se ha usado Python 3.12).
•	Git para la gestión del repositorio.
•	GPU opcional (CUDA) para acelerar el entrenamiento, aunque los experimentos base
se han planteado para poder ejecutarse también en CPU.

⸻

## 3. Preparación del entorno
Desde la carpeta raíz del proyecto (agarre_inteligente/):

1) Crear y activar entorno virtual
    python -m venv .venv
    source .venv/bin/activate

2) Instalar dependencias
    pip install --upgrade pip
    pip install -r requirements.txt

3) Añadir src/ al PYTHONPATH
    export PYTHONPATH=$(pwd)/src:$PYTHONPATH

Cada nueva sesión de terminal requiere:
    cd ~/TFM/agarre_inteligente
    source .venv/bin/activate
    export PYTHONPATH=$(pwd)/src:$PYTHONPATH

⸻

## 4. Preparación del dataset Cornell
El repositorio no incluye el Cornell Grasping Dataset por cuestiones de tamaño
y licencia. Para reproducir los experimentos:
1.	Descargar el Cornell Grasping Dataset desde su fuente oficial.
2.	Descomprimirlo en:
        agarre_inteligente/data/cornell_raw/
quedando una estructura similar a:
        data/cornell_raw/
        ├── 01/
        ├── 02/
        ├── ...
        ├── 10/
        ├── backgrounds/
        └── cornell_grasp/    # según organización original
3.	Ejecutar, si es necesario, el script de preprocesado definido en
src/graspnet/datasets/cornell_dataset.py (la lógica interna del dataloader
se encarga de generar/gestionar cornell_processed/ según los parámetros
de configuración).

En este trabajo se trabaja con imágenes re-escaladas y normalizadas, adaptadas al
tamaño de entrada de los modelos (por ejemplo 224×224).

⸻

## 5. Modelos implementados

5.1. SimpleGraspCNN
•	Arquitectura CNN ligera, diseñada específicamente para el TFM.
•	Entrada: imagen RGB (o RGB-D fusionado) preprocesada.
•	Salida: vector de 5 parámetros por agarre:
•	(cx, cy): centro del rectángulo de agarre.
•	(w, h): ancho y alto.
•	angle: ángulo de rotación del agarre (en radianes o grados, según config).

Configuración base: config/cornell_simple.yaml.

5.2. ResNet18Grasp
	•	Basado en torchvision.models.resnet18 con pesos preentrenados.
	•	Se adapta la primera capa convolucional para admitir tanto entradas:
	•	RGB (3 canales), como
	•	RGB-D (4 canales, RGB + mapa de profundidad), según la configuración del experimento (in_channels).
	•	Se sustituye la capa fc final para producir 5 parámetros (cx, cy, w, h, angle).
	•	Pensado como modelo de mayor capacidad para comparar con la CNN ligera.

Configuración base (RGB): config/cornell_resnet18.yaml.
Configuración experimental RGB-D: config/exp3_resnet18_rgbd.yaml.

⸻

## 6. Métricas tipo Cornell

En src/graspnet/utils/metrics.py se implementan las métricas utilizadas en el
Cornell Grasping Dataset:
•	IoU entre el rectángulo de agarre predicho y el de la anotación.
•	Error angular entre el ángulo predicho y el de la anotación (en grados).
•	Grasp success binaria:
•	Éxito si IoU ≥ 0.25 y Δθ ≤ 30°.
•	Fracaso en caso contrario.

Durante la validación, el script de entrenamiento registra:
•	val_loss
•	val_iou
•	val_angle (error angular medio en grados)
•	val_success (porcentaje de agarres correctos)

Estas métricas se guardan en los distintos metrics.csv dentro de experiments/.

⸻

## 7. Ejecución básica

7.1. Comprobar el dataloader

Tras preparar el entorno y el dataset:
cd ~/TFM/agarre_inteligente
source .venv/bin/activate
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

python src/graspnet/train/debug_dataloader.py

Este script carga varios batches del dataset y muestra por pantalla las formas
de los tensores (RGB, Depth, parámetros de agarre) para verificar que todo
está correctamente configurado.

⸻

## 8. Experimentos base del TFM

Los experimentos se lanzan mediante el script genérico
src/graspnet/train/train_cornell.py, indicando un fichero de configuración
YAML y una semilla aleatoria.

8.1. EXP1 – SimpleGraspCNN, RGB sin data augmentation
python src/graspnet/train/train_cornell.py \
    --config config/exp1_simple_rgb.yaml \
    --seed 0
8.2. EXP2 – SimpleGraspCNN, RGB con data augmentation
python src/graspnet/train/train_cornell.py \
    --config config/exp2_simple_rgb_augment.yaml \
    --seed 0
8.3. EXP3 – ResNet18Grasp, RGB con data augmentation
python src/graspnet/train/train_cornell.py \
    --config config/exp3_resnet18_rgb_augment.yaml \
    --seed 0
En este experimento se activa el uso de profundidad (use_depth: true en el YAML),
fusionando el mapa de profundidad como un cuarto canal de entrada (RGB-D) y
adaptando ResNet-18 para trabajar con 4 canales.

Cada ejecución genera:
	•	Una carpeta en experiments/<nombre_experimento>/ con:
	•	config_used.yaml: configuración efectivamente utilizada.
	•	metrics.csv: tabla con métricas por época.
	•	best.pth o similar (si se incluye guardado de checkpoints, opcional en el TFM).

⸻

## 9. Scripts de análisis y comparativa A/B

En la carpeta src/graspnet/train/ se incluyen scripts auxiliares para el
análisis de resultados:
	•	build_summary_base.py
Recorre las carpetas de experiments/ y construye summary_base.csv con
un resumen de las mejores métricas por experimento.
	•	compare_ab.py
Genera comparativas A/B entre dos experimentos, produciendo ficheros
como:
	•	experiments/ab_exp1_vs_exp2_val_success.csv
	•	experiments/ab_exp2_vs_exp3_val_success.csv

Estos ficheros se utilizan posteriormente en la memoria del TFM para elaborar
tablas y gráficos de comparación entre modelos.

⸻

## 10. Reproducibilidad

Para reproducir los resultados de este trabajo:
	1.	Clonar el repositorio.
	2.	Preparar el entorno virtual e instalar dependencias.
	3.	Descargar Cornell y colocarlo en data/cornell_raw/.
	4.	Ejecutar los tres experimentos base (EXP1 y EXP2 en RGB, EXP3 en RGB-D).
	5.	Ejecutar build_summary_base.py y compare_ab.py para generar los CSV
de resumen y comparativas.

Dado que el código, las configuraciones YAML y las métricas generadas se
versionan en este repositorio, un tercero puede replicar el proceso siguiendo
las instrucciones anteriores.

⸻

## 11. Trabajo futuro

Para reproducir los resultados de este trabajo:
	1.	Clonar el repositorio.
	2.	Preparar el entorno virtual e instalar dependencias.
	3.	Descargar Cornell y colocarlo en data/cornell_raw/.
	4.	Ejecutar los tres experimentos base (EXP1 y EXP2 en RGB, EXP3 en RGB-D).
	5.	Ejecutar build_summary_base.py y compare_ab.py para generar los CSV
de resumen y comparativas.

Dado que el código, las configuraciones YAML y las métricas generadas se
versionan en este repositorio, un tercero puede replicar el proceso siguiendo
las instrucciones anteriores.

⸻

## 12. Autoría
	•	Autor: Jesús Lozano Rodríguez
	•	Usuario GitHub: JLRRC￼

Este repositorio forma parte del Trabajo Fin de Máster del
Máster Universitario en Inteligencia Artificial (MIAR).
