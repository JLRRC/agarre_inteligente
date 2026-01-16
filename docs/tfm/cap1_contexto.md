# Ruta/URL: file:///home/laboratorio/TFM/agarre_inteligente/docs/tfm/cap1_contexto.md
# Nombre: cap1_contexto.md
# Qué hace: Resume el contexto académico y los dos workspaces usados en el TFM.

# Capítulo 1 — Contexto

Este trabajo aborda el **agarre inteligente en entornos no estructurados** combinando:

- Dataset sintético/real del Cornell Grasping Dataset (`data/cornell_raw`).
- Entrenamiento/probado de redes ligeras (`src/graspnet/models/`, `config/*.yaml`).
- Simulación y control de un UR5e con gripper RG2 dentro del workspace `agarre_ros2_ws`.

Los dos workspaces que componen el proyecto son:

- `agarre_inteligente/`: pipeline de percepción, dataset, modelos y métricas. Aquí se ejecutan los modelos, se almacenan los experimentos (`experiments/EXP*`) y se generan las métricas que alimentan los capítulos teóricos.
- `agarre_ros2_ws/`: ROS 2 Jazzy + Gazebo + panel Qt (en `src/ur5_qt_panel/…`) encargados de la demo. Las escenas (`worlds/`), modelos (`models/ur5_rg2/`), scripts (`scripts/`), bridges (`ros_gz_bridge`) y controladores mock (`src/ur5_bringup/config/ur5_mock_controllers.yaml`) se coordinan desde este workspace.

El TFM conecta ambos mundos mediante scripts (`scripts/run_one.sh` y `scripts/run_seeds.sh` en ML) para que el panel pueda disparar entrenamientos y recoger evidencias reproducibles.
