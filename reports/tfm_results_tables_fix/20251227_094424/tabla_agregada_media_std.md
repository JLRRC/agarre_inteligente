# Tabla agregada por experimento (media±desv muestral)

| Experimento | Modelo | Modalidad | Augment | n | val_success (mean±std) | val_iou (mean±std) | val_angle (mean±std) | val_loss (mean±std) |
|---|---|---|---|---:|---:|---:|---:|---:|
| EXP1_SIMPLE_RGB | simple_cnn | RGB | - | 3 | 0.181 ± 0.020 | 0.290 ± 0.008 | 73.37 ± 2.12 | 19.300 ± 0.481 |
| EXP2_SIMPLE_RGBD | simple_cnn | RGBD | - | 3 | 0.190 ± 0.023 | 0.287 ± 0.018 | 72.98 ± 2.01 | 19.527 ± 0.727 |
| EXP3_RESNET18_RGBD | resnet18 | RGBD | - | 3 | 0.254 ± 0.009 | 0.282 ± 0.012 | 65.38 ± 3.26 | 17.646 ± 0.555 |
| EXP3_RESNET18_RGB_AUGMENT | resnet18 | RGB | geo=1\|photo=1 | 3 | 0.266 ± 0.033 | 0.294 ± 0.024 | 62.64 ± 5.74 | 16.913 ± 0.983 |
