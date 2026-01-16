# Tabla resultados por seed (desde summary_base.csv)

| exp_base | seed | val_success | val_iou | val_angle | val_loss | model | modality | augment | best_epoch |
|---|---:|---:|---:|---:|---:|---|---|---|---:|
| EXP1_SIMPLE_RGB | 0 | 0.199623 | 0.298672 | 71.72 | 18.777305 | simple_cnn | RGB | - | 8 |
| EXP1_SIMPLE_RGB | 1 | 0.160075 | 0.282304 | 75.76 | 19.722280 | simple_cnn | RGB | - | 10 |
| EXP1_SIMPLE_RGB | 2 | 0.182674 | 0.290347 | 72.63 | 19.401650 | simple_cnn | RGB | - | 9 |
| EXP2_SIMPLE_RGBD | 0 | 0.216573 | 0.307491 | 70.73 | 18.689281 | simple_cnn | RGBD | - | 9 |
| EXP2_SIMPLE_RGBD | 1 | 0.173258 | 0.279090 | 74.59 | 19.899670 | simple_cnn | RGBD | - | 9 |
| EXP2_SIMPLE_RGBD | 2 | 0.180791 | 0.274909 | 73.61 | 19.993421 | simple_cnn | RGBD | - | 8 |
| EXP4_RESNET18_RGBD | 0 | 0.258004 | 0.278729 | 66.84 | 17.983090 | resnet18 | RGBD | - | 10 |
| EXP4_RESNET18_RGBD | 1 | 0.242938 | 0.295328 | 67.66 | 17.950211 | resnet18 | RGBD | - | 9 |
| EXP4_RESNET18_RGBD | 2 | 0.259887 | 0.272065 | 61.64 | 17.005788 | resnet18 | RGBD | - | 8 |
| EXP3_RESNET18_RGB_AUGMENT | 0 | 0.303202 | 0.309791 | 56.96 | 15.874630 | resnet18 | RGB | geo=1\|photo=1 | 9 |
| EXP3_RESNET18_RGB_AUGMENT | 1 | 0.239171 | 0.305115 | 68.43 | 17.828861 | resnet18 | RGB | geo=1\|photo=1 | 8 |
| EXP3_RESNET18_RGB_AUGMENT | 2 | 0.254237 | 0.266338 | 62.52 | 17.034867 | resnet18 | RGB | geo=1\|photo=1 | 8 |
