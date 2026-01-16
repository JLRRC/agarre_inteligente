# Tabla 4-6

WARNING: IoU calculado sobre bbox axis-aligned derivada del rectangulo orientado.

| metric | definition | threshold |
|---|---|---|
| IoU | IoU(bbox_pred, bbox_gt) >= iou_thresh | 0.25 |
| DeltaTheta | angle_diff_deg(pred, gt) <= angle_thresh | 30.0 |
| grasp_success | IoU and DeltaTheta criteria | IoU>=0.25 & dTheta<=30.0 |
