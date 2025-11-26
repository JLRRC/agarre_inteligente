from graspnet.utils.metrics import (
    params_to_rect,
    rect_to_bbox,
    bbox_iou,
    angle_diff_deg,
    compute_grasp_success,
)


def main():
    # Dos agarres muy parecidos
    pred = [112.0, 112.0, 60.0, 30.0, 10.0]
    gt   = [110.0, 110.0, 62.0, 28.0, 15.0]

    rect_p = params_to_rect(*pred)
    rect_g = params_to_rect(*gt)

    bbox_p = rect_to_bbox(rect_p)
    bbox_g = rect_to_bbox(rect_g)

    iou = bbox_iou(bbox_p, bbox_g)
    ang = angle_diff_deg(pred[-1], gt[-1])
    success = compute_grasp_success(pred, gt)

    print("bbox_pred:", bbox_p)
    print("bbox_gt  :", bbox_g)
    print("IoU      :", iou)
    print("Δángulo  :", ang)
    print("Success  :", success)


if __name__ == "__main__":
    main()
