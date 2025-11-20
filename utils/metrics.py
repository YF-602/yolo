import numpy as np
from .core_computation import *


def calculate_class_tp_conf(gt_labels, pred_labels, iou_threshold):
    # 类别名称
    class_names = ['car', 'person', 'bicycle']
    num_classes = len(class_names)
    
    # 为每个类别存储(gt, confidence)信息
    class_tp_conf = {i: [] for i in range(num_classes)}
    # e.g.
    # {
    #     0: [(1, 0.80), (0, 0.31), (2, 0.87)...],
    #     1: [(1, 0.80), (0, 0.31), (1, 0.57)...],
    #     2: [(1, 0.80), (0, 0.31), (3, 0.85)...],
    #     3: [(1, 0.80), (0, 0.31), (1, 0.32)...],
    # }
    
    # 处理每个图像
    for img_name in gt_labels:
        gt_boxes = gt_labels[img_name]
        pred_boxes = pred_labels.get(img_name, [])
        
        # 为每个预测框找到最佳匹配的真实框
        used_gt = set()
        
        for pred_cls, pred_coords, conf in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_cls, gt_coords) in enumerate(gt_boxes):
                if gt_idx in used_gt:
                    continue
                
                iou = calculate_iou(pred_coords, gt_coords)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # 记录
            if best_iou < iou_threshold: # 简化了or best_gt_idx == -1
                class_tp_conf[pred_cls].append((0, conf))
            elif pred_cls != gt_boxes[best_gt_idx][0]:
                class_tp_conf[pred_cls].append((0, conf))
            else: 
                class_tp_conf[pred_cls].append((1, conf))
                used_gt.add(best_gt_idx)

    return class_tp_conf

def calculate_class_gt_count(gt_labels):
    # 类别名称
    class_names = ['car', 'person', 'bicycle']
    num_classes = len(class_names)
    # 存储
    class_gt_count = {i: 0 for i in range(num_classes)}
    # e.g.
    # {
    #     0: 34,
    #     1: 88,
    #     2: 43,
    #     3: 56
    # }

    # 处理每个图像
    for img_name in gt_labels:
        gt_boxes = gt_labels[img_name]
        
        # 统计真实框数量
        for cls_id, _ in gt_boxes:
            class_gt_count[cls_id] += 1
    
    return class_gt_count

def calculate_map50_map5095(class_gt_count, class_names, gt_labels, pred_labels):
    """
    class_gt_count: from calculate_class_gt_count(gt_labels)
    """
    # 类别名称
    num_classes = len(class_names)

    # 计算mAP50和mAP50-95
    aps_50 = []
    aps_50_95 = []

    tp_conf_dict_50 = calculate_class_tp_conf(gt_labels, pred_labels, 0.5)
    thresholds_50_95 = np.arange(0.5, 1.0, 0.05)
    tp_conf_dict_50_95_list = []
    for t in thresholds_50_95:
        tp_conf_dict_50_95_list.append(calculate_class_tp_conf(gt_labels, pred_labels, t))

    for cls_id in range(num_classes):       
        # 计算不同IoU阈值下的AP
        tp_conf_list_50 = tp_conf_dict_50[cls_id]
        ap_50 = calculate_ap(tp_conf_list_50, class_gt_count[cls_id])

        ap_50_95 = []
        for i in range(len(tp_conf_dict_50_95_list)):
            tp_conf_list_t = tp_conf_dict_50_95_list[i][cls_id]
            ap_50_95.append(calculate_ap(tp_conf_list_t, class_gt_count[cls_id]))
        ap_50_95 = np.mean(ap_50_95)
        
        aps_50.append(ap_50)
        aps_50_95.append(ap_50_95)
    
    mAP50 = np.mean(aps_50)
    mAP50_95 = np.mean(aps_50_95)

    return mAP50, mAP50_95