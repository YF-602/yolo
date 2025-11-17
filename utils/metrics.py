import numpy as np
from .core_computation import *


def calculate_class_tp_fp(gt_labels, pred_labels, threshold):
    # 类别名称
    class_names = ['car', 'person', 'bicycle']
    num_classes = len(class_names)
    
    # 为每个类别存储TP/FP信息
    class_tp_fp = {i: [] for i in range(num_classes)}
    # e.g.
    # {
    #     0: [(1, 0.80), (0, 0.31)],
    #     1: [(1, 0.80), (0, 0.31)],
    #     2: [(1, 0.80), (0, 0.31)],
    #     3: [(1, 0.80), (0, 0.31)],
    # }
    
    # 处理每个图像
    for img_name in gt_labels:
        gt_boxes = gt_labels[img_name]
        pred_boxes = pred_labels.get(img_name, [])
        
        # 为每个预测框找到最佳匹配的真实框
        used_gt = set()
        
        for pred_cls, pred_coords in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (_, gt_coords) in enumerate(gt_boxes):
                if gt_idx in used_gt:
                    continue
                
                iou = calculate_iou(pred_coords, gt_coords)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # 记录TP/FP
            if best_gt_idx != -1 and best_iou >= threshold and pred_cls == gt_boxes[best_gt_idx][0]:
                class_tp_fp[pred_cls].append((1, best_iou))  # TP
                used_gt.add(best_gt_idx)
            else:
                class_tp_fp[pred_cls].append((0, best_iou))  # FP

    return class_tp_fp

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

def calculate_map50_map5095(class_tp_fp, class_gt_count, class_names):
    """
    class_tp_fp: from calculate_class_tp_fp(gt_labels, pred_labels, threshold)
    class_gt_count: from calculate_class_gt_count(gt_labels)
    """
    # 类别名称
    num_classes = len(class_names)

    # 计算mAP50和mAP50-95
    aps_50 = []
    aps_50_95 = []

    for cls_id in range(num_classes):
        # 按置信度排序 (这里使用IoU作为置信度代理)
        tp_fp_list = class_tp_fp[cls_id]
        if not tp_fp_list:
            aps_50.append(0)
            aps_50_95.append(0)
            continue
        
        # 按IoU降序排序
        tp_fp_list.sort(key=lambda x: x[1], reverse=True)
        
        # 计算不同IoU阈值下的AP
        thresholds_50 = [0.5]
        thresholds_50_95 = np.arange(0.5, 1.0, 0.05)
        
        ap_50 = calculate_ap_for_threshold(tp_fp_list, class_gt_count[cls_id], 0.5)
        ap_50_95 = np.mean([calculate_ap_for_threshold(tp_fp_list, class_gt_count[cls_id], t) 
                           for t in thresholds_50_95])
        
        aps_50.append(ap_50)
        aps_50_95.append(ap_50_95)
    
    mAP50 = np.mean(aps_50)
    mAP50_95 = np.mean(aps_50_95)

    return mAP50, mAP50_95