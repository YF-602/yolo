import os
from .data import *
from .metrics import *
from .visualization import *


def evaluate_detection(
        gt_dir, 
        pred_dir, 
        class_names, 
        iou_threshold,
        output_dir='./evaluation_results'):
    """主评估函数"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    gt_labels = read_label_files(gt_dir)
    pred_labels = read_label_files(pred_dir)

    class_tp_conf = calculate_class_tp_conf(gt_labels, pred_labels, iou_threshold)
    class_gt_count = calculate_class_gt_count(gt_labels)

    # 生成图表
    generate_complete_confusion_matrix(
        gt_labels, pred_labels, iou_threshold, class_names, output_dir)
    generate_f1_curve(class_tp_conf, class_gt_count, class_names, output_dir)
    generate_pr_curves(class_tp_conf, class_gt_count, class_names, output_dir)
    
    return calculate_map50_map5095(class_gt_count, class_names, gt_labels, pred_labels)