import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .core_computation import *


def generate_complete_confusion_matrix(gt_labels_dict, pred_labels_dict, threshold, class_names, output_dir):
    """生成完整的目标检测混淆矩阵（兼容字典格式）"""
    
    # 扩展类别名称，加入背景
    extended_class_names = ['background'] + class_names
    num_classes = len(extended_class_names)
    
    # 初始化混淆矩阵 (包含背景)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # 确保处理相同的图像
    common_image_names = sorted(set(gt_labels_dict.keys()) | set(pred_labels_dict.keys()))
    
    # 处理每个图像
    for img_name in common_image_names:
        gt_boxes = gt_labels_dict.get(img_name, [])
        pred_boxes = pred_labels_dict.get(img_name, [])
        
        used_gt = set()
        used_pred = set()
        
        # 第一步：处理匹配的检测 (TP和分类错误)
        for i, pred_box in enumerate(pred_boxes):
            pred_cls, pred_coords = pred_box  # 解包 (class_id, [cx, cy, w, h])
            pred_class = pred_cls + 1  # +1因为0是背景
            
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if j in used_gt:
                    continue
                gt_cls, gt_coords = gt_box  # 解包 (class_id, [cx, cy, w, h])
                iou = calculate_iou(pred_coords, gt_coords)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_gt_idx != -1 and best_iou >= threshold:
                # 有匹配的真实框
                gt_cls, _ = gt_boxes[best_gt_idx]
                gt_class = gt_cls + 1
                cm[pred_class, gt_class] += 1  # [预测, 真实]
                used_gt.add(best_gt_idx)
                used_pred.add(i)
            else:
                # 没有匹配的真实框 → 把背景预测为物体
                cm[pred_class, 0] += 1  # [预测类别, 背景]
                used_pred.add(i)
        
        # 第二步：处理漏检 (真实框没有被匹配)
        for j, gt_box in enumerate(gt_boxes):
            if j not in used_gt:
                gt_cls, _ = gt_box
                gt_class = gt_cls + 1
                cm[0, gt_class] += 1  # [背景, 真实类别]
    
    # 生成混淆矩阵图表
    generate_confusion_matrix_plots(cm, extended_class_names, output_dir)
    
    return cm

def generate_confusion_matrix_plots(cm, class_names, output_dir):
    """生成混淆矩阵的可视化"""
    
    # 非归一化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Complete Confusion Matrix (Including Background)')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_complete.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 归一化混淆矩阵（按行归一化）
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    plt.title('Normalized Confusion Matrix (Row-wise)')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成简化的混淆矩阵（不含背景，便于查看类别间混淆）
    if len(cm) > 1:
        cm_simple = cm[1:, 1:]  # 去掉背景行和列
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_simple, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names[1:], yticklabels=class_names[1:])
        plt.title('Simplified Confusion Matrix (Objects Only)')
        plt.xlabel('True Label')
        plt.ylabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_simple.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

# TODO 生成的曲线形态貌似有点问题
def generate_pr_curves(class_tp_fp, class_gt_count, class_names, output_dir):
    """生成PR曲线、P曲线、R曲线"""
    
    # PR曲线
    plt.figure(figsize=(10, 8))
    
    for cls_id in range(len(class_names)):
        tp_fp_list = class_tp_fp[cls_id]
        if not tp_fp_list:
            continue
            
        # 按置信度排序
        tp_fp_list.sort(key=lambda x: x[1], reverse=True)
        
        tp_cumsum = np.cumsum([tp for tp, _ in tp_fp_list])
        fp_cumsum = np.cumsum([1 - tp for tp, _ in tp_fp_list])
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / (class_gt_count[cls_id] + 1e-6)
        
        # 确保曲线从(0,0)开始
        recalls = np.concatenate(([0], recalls, [1]))
        precisions = np.concatenate(([1], precisions, [0]))
        
        plt.plot(recalls, precisions, label=f'{class_names[cls_id]}', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'PR_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # P曲线和R曲线
    thresholds = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(12, 5))
    
    # P曲线
    plt.subplot(1, 2, 1)
    for cls_id in range(len(class_names)):
        tp_fp_list = class_tp_fp[cls_id]
        if not tp_fp_list:
            continue
            
        precisions = []
        for threshold in thresholds:
            valid_detections = [tp for tp, iou in tp_fp_list if iou >= threshold]
            if not valid_detections:
                precisions.append(0)
                continue
            precision = sum(valid_detections) / len(valid_detections)
            precisions.append(precision)
        
        plt.plot(thresholds, precisions, label=f'{class_names[cls_id]}', linewidth=2)
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Precision')
    plt.title('Precision Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # R曲线
    plt.subplot(1, 2, 2)
    for cls_id in range(len(class_names)):
        tp_fp_list = class_tp_fp[cls_id]
        if not tp_fp_list:
            continue
            
        recalls = []
        for threshold in thresholds:
            valid_tp = sum(tp for tp, iou in tp_fp_list if iou >= threshold)
            recall = valid_tp / (class_gt_count[cls_id] + 1e-6)
            recalls.append(recall)
        
        plt.plot(thresholds, recalls, label=f'{class_names[cls_id]}', linewidth=2)
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Recall')
    plt.title('Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'P_R_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_f1_curve(class_tp_fp, class_gt_count, class_names, output_dir):
    """生成F1曲线"""
    
    thresholds = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(10, 6))
    
    for cls_id in range(len(class_names)):
        tp_fp_list = class_tp_fp[cls_id]
        if not tp_fp_list:
            continue
            
        f1_scores = []
        for threshold in thresholds:
            valid_detections = [(tp, iou) for tp, iou in tp_fp_list if iou >= threshold]
            if not valid_detections:
                f1_scores.append(0)
                continue
                
            tp_count = sum(tp for tp, _ in valid_detections)
            fp_count = len(valid_detections) - tp_count
            fn_count = class_gt_count[cls_id] - tp_count
            
            precision = tp_count / (tp_count + fp_count + 1e-6)
            recall = tp_count / (tp_count + fn_count + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            f1_scores.append(f1)
        
        plt.plot(thresholds, f1_scores, label=f'{class_names[cls_id]}', linewidth=2)
    
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'F1_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_val_batch_plots(gt_dir, pred_dir, output_dir):
    """生成val_batch0_labels和val_batch0_pred图表"""
    # 这里可以添加生成验证批次对比图的代码
    # 由于需要图像数据，这里暂时跳过
    pass