import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from pathlib import Path

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    # box: [x_center, y_center, width, height]
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 转换为 [x1, y1, x2, y2] 格式
    box1_x1 = x1 - w1/2
    box1_y1 = y1 - h1/2
    box1_x2 = x1 + w1/2
    box1_y2 = y1 + h1/2
    
    box2_x1 = x2 - w2/2
    box2_y1 = y2 - h2/2
    box2_x2 = x2 + w2/2
    box2_y2 = y2 + h2/2
    
    # 计算交集
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # 计算并集
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def read_label_files(labels_dir):
    """读取标签文件"""
    labels = {}
    for label_file in Path(labels_dir).glob('*.txt'):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                boxes.append((cls_id, coords))
        labels[label_file.stem] = boxes
    return labels
# e.g.
# {
#     '000016': [(2, [0.598802, 0.539, 0.592814, 0.798])], 
#     '000021': [(1, [0.346726, 0.259, 0.33631, 0.222]), 
#                (1, [0.266369, 0.601, 0.425595, 0.454]), 
#                (1, [0.81994, 0.513, 0.342262, 0.882])], 
#     '000023': [(1, [0.471557, 0.455, 0.583832, 0.906]), 
#                (2, [0.387725, 0.737, 0.757485, 0.526]), 
#                (1, [0.892216, 0.222, 0.215569, 0.396]), 
#                (1, [0.080838, 0.105, 0.155689, 0.206]), 
#                (2, [0.892216, 0.71, 0.215569, 0.532])], 
# }

def calculate_ap(recalls, precisions):
    """
    计算AP (Average Precision)
    PR曲线下的面积, 越接近1越好
    """
    # 将recall从0到1进行插值
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    
    # 确保precision是单调递减的
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # 计算AP
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    
    return ap

def calculate_ap_for_threshold(tp_fp_list, gt_count, iou_threshold):
    """计算特定IoU阈值下的AP"""
    if gt_count == 0:
        return 0
    
    # 过滤出满足IoU阈值的检测
    valid_detections = [(tp, iou) for tp, iou in tp_fp_list if iou >= iou_threshold]
    
    if not valid_detections:
        return 0
    
    # 计算precision-recall曲线
    tp_cumsum = np.cumsum([tp for tp, _ in valid_detections])
    fp_cumsum = np.cumsum([1 - tp for tp, _ in valid_detections])
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (gt_count + 1e-6)
    
    return calculate_ap(recalls, precisions)

def generate_complete_confusion_matrix(gt_labels_dict, pred_labels_dict, threshold, class_names, output_dir):
    """生成完整的目标检测混淆矩阵（兼容字典格式）"""
    
    # e.g.(gt_labels_dict and pred_labels_dict)
    # {
    #     '000016': [(2, [0.598802, 0.539, 0.592814, 0.798])], 
    #     '000021': [(1, [0.346726, 0.259, 0.33631, 0.222]), 
    #                (1, [0.266369, 0.601, 0.425595, 0.454]), 
    #                (1, [0.81994, 0.513, 0.342262, 0.882])], 
    #     '000023': [(1, [0.471557, 0.455, 0.583832, 0.906]), 
    #                (2, [0.387725, 0.737, 0.757485, 0.526]), 
    #                (1, [0.892216, 0.222, 0.215569, 0.396]), 
    #                (1, [0.080838, 0.105, 0.155689, 0.206]), 
    #                (2, [0.892216, 0.71, 0.215569, 0.532])], 
    # }
    
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

def evaluate_detection(gt_dir, pred_dir, output_dir='./evaluation_results'):
    """主评估函数"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取标签文件
    # e.g.(gt_labels and pred_labels)
    # {
    #     '000016': [(2, [0.598802, 0.539, 0.592814, 0.798])], 
    #     '000021': [(1, [0.346726, 0.259, 0.33631, 0.222]), 
    #                (1, [0.266369, 0.601, 0.425595, 0.454]), 
    #                (1, [0.81994, 0.513, 0.342262, 0.882])], 
    #     '000023': [(1, [0.471557, 0.455, 0.583832, 0.906]), 
    #                (2, [0.387725, 0.737, 0.757485, 0.526]), 
    #                (1, [0.892216, 0.222, 0.215569, 0.396]), 
    #                (1, [0.080838, 0.105, 0.155689, 0.206]), 
    #                (2, [0.892216, 0.71, 0.215569, 0.532])], 
    # }

    gt_labels = read_label_files(gt_dir)
    pred_labels = read_label_files(pred_dir)
    
    # 类别名称
    class_names = ['car', 'person', 'bicycle']
    num_classes = len(class_names)
    
    # 初始化统计变量
    confusion_data = []
    
    # 为每个类别存储TP/FP信息
    class_tp_fp = {i: [] for i in range(num_classes)}
    class_gt_count = {i: 0 for i in range(num_classes)}
    
    # 处理每个图像
    for img_name in gt_labels:
        gt_boxes = gt_labels[img_name]
        pred_boxes = pred_labels.get(img_name, [])
        
        # 统计真实框数量
        for cls_id, _ in gt_boxes:
            class_gt_count[cls_id] += 1
        
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
            if best_gt_idx != -1 and best_iou >= 0.5 and pred_cls == gt_boxes[best_gt_idx][0]:
                class_tp_fp[pred_cls].append((1, best_iou))  # TP
                used_gt.add(best_gt_idx)
                confusion_data.append((gt_boxes[best_gt_idx][0], pred_cls))
            else:
                class_tp_fp[pred_cls].append((0, best_iou))  # FP
                if best_gt_idx != -1:
                    confusion_data.append((gt_boxes[best_gt_idx][0], pred_cls))
    
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
    
    # 生成图表
    generate_complete_confusion_matrix(
        gt_labels_dict=gt_labels,           # 真实标签字典
        pred_labels_dict=pred_labels,       # 预测标签字典  
        threshold=0.5,                      # IoU阈值
        class_names=class_names,            # 类别名称
        output_dir=output_dir               # 输出目录
    )
    generate_pr_curves(class_tp_fp, class_gt_count, class_names, output_dir)
    generate_f1_curve(class_tp_fp, class_gt_count, class_names, output_dir)
    
    return mAP50, mAP50_95

# 使用示例
if __name__ == "__main__":
    gt_dir = './datasets/labels/test'      # 真实标签目录
    pred_dir = './output/labels/test'      # 预测标签目录
    output_dir = './evaluation_results'    # 输出目录
    
    # 执行评估
    mAP50, mAP50_95 = evaluate_detection(gt_dir, pred_dir, output_dir)
    
    print("=" * 50)
    print("📊 评估结果:")
    print(f"mAP50: {mAP50:.4f}")
    print(f"mAP50-95: {mAP50_95:.4f}")
    print(f"结果保存在: {output_dir}")
    print("=" * 50)
    
    # 显示生成的文件
    result_files = os.listdir(output_dir)
    print("📁 生成的文件:")
    for file in result_files:
        print(f"  - {file}")