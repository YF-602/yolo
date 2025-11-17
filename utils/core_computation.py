import numpy as np


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