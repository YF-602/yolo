from ultralytics import YOLO
import os
import shutil


def get_cls(cls):
    if(cls == 0): return [1, 'person']
    elif(cls == 1): return [2, 'bicycle']
    else: return [0, 'car']


def save_detection_results(results, output_dir='./output'):
    """保存检测结果：图像和标签文件"""
    
    # 创建输出目录
    images_dir = os.path.join(output_dir, 'images/test')
    labels_dir = os.path.join(output_dir, 'labels/test')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    all_boxes = []
    
    for i, result in enumerate(results):
        # 获取原始图像文件名（不带扩展名）
        orig_path = result.path
        filename = os.path.splitext(os.path.basename(orig_path))[0]
        
        # 保存带标注框的图像
        result.save(filename=os.path.join(images_dir, f'{filename}.jpg'))
        
        # 保存标签文件
        label_path = os.path.join(labels_dir, f'{filename}.txt')
        boxes_info = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            with open(label_path, 'w') as f:
                for j in range(len(result.boxes)):
                    # 获取归一化坐标 (中心点x, 中心点y, 宽度, 高度)
                    xywhn = result.boxes.xywhn[j]
                    center_x, center_y, width, height = xywhn.tolist()
                    
                    # 获取类别信息并转换
                    coco_cls = int(result.boxes.cls[j])
                    target_cls, cls_name = get_cls(coco_cls)
                    confidence = result.boxes.conf[j].item()
                    
                    # 写入标签文件：目标类别 中心点x 中心点y 框宽 框高
                    f.write(f"{target_cls} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                    
                    # 保存用于计算mAP的信息
                    boxes_info.append({
                        'class_id': target_cls,
                        'class_name': cls_name,
                        'confidence': confidence,
                        'bbox': [center_x, center_y, width, height]
                    })
                    
                    print(f"图像 {filename}: 检测到 {cls_name} (置信度: {confidence:.3f})")
        
        all_boxes.append({
            'filename': filename,
            'boxes': boxes_info
        })
    
    print(f"\n✅ 结果已保存到: {output_dir}")
    print(f"📁 图像文件: {images_dir}")
    print(f"📁 标签文件: {labels_dir}")
    
    return all_boxes
