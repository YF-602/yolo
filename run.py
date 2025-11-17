from ultralytics import YOLO
from utils.data import *


if __name__ == "__main__":
    # 1. 进行预测
    print("🚀 开始目标检测...")
    # 加载预训练模型
    model = YOLO('yolo11n.pt')

    results = model.predict(
        source='./datasets/images/test', 
        classes=[0, 1, 2],          # 只检测人、自行车、汽车
        conf=0.50,                  # 置信度阈值
        # save=True,                  # 保存结果图像
        # save_txt=True,              # 保存标签文件
        # save_conf=True,             # 在标签中保存置信度
        # exist_ok=True               # 覆盖已存在的结果
    )
    
    # 2. 保存自定义格式的结果
    print("\n💾 保存检测结果...")
    detection_results = save_detection_results(results)
    
    # 3. 评估模型性能
    print("\n📊 评估模型性能...")
    # TODO 计算系列指标
    # mAP50
    # mAP50-95
    # confusion_matrix_normalized.png
    # confusion_matrix.png
    # P_curve.png
    # R_curve.png（召回率曲线）
    # PR_curve.png（PR曲线）
    # F1_curve.png（F1曲线）
    # val_batch0_labels与val_batch0_pred
    
    # 4. 显示检测统计信息
    print(f"\n📈 检测统计:")
    total_detections = 0
    class_counts = {'car': 0, 'person': 0, 'bicycle': 0}
    
    for result in detection_results:
        total_detections += len(result['boxes'])
        for box in result['boxes']:
            class_counts[box['class_name']] += 1
    
    print(f"总检测数量: {total_detections}")
    print(f"各类别检测数量: {class_counts}")
    
    print("\n✅ 任务完成！")