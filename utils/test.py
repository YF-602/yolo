from . import *


# 使用示例
if __name__ == "__main__":
    gt_dir = './datasets/labels/test'      # 真实标签目录
    pred_dir = './output/labels/test'      # 预测标签目录
    output_dir = './evaluation_results'    # 输出目录
    
    # 执行评估
    class_names = ['car', 'person', 'bicycle']
    threshold = 0.5

    mAP50, mAP50_95 = evaluate_detection(gt_dir, pred_dir, class_names, threshold, output_dir)
    
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