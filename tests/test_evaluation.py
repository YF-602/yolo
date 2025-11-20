# tests/test_evaluation.py
import os
import pytest
from utils import evaluate_detection

class TestEvaluation:
    """评估功能测试类"""
    
    def test_evaluation_basic(self):
        """基础评估测试"""
        gt_dir = './datasets/labels/test'
        pred_dir = './output/labels/test'
        output_dir = './evaluation_results'
        
        class_names = ['car', 'person', 'bicycle']
        iou_threshold = 0.5

        mAP50, mAP50_95 = evaluate_detection(gt_dir, pred_dir, class_names, iou_threshold, output_dir)
        
        # 断言结果在合理范围内
        assert 0 <= mAP50 <= 1
        assert 0 <= mAP50_95 <= 1
        assert os.path.exists(output_dir)

if __name__ == "__main__":
    pytest.main([__file__])