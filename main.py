import os
import cv2
from detector import SignDetector
from template_matcher import TemplateMatcher
from statistics_visualizer import StatisticsVisualizer

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    image_folder = os.path.join(current_dir, '1')
    template_folder = os.path.join(current_dir, '2')
    output_folder = os.path.join(current_dir, 'output')
    csv_path = os.path.join(current_dir, '1.csv')
    histogram_path = os.path.join(current_dir, '1.jpg')
    
    if not os.path.exists(image_folder):
        print(f"错误: 图片文件夹不存在: {image_folder}")
        return
    
    if not os.path.exists(template_folder):
        print(f"错误: 模板文件夹不存在: {template_folder}")
        return
    
    os.makedirs(output_folder, exist_ok=True)
    
    print("="*60)
    print("交通标志牌检测与统计系统")
    print("="*60)
    print(f"图片文件夹: {image_folder}")
    print(f"模板文件夹: {template_folder}")
    print(f"输出文件夹: {output_folder}")
    print("="*60 + "\n")
    
    try:
        print("正在初始化检测器...")
        detector = SignDetector()
        print("检测器初始化完成\n")
        
        print("正在加载模板...")
        matcher = TemplateMatcher(template_folder)
        print("模板加载完成\n")
        
        visualizer = StatisticsVisualizer()
        
        image_files = []
        for f in os.listdir(image_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(f)
        
        if not image_files:
            print("错误: 没有找到任何图片文件")
            return
        
        print(f"找到 {len(image_files)} 张图片，开始处理...")
        print("-"*60)
        
        for idx, image_name in enumerate(image_files, 1):
            image_path = os.path.join(image_folder, image_name)
            print(f"\n[{idx}/{len(image_files)}] 正在处理: {image_name}")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"  警告: 无法读取图片 {image_name}")
                continue
            
            print(f"  图片尺寸: {image.shape[1]} x {image.shape[0]}")
            
            print("  正在检测候选区域...")
            min_area = max(100, int(image.shape[0] * image.shape[1] * 0.001))
            candidates = detector.detect_candidate_regions(image, min_area=min_area)
            print(f"  检测到 {len(candidates)} 个候选区域")
            
            print("  正在进行模板匹配分类...")
            detections = matcher.classify_multiple(candidates, threshold=0.55)
            
            valid_detections = [d for d in detections if d.get('classification_success', False)]
            print(f"  成功分类 {len(valid_detections)} 个标志牌")
            
            for det in valid_detections:
                print(f"    - 类别: {det['label']}, 置信度: {det['confidence']:.2f}, "
                      f"位置: {det['bounding_rect']}")
            
            visualizer.add_image_results(image_name, valid_detections)
            
            output_image = image.copy()
            for det in valid_detections:
                x, y, w, h = det['bounding_rect']
                label = det['label']
                confidence = det['confidence']
                
                color = (0, 255, 0)
                if det['color'] == 'red':
                    color = (0, 0, 255)
                elif det['color'] == 'blue':
                    color = (255, 0, 0)
                elif det['color'] == 'yellow':
                    color = (0, 255, 255)
                
                cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
                
                text = f"{label} ({confidence:.2f})"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(output_image, (x, y - text_h - 10), (x + text_w, y), color, -1)
                cv2.putText(output_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, output_image)
            print(f"  标注图片已保存: {output_path}")
        
        print("\n" + "="*60)
        print("处理完成，正在生成统计结果...")
        print("="*60)
        
        visualizer.generate_csv(csv_path)
        visualizer.generate_histogram(histogram_path)
        visualizer.print_summary()
        
        print("\n" + "="*60)
        print("所有任务已完成！")
        print(f"输出文件夹: {output_folder}")
        print(f"统计表格: {csv_path}")
        print(f"直方图: {histogram_path}")
        print("="*60)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
