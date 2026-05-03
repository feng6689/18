import cv2
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, '1')
    temp_dir = os.path.join(base_dir, '2')
    out_dir = os.path.join(base_dir, 'output')
    csv_file = os.path.join(base_dir, '1.csv')
    hist_file = os.path.join(base_dir, '1.jpg')
    
    os.makedirs(out_dir, exist_ok=True)
    
    templates = {}
    for f in os.listdir(temp_dir):
        if f.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(temp_dir, f)
            img = cv2.imread(path)
            name = os.path.splitext(f)[0]
            templates[name] = img
            print(f"加载模板: {name}")
    
    stats = defaultdict(int)
    csv_data = []
    
    for img_name in os.listdir(img_dir):
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        
        print(f"\n处理: {img_name}")
        
        all_dets = []
        for name, temp in templates.items():
            temp_h, temp_w = temp.shape[:2]
            img_h, img_w = img.shape[:2]
            
            scales = np.linspace(0.2, 4.0, 30)
            
            for scale in scales:
                new_w = int(temp_w * scale)
                new_h = int(temp_h * scale)
                
                if new_w < 30 or new_h < 30:
                    continue
                if new_w > img_w or new_h > img_h:
                    continue
                
                resized = cv2.resize(temp, (new_w, new_h))
                
                result = cv2.matchTemplate(img, resized, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > 0.15:
                    all_dets.append({
                        'x': max_loc[0],
                        'y': max_loc[1],
                        'w': new_w,
                        'h': new_h,
                        'conf': max_val,
                        'name': name
                    })
        
        all_dets.sort(key=lambda d: -d['conf'])
        final_dets = []
        used = set()
        
        for det in all_dets:
            overlap = False
            x1, y1, w1, h1 = det['x'], det['y'], det['w'], det['h']
            
            for f in final_dets:
                x2, y2, w2, h2 = f['x'], f['y'], f['w'], f['h']
                
                ix1 = max(x1, x2)
                iy1 = max(y1, y2)
                ix2 = min(x1 + w1, x2 + w2)
                iy2 = min(y1 + h1, y2 + h2)
                
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    min_a = min(w1 * h1, w2 * h2)
                    iou = inter / min_a
                    
                    if iou > 0.25:
                        overlap = True
                        break
            
            if not overlap:
                final_dets.append(det)
        
        print(f"  检测到 {len(final_dets)} 个标志")
        
        out_img = img.copy()
        for det in final_dets:
            x, y, w, h = det['x'], det['y'], det['w'], det['h']
            name = det['name']
            conf = det['conf']
            
            color = (0, 255, 0)
            if name == 'stop':
                color = (0, 0, 255)
            elif name == 'slow':
                color = (0, 255, 255)
            elif name == '80':
                color = (255, 0, 0)
            
            cv2.rectangle(out_img, (x, y), (x + w, y + h), color, 4)
            
            text = f"{name} ({conf:.2f})"
            cv2.rectangle(out_img, (x, y - 30), (x + 200, y), color, -1)
            cv2.putText(out_img, text, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            stats[name] += 1
            csv_data.append([img_name, name, 1])
        
        cv2.imwrite(os.path.join(out_dir, img_name), out_img)
        print(f"  保存到: output/{img_name}")
    
    with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['图片名', '标志类别', '数量'])
        writer.writerows(csv_data)
        
        writer.writerow([])
        writer.writerow(['汇总', '', ''])
        for name, cnt in stats.items():
            writer.writerow(['', name, cnt])
    
    print(f"\n保存CSV: {csv_file}")
    
    if stats:
        plt.figure(figsize=(10, 6))
        names = list(stats.keys())
        counts = list(stats.values())
        
        colors = []
        for name in names:
            if name == 'stop':
                colors.append('red')
            elif name == 'slow':
                colors.append('gold')
            elif name == '80':
                colors.append('blue')
            else:
                colors.append('green')
        
        bars = plt.bar(names, counts, color=colors)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, str(count),
                    ha='center', va='bottom', fontsize=12)
        
        plt.xlabel('标志类别', fontsize=14)
        plt.ylabel('数量', fontsize=14)
        plt.title('交通标志统计', fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(hist_file, dpi=150)
        plt.close()
        print(f"保存直方图: {hist_file}")
    
    print("\n" + "="*50)
    print("完成统计:")
    for name, cnt in stats.items():
        print(f"  {name}: {cnt} 个")
    print("="*50)

if __name__ == "__main__":
    import numpy as np
    main()
