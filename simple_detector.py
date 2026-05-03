import cv2
import numpy as np
import os

class SimpleDetector:
    def __init__(self, template_folder):
        self.templates = {}
        self.load_templates(template_folder)
        
    def load_templates(self, folder):
        for filename in os.listdir(folder):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(folder, filename)
                img = cv2.imread(path)
                if img is not None:
                    name = os.path.splitext(filename)[0]
                    self.templates[name] = img
                    print(f"加载模板: {name}, 尺寸: {img.shape}")
    
    def match_template(self, image, template, name):
        detections = []
        img_h, img_w = image.shape[:2]
        temp_h, temp_w = template.shape[:2]
        
        scales = np.linspace(0.3, 3.0, 20)
        
        for scale in scales:
            new_w = int(temp_w * scale)
            new_h = int(temp_h * scale)
            
            if new_w < 20 or new_h < 20:
                continue
            if new_w > img_w or new_h > img_h:
                continue
            
            resized = cv2.resize(template, (new_w, new_h))
            
            result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.3:
                detections.append({
                    'bbox': (max_loc[0], max_loc[1], new_w, new_h),
                    'confidence': max_val,
                    'name': name
                })
        
        return detections
    
    def detect(self, image):
        all_detections = []
        
        for name, template in self.templates.items():
            dets = self.match_template(image, template, name)
            all_detections.extend(dets)
        
        final = []
        used = set()
        
        all_detections.sort(key=lambda x: -x['confidence'])
        
        for det in all_detections:
            x, y, w, h = det['bbox']
            overlap = False
            
            for i, f in enumerate(final):
                fx, fy, fw, fh = f['bbox']
                
                inter_x1 = max(x, fx)
                inter_y1 = max(y, fy)
                inter_x2 = min(x + w, fx + fw)
                inter_y2 = min(y + h, fy + fh)
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    min_area = min(w * h, fw * fh)
                    iou = inter_area / min_area
                    
                    if iou > 0.3:
                        overlap = True
                        break
            
            if not overlap:
                final.append(det)
        
        return final
