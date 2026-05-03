import cv2
import numpy as np
import os

class TemplateMatcher:
    def __init__(self, template_folder):
        self.template_folder = template_folder
        self.templates = {}
        self.template_names = {}
        self._load_templates()

    def _load_templates(self):
        if not os.path.exists(self.template_folder):
            raise ValueError(f"模板文件夹不存在: {self.template_folder}")
        
        for filename in os.listdir(self.template_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(self.template_folder, filename)
                template = cv2.imread(filepath)
                
                if template is not None:
                    name = os.path.splitext(filename)[0]
                    self.templates[name] = template
                    self.template_names[name] = name
                    print(f"加载模板: {name}")
        
        if not self.templates:
            raise ValueError("没有找到任何模板图片")

    def preprocess_image(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray = cv2.equalizeHist(gray)
        return gray

    def detect_multiscale(self, image, threshold=0.3):
        detections = []
        
        for template_name, template in self.templates.items():
            template_gray = self.preprocess_image(template)
            image_gray = self.preprocess_image(image)
            
            h, w = template_gray.shape
            
            scales = np.linspace(0.5, 3.0, 10)[::-1]
            
            for scale in scales:
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                if new_w < 20 or new_h < 20:
                    continue
                if new_w > image_gray.shape[1] * 0.8 or new_h > image_gray.shape[0] * 0.8:
                    continue
                
                resized = cv2.resize(template_gray, (new_w, new_h))
                
                try:
                    result = cv2.matchTemplate(image_gray, resized, cv2.TM_CCOEFF_NORMED)
                    
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val >= threshold:
                        top_left = max_loc
                        bottom_right = (top_left[0] + new_w, top_left[1] + new_h)
                        
                        x = top_left[0]
                        y = top_left[1]
                        w_box = new_w
                        h_box = new_h
                        
                        roi = image[y:y+h_box, x:x+w_box]
                        
                        detections.append({
                            'roi': roi,
                            'bounding_rect': (x, y, w_box, h_box),
                            'label': template_name,
                            'confidence': max_val,
                            'classification_success': True,
                            'shape_type': 'unknown',
                            'color': 'unknown'
                        })
                except Exception as e:
                    continue
        
        final_detections = []
        for det in detections:
            x1, y1, w1, h1 = det['bounding_rect']
            overlapping = False
            
            for final_det in final_detections:
                x2, y2, w2, h2 = final_det['bounding_rect']
                
                overlap_x1 = max(x1, x2)
                overlap_y1 = max(y1, y2)
                overlap_x2 = min(x1 + w1, x2 + w2)
                overlap_y2 = min(y1 + h1, y2 + h2)
                
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    area1 = w1 * h1
                    area2 = w2 * h2
                    iou = overlap_area / min(area1, area2)
                    
                    if iou > 0.3:
                        overlapping = True
                        if det['confidence'] > final_det['confidence']:
                            final_detections.remove(final_det)
                            final_detections.append(det)
                        break
            
            if not overlapping:
                final_detections.append(det)
        
        return final_detections

    def classify(self, image, threshold=0.25):
        best_match = None
        best_score = 0.0
        
        for name, template in self.templates.items():
            try:
                img_gray = self.preprocess_image(image)
                temp_gray = self.preprocess_image(template)
                
                if img_gray.shape[0] < 10 or img_gray.shape[1] < 10:
                    continue
                
                scales = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]
                
                for scale in scales:
                    new_w = int(temp_gray.shape[1] * scale)
                    new_h = int(temp_gray.shape[0] * scale)
                    
                    if new_w < 10 or new_h < 10:
                        continue
                    
                    resized = cv2.resize(temp_gray, (new_w, new_h))
                    
                    if resized.shape[0] > img_gray.shape[0] or resized.shape[1] > img_gray.shape[1]:
                        continue
                    
                    result = cv2.matchTemplate(img_gray, resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    if max_val > best_score:
                        best_score = max_val
                        best_match = name
            except Exception as e:
                continue
        
        if best_score >= threshold and best_match is not None:
            return {
                'label': best_match,
                'confidence': best_score,
                'success': True
            }
        else:
            return {
                'label': 'unknown',
                'confidence': best_score,
                'success': False
            }

    def classify_multiple(self, regions, threshold=0.25):
        results = []
        for region in regions:
            roi = region.get('roi')
            if roi is None:
                continue
            
            classification = self.classify(roi, threshold)
            result = {
                **region,
                'label': classification['label'],
                'confidence': classification['confidence'],
                'classification_success': classification['success']
            }
            results.append(result)
        
        return results
