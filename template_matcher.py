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

    def match_single_template(self, image, template, method=cv2.TM_CCOEFF_NORMED):
        if image is None or template is None:
            return 0.0
        
        img_gray = self.preprocess_image(image)
        temp_gray = self.preprocess_image(template)
        
        if img_gray.shape[0] < 10 or img_gray.shape[1] < 10:
            return 0.0
        
        img_h, img_w = img_gray.shape
        temp_h, temp_w = temp_gray.shape
        
        scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        best_score = 0.0
        
        for scale in scales:
            try:
                new_w = int(temp_w * scale)
                new_h = int(temp_h * scale)
                
                if new_w < 10 or new_h < 10:
                    continue
                if new_w > img_w * 2 or new_h > img_h * 2:
                    continue
                
                resized_template = cv2.resize(temp_gray, (new_w, new_h))
                
                if img_h < new_h or img_w < new_w:
                    if new_h > 0 and new_w > 0:
                        resized_img = cv2.resize(img_gray, (new_w, new_h))
                        result = cv2.matchTemplate(resized_img, resized_template, method)
                        _, max_val, _, _ = cv2.minMaxLoc(result)
                        if max_val > best_score:
                            best_score = max_val
                    continue
                
                result = cv2.matchTemplate(img_gray, resized_template, method)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    
            except Exception as e:
                continue
        
        return best_score

    def match_with_features(self, image, template):
        try:
            img_gray = self.preprocess_image(image)
            temp_gray = self.preprocess_image(template)
            
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(img_gray, None)
            kp2, des2 = orb.detectAndCompute(temp_gray, None)
            
            if des1 is None or des2 is None:
                return 0.0
            if len(des1) < 2 or len(des2) < 2:
                return 0.0
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if len(matches) == 0:
                return 0.0
            
            good_matches = [m for m in matches if m.distance < 50]
            score = len(good_matches) / max(len(kp1), len(kp2), 1)
            
            return min(score * 2, 1.0)
        except Exception as e:
            return 0.0

    def classify(self, image, threshold=0.55):
        best_match = None
        best_score = 0.0
        
        for name, template in self.templates.items():
            tm_score = self.match_single_template(image, template)
            feat_score = self.match_with_features(image, template)
            
            combined_score = tm_score * 0.7 + feat_score * 0.3
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = name
        
        if best_score >= threshold:
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

    def classify_multiple(self, regions, threshold=0.55):
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
