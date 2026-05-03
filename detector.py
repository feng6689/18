import cv2
import numpy as np

class SignDetector:
    def __init__(self):
        self.red_lower1 = np.array([0, 120, 70])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 120, 70])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.blue_lower = np.array([90, 120, 70])
        self.blue_upper = np.array([130, 255, 255])
        
        self.yellow_lower = np.array([15, 120, 70])
        self.yellow_upper = np.array([35, 255, 255])

    def detect_by_color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        
        return {
            'red': red_mask,
            'blue': blue_mask,
            'yellow': yellow_mask
        }

    def find_shapes(self, mask, min_area=500):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            vertices = len(approx)
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            shape_type = 'unknown'
            if vertices == 3:
                shape_type = 'triangle'
            elif vertices == 4:
                if 0.85 <= aspect_ratio <= 1.15:
                    shape_type = 'square'
                else:
                    shape_type = 'rectangle'
            elif vertices >= 8:
                shape_type = 'circle'
            else:
                (x_circle, y_circle), radius = cv2.minEnclosingCircle(contour)
                circle_area = np.pi * radius * radius
                if area / circle_area > 0.7:
                    shape_type = 'circle'
            
            shapes.append({
                'contour': contour,
                'approx': approx,
                'vertices': vertices,
                'shape_type': shape_type,
                'area': area,
                'bounding_rect': (x, y, w, h),
                'center': (x + w // 2, y + h // 2)
            })
        
        return shapes

    def detect_candidate_regions(self, image, min_area=500):
        color_masks = self.detect_by_color(image)
        
        all_candidates = []
        
        for color_name, mask in color_masks.items():
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            shapes = self.find_shapes(mask, min_area)
            
            for shape in shapes:
                x, y, w, h = shape['bounding_rect']
                roi = image[y:y+h, x:x+w]
                
                all_candidates.append({
                    'roi': roi,
                    'bounding_rect': shape['bounding_rect'],
                    'shape_type': shape['shape_type'],
                    'color': color_name,
                    'area': shape['area'],
                    'contour': shape['contour']
                })
        
        return all_candidates

    def draw_detections(self, image, detections, color=(0, 255, 0), thickness=2):
        output = image.copy()
        
        for det in detections:
            x, y, w, h = det['bounding_rect']
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
            
            label = f"{det.get('label', 'unknown')}"
            cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        
        return output
