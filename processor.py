import cv2
import os
import numpy as np
from collections import deque
from sklearn.cluster import KMeans

from stabilizer import Stabilizer
from detector import Detector

class DroneVideoProcessor:
    """
    Combines:
    - Video capture
    - Optional advanced stabilization
    - Multi-mode detection
    - Rolling mask history
    - K-Means grouping
    - Strict bounding box limit
    - Dynamic Auto Ratio with fallback scanning
    """

    def __init__(
        self,
        video_path,
        mode='floodfill',
        ratio=0.8,
        use_auto_ratio=False,
        max_objects=3,
        morph_kernel=3,
        morph_iterations=2,
        alpha=0.5,
        use_history=True,
        max_area=50000,
        use_stabilization=False,
        smoothing_radius=5,
        ransac_thresh=3.0,
        use_kmeans=False,
        kmeans_clusters=3
    ):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.use_auto_ratio = use_auto_ratio
        # We'll keep a separate dynamic_ratio that starts from 0
        self.dynamic_ratio = 0.0  
        self.missed_frames = 0  # how many consecutive frames found 0 objects

        self.max_objects = max_objects
        self.alpha = alpha
        self.use_history = use_history
        self.max_area = max_area
        self.use_kmeans = use_kmeans
        self.kmeans_clusters = kmeans_clusters

        self.history_length = 5
        self.masks_history = deque(maxlen=self.history_length)

        self.current_frame_idx = 0

        # Stabilizer
        self.use_stabilization = use_stabilization
        self.stabilizer = Stabilizer(smoothing_radius=smoothing_radius, ransac_thresh=ransac_thresh)

        # Detector
        self.detector = Detector(mode=mode, ratio=ratio, morph_kernel=morph_kernel, morph_iterations=morph_iterations)

    def release(self):
        self.cap.release()

    # --- Parameter setters
    def set_mode(self, mode):
        self.detector.set_mode(mode)

    def set_ratio(self, val):
        self.detector.ratio = val

    def set_use_auto_ratio(self, flag):
        self.use_auto_ratio = flag

    def get_auto_ratio_value(self):
        return self.dynamic_ratio

    def set_max_objects(self, val):
        self.max_objects = val

    def set_morph_kernel(self, val):
        self.detector.morph_kernel = val

    def set_morph_iterations(self, val):
        self.detector.morph_iterations = val

    def set_alpha(self, val):
        self.alpha = val

    def set_use_history(self, flag):
        self.use_history = flag
        if not flag:
            self.masks_history.clear()

    def set_max_area(self, val):
        self.max_area = val

    def set_use_stabilization(self, flag):
        self.use_stabilization = flag
        self.stabilizer.reset()

    def set_smoothing_radius(self, val):
        self.stabilizer.smoothing_radius = val

    def set_ransac_thresh(self, val):
        self.stabilizer.ransac_thresh = val

    def set_use_kmeans(self, flag):
        self.use_kmeans = flag

    def set_kmeans_clusters(self, val):
        self.kmeans_clusters = val

    # --- Video read/seek
    def seek_frame(self, frame_idx):
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = frame_idx
            self.masks_history.clear()
            self.stabilizer.reset()
            self.missed_frames = 0
            return frame
        return None

    def get_next_frame(self):
        if self.current_frame_idx >= self.total_frames:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.current_frame_idx += 1

        if self.use_stabilization:
            frame = self.stabilizer.stabilize(frame)

        return frame

    # --- Fallback scanning if we keep missing
    def _fallback_rescan(self, frame, start_ratio=0.0):
        """
        If we've missed objects for multiple consecutive frames,
        do a quick partial scan from 'start_ratio' up to 1.0 
        in bigger steps (e.g., 0.05) to find a ratio 
        yielding 1-2 bounding boxes. If found, return that ratio; else None.
        """
        step = 0.05
        r = start_ratio
        while r <= 1.0:
            tmp_mask = self.detector.detect_objects(frame, ratio=r)
            cnts, _ = cv2.findContours(tmp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_count = 0
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w*h <= self.max_area:
                    valid_count += 1

            if 1 <= valid_count <= 2:
                # Found a decent ratio
                return r

            r += step

        return None

    # --- Main pipeline
    def process_frame(self, frame):
        if frame is None:
            return None

        if self.use_auto_ratio and self.detector.mode == 'floodfill':
            # Step 1) Detect with dynamic_ratio
            mask = self.detector.detect_objects(frame, ratio=self.dynamic_ratio)

            # Step 2) Count bounding boxes
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_count = 0
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if w*h <= self.max_area:
                    valid_count += 1

            # Step 3) Incremental logic (aim for 1-2 objects)
            step = 0.02
            if valid_count == 0:
                # No objects => ratio too small
                self.missed_frames += 1
                self.dynamic_ratio += step
            elif valid_count >= 3:
                # Too many => ratio too big
                self.missed_frames = 0
                self.dynamic_ratio -= step
            else:
                # 1 or 2 => good, do nothing
                self.missed_frames = 0

            # Step 4) clamp ratio
            self.dynamic_ratio = max(0.0, min(1.0, self.dynamic_ratio))

            # Step 5) fallback if we keep missing
            MISS_THRESHOLD = 3
            if self.missed_frames >= MISS_THRESHOLD:
                found_ratio = self._fallback_rescan(frame, start_ratio=self.dynamic_ratio)
                if found_ratio is not None:
                    self.dynamic_ratio = found_ratio
                self.missed_frames = 0

        else:
            # Normal detection (manual ratio or mog2/knn)
            mask = self.detector.detect_objects(frame)

        # Rolling history
        if self.use_history:
            self.masks_history.append(mask)
            hist_mask = np.zeros_like(mask, dtype=np.uint8)
            for m in self.masks_history:
                hist_mask = cv2.bitwise_or(hist_mask, m)
            final_mask = hist_mask
        else:
            final_mask = mask

        # Find bounding boxes
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area <= self.max_area:
                boxes.append((x, y, w, h))

        # K-Means grouping
        if self.use_kmeans and len(boxes) > 0:
            centers = np.array([[bx + bw/2, by + bh/2] for (bx, by, bw, bh) in boxes], dtype=np.float32)
            n_clusters = min(self.kmeans_clusters, len(centers))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(centers)

            grouped_dict = {}
            for i, label in enumerate(labels):
                if label not in grouped_dict:
                    grouped_dict[label] = []
                grouped_dict[label].append(boxes[i])

            new_boxes = []
            for label, group in grouped_dict.items():
                minx = min(b[0] for b in group)
                miny = min(b[1] for b in group)
                maxx = max((b[0] + b[2]) for b in group)
                maxy = max((b[1] + b[3]) for b in group)
                new_boxes.append((minx, miny, maxx - minx, maxy - miny))

            boxes = new_boxes

        # Sort by area desc, keep up to max_objects
        boxes.sort(key=lambda b: b[2]*b[3], reverse=True)
        boxes = boxes[:self.max_objects]

        # Draw
        output_frame = frame.copy()
        overlay = output_frame.copy()

        for (x, y, w, h) in boxes:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)

        output_frame = cv2.addWeighted(overlay, self.alpha, output_frame, 1 - self.alpha, 0)

        for (x, y, w, h) in boxes:
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return output_frame
