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
    - Multi-mode detection (floodfill, mog2, knn)
    - Rolling mask history
    - K-Means grouping
    - Strict bounding box limit
    """

    def __init__(
        self,
        video_path,
        mode='floodfill',
        ratio=0.8,
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

        self.max_objects = max_objects
        self.alpha = alpha
        self.use_history = use_history
        self.max_area = max_area
        self.use_kmeans = use_kmeans
        self.kmeans_clusters = kmeans_clusters

        # Rolling mask history
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

    # --- Core processing
    def process_frame(self, frame):
        """
        1) Build a mask using the detector.
        2) Optionally merge with rolling mask history.
        3) Extract bounding boxes, do K-Means grouping, limit to self.max_objects.
        4) Fill + bounding box on the original frame, return final BGR.
        """
        if frame is None:
            return None

        # 1) Detector
        mask = self.detector.detect_objects(frame)

        # 2) Rolling history
        if self.use_history:
            self.masks_history.append(mask)
            hist_mask = np.zeros_like(mask, dtype=np.uint8)
            for m in self.masks_history:
                hist_mask = cv2.bitwise_or(hist_mask, m)
            final_mask = hist_mask
        else:
            final_mask = mask

        # 3) Find contours, skip area > max_area
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area <= self.max_area:
                boxes.append((x, y, w, h))

        # Optional K-Means grouping
        if self.use_kmeans and len(boxes) > 0:
            # cluster bounding box centers
            centers = np.array(
                [[x + w/2, y + h/2] for (x, y, w, h) in boxes], dtype=np.float32
            )
            n_clusters = min(self.kmeans_clusters, len(centers))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(centers)

            grouped_boxes = {}
            for i, label in enumerate(labels):
                if label not in grouped_boxes:
                    grouped_boxes[label] = []
                grouped_boxes[label].append(boxes[i])

            new_boxes = []
            for label, group in grouped_boxes.items():
                minx = min(b[0] for b in group)
                miny = min(b[1] for b in group)
                maxx = max((b[0] + b[2]) for b in group)
                maxy = max((b[1] + b[3]) for b in group)
                new_boxes.append((minx, miny, maxx - minx, maxy - miny))

            boxes = new_boxes

        # Sort by area descending, keep up to max_objects
        boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
        boxes = boxes[:self.max_objects]

        # 4) Fill + bounding box
        output_frame = frame.copy()
        overlay = output_frame.copy()

        for (x, y, w, h) in boxes:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)

        # alpha-blend
        output_frame = cv2.addWeighted(overlay, self.alpha, output_frame, 1 - self.alpha, 0)

        # draw bounding boxes in opaque
        for (x, y, w, h) in boxes:
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return output_frame
