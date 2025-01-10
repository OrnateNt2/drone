import cv2
import numpy as np

class Detector:
    """
    Encapsulates multiple detection modes:
    - 'floodfill' (manual ratio-based bright-spot detection)
    - 'mog2' (background subtraction)
    - 'knn' (background subtraction)
    """

    def __init__(self, mode='floodfill', ratio=0.8, morph_kernel=3, morph_iterations=2):
        """
        :param mode: 'floodfill', 'mog2', or 'knn'
        :param ratio: fraction of brightest pixel for flood-fill
        :param morph_kernel: kernel size for morphological closing
        :param morph_iterations: morphological iterations
        """
        self.mode = mode
        self.ratio = ratio
        self.morph_kernel = morph_kernel
        self.morph_iterations = morph_iterations

        self.bg_subtractor = None
        self._init_bg_subtractor()

    def _init_bg_subtractor(self):
        """Create or reset the background subtractor if using 'mog2' or 'knn'."""
        if self.mode == 'mog2':
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=50, detectShadows=False
            )
        elif self.mode == 'knn':
            self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                history=500, dist2Threshold=400.0, detectShadows=False
            )
        else:
            self.bg_subtractor = None

    def set_mode(self, new_mode):
        """Set new detection mode and re-initialize if needed."""
        self.mode = new_mode
        self._init_bg_subtractor()

    def detect_objects(self, frame_bgr):
        """
        Return a binary mask of detected regions, 
        depending on the chosen mode.
        """
        if self.mode in ('mog2', 'knn'):
            # background subtraction
            # make sure bg_subtractor is not None
            if self.bg_subtractor is None:
                self._init_bg_subtractor()

            fg_mask = self.bg_subtractor.apply(frame_bgr)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel))
            cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)
            return cleaned
        else:
            # fallback to flood-fill mode
            return self._floodfill_mode(frame_bgr)

    def _floodfill_mode(self, frame_bgr):
        """
        Implementation of repeated flood-fill for bright spots
        using self.ratio * maxVal as threshold.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        combined_mask = np.zeros_like(gray, dtype=np.uint8)
        working_gray = gray.copy()

        # We'll do up to 10 repeated finds
        for _ in range(10):
            _, maxVal, _, maxLoc = cv2.minMaxLoc(working_gray)
            if maxVal <= 0:
                break

            threshold_val = self.ratio * maxVal
            mask = (working_gray >= threshold_val).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)

            flood = mask.copy()
            h, w = flood.shape
            floodmask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(flood, floodmask, maxLoc, 128)
            region_mask = (flood == 128).astype(np.uint8) * 255

            combined_mask = cv2.bitwise_or(combined_mask, region_mask)
            working_gray[region_mask > 0] = 0

        return combined_mask
