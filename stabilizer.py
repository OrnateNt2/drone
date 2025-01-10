import cv2
import numpy as np
from collections import deque

class Stabilizer:
    """
    Advanced frame stabilization via:
    - RANSAC-based partial-affine transform
    - Smoothing transforms over a window
    """

    def __init__(self, smoothing_radius=5, ransac_thresh=3.0, max_corners=300):
        """
        :param smoothing_radius: how many transforms to store & average for smoother motion
        :param ransac_thresh: RANSAC reprojection threshold
        :param max_corners: max corners to track (for optical flow)
        """
        self.smoothing_radius = smoothing_radius
        self.ransac_thresh = ransac_thresh
        self.max_corners = max_corners

        # We'll store recent transforms to smooth them
        self.transform_buffer = deque(maxlen=50)
        self.prev_gray = None

    def reset(self):
        """Clears buffers and resets the previous frame."""
        self.prev_gray = None
        self.transform_buffer.clear()

    def stabilize(self, frame_bgr):
        """
        Steps:
          1) Convert current frame to gray.
          2) If no prev_gray, store it & return frame as-is.
          3) Otherwise, find good features, track with optical flow, 
             estimate partial-affine transform with RANSAC, store transform.
          4) Compute a smoothed transform over the last 'smoothing_radius'.
          5) Warp the frame by that transform.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame_bgr

        # 1) Good features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=30
        )
        if prev_pts is None:
            self.prev_gray = gray
            return frame_bgr

        # 2) Optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_pts, None)
        good_old = prev_pts[status == 1]
        good_new = curr_pts[status == 1]

        if len(good_old) < 4:
            self.prev_gray = gray
            return frame_bgr

        # 3) Estimate partial-affine transform with RANSAC
        transform_matrix, _ = cv2.estimateAffinePartial2D(
            good_old,
            good_new,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
            maxIters=2000,
            confidence=0.98,
            refineIters=10
        )
        if transform_matrix is None:
            self.prev_gray = gray
            return frame_bgr

        # Convert 2x3 to 3x3
        full_transform = np.eye(3, dtype=np.float32)
        full_transform[:2, :] = transform_matrix

        self.transform_buffer.append(full_transform.copy())

        # 4) Smooth transform
        smoothed_transform = self._smooth_transforms()

        # 5) Warp current frame
        stabilized = cv2.warpPerspective(
            frame_bgr,
            smoothed_transform,
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            flags=cv2.INTER_LINEAR
        )

        self.prev_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        return stabilized

    def _smooth_transforms(self):
        """
        Average the last 'smoothing_radius' transforms for stable motion.
        """
        if len(self.transform_buffer) == 0:
            return np.eye(3, dtype=np.float32)

        radius = min(self.smoothing_radius, len(self.transform_buffer))
        transforms = list(self.transform_buffer)[-radius:]
        accum = np.zeros((3, 3), dtype=np.float32)
        for t in transforms:
            accum += t
        accum /= radius
        return accum
