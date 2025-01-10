import sys
import os
import cv2
import traceback

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QCheckBox, QStyleFactory, QDoubleSpinBox, QComboBox
)

from processor import DroneVideoProcessor

class VideoWorker(QObject):
    """
    Runs in a separate thread to fetch frames and process them.
    """
    frameReady = pyqtSignal(object)  # Emitted with the processed frame (BGR) 
    finished = pyqtSignal()

    def __init__(self, processor: DroneVideoProcessor):
        super().__init__()
        self.processor = processor
        self.running = True

    def run(self):
        """ Main loop: read frames, process, emit. Stop if end or not running. """
        try:
            while self.running:
                frame = self.processor.get_next_frame()
                if frame is None:
                    break
                processed = self.processor.process_frame(frame)
                self.frameReady.emit(processed)
        except Exception as e:
            print("VideoWorker exception:", e)
            traceback.print_exc()
        self.finished.emit()

    def stop(self):
        self.running = False


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Detection + Stabilization + K-Means + BG Subtraction")

        # Hard-coded path or override with --video
        self.video_path = os.path.join("input", "5.mp4")

        # Initialize the processor
        self.processor = DroneVideoProcessor(
            video_path=self.video_path,
            mode='floodfill',         # can be 'floodfill', 'mog2', or 'knn'
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
        )

        self.worker_thread = None
        self.worker = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1) Video display label
        self.video_label = QLabel("Video")
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        # Let it resize with the window
        main_layout.addWidget(self.video_label)

        # 2) Timeline slider
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, self.processor.total_frames - 1)
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderReleased.connect(self.on_timeline_slider_released)
        main_layout.addWidget(self.timeline_slider)

        # 3) Controls
        controls_layout = QHBoxLayout()

        # Mode combo
        controls_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["floodfill", "mog2", "knn"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        controls_layout.addWidget(self.mode_combo)

        # Ratio slider
        controls_layout.addWidget(QLabel("Ratio:"))
        self.ratio_slider = QSlider(Qt.Horizontal)
        self.ratio_slider.setRange(0, 100)
        self.ratio_slider.setValue(int(self.processor.detector.ratio * 100))
        self.ratio_slider.valueChanged.connect(self.on_ratio_changed)
        controls_layout.addWidget(self.ratio_slider)

        # Max objects
        controls_layout.addWidget(QLabel("MaxObj:"))
        self.maxobj_spin = QSpinBox()
        self.maxobj_spin.setRange(1, 20)
        self.maxobj_spin.setValue(self.processor.max_objects)
        self.maxobj_spin.valueChanged.connect(self.on_max_objects_changed)
        controls_layout.addWidget(self.maxobj_spin)

        # Kernel spin
        controls_layout.addWidget(QLabel("Kernel:"))
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 21)
        self.kernel_spin.setValue(self.processor.detector.morph_kernel)
        self.kernel_spin.valueChanged.connect(self.on_kernel_changed)
        controls_layout.addWidget(self.kernel_spin)

        # Iter spin
        controls_layout.addWidget(QLabel("Iter:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 10)
        self.iter_spin.setValue(self.processor.detector.morph_iterations)
        self.iter_spin.valueChanged.connect(self.on_iter_changed)
        controls_layout.addWidget(self.iter_spin)

        # Alpha spin
        controls_layout.addWidget(QLabel("Alpha:"))
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(self.processor.alpha)
        self.alpha_spin.valueChanged.connect(self.on_alpha_changed)
        controls_layout.addWidget(self.alpha_spin)

        # Alpha slider
        alpha_slider_label = QLabel("Alpha Slider:")
        controls_layout.addWidget(alpha_slider_label)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(int(self.processor.alpha * 100))
        self.alpha_slider.valueChanged.connect(self.on_alpha_slider_changed)
        controls_layout.addWidget(self.alpha_slider)

        # Use history
        self.history_check = QCheckBox("Use History")
        self.history_check.setChecked(self.processor.use_history)
        self.history_check.stateChanged.connect(self.on_history_changed)
        controls_layout.addWidget(self.history_check)

        # MaxArea
        controls_layout.addWidget(QLabel("MaxArea:"))
        self.maxarea_spin = QSpinBox()
        self.maxarea_spin.setRange(100, 9999999)
        self.maxarea_spin.setValue(self.processor.max_area)
        self.maxarea_spin.valueChanged.connect(self.on_max_area_changed)
        controls_layout.addWidget(self.maxarea_spin)

        # Stabilization
        self.stab_check = QCheckBox("Stabilization")
        self.stab_check.setChecked(self.processor.use_stabilization)
        self.stab_check.stateChanged.connect(self.on_stab_changed)
        controls_layout.addWidget(self.stab_check)

        # Smoothing radius
        controls_layout.addWidget(QLabel("SmoothRad:"))
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(1, 50)
        self.smooth_spin.setValue(self.processor.stabilizer.smoothing_radius)
        self.smooth_spin.valueChanged.connect(self.on_smooth_radius_changed)
        controls_layout.addWidget(self.smooth_spin)

        # RANSAC
        controls_layout.addWidget(QLabel("RANSAC:"))
        self.ransac_spin = QDoubleSpinBox()
        self.ransac_spin.setRange(0.1, 10.0)
        self.ransac_spin.setSingleStep(0.1)
        self.ransac_spin.setValue(self.processor.stabilizer.ransac_thresh)
        self.ransac_spin.valueChanged.connect(self.on_ransac_changed)
        controls_layout.addWidget(self.ransac_spin)

        # K-Means
        self.kmeans_check = QCheckBox("Use K-Means")
        self.kmeans_check.setChecked(self.processor.use_kmeans)
        self.kmeans_check.stateChanged.connect(self.on_kmeans_changed)
        controls_layout.addWidget(self.kmeans_check)

        # K-Means clusters
        controls_layout.addWidget(QLabel("KClusters:"))
        self.kmeans_spin = QSpinBox()
        self.kmeans_spin.setRange(1, 20)
        self.kmeans_spin.setValue(self.processor.kmeans_clusters)
        self.kmeans_spin.valueChanged.connect(self.on_kmeans_clusters_changed)
        controls_layout.addWidget(self.kmeans_spin)

        main_layout.addLayout(controls_layout)

        # 4) Playback buttons
        buttons_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_video)
        buttons_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_video)
        buttons_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        buttons_layout.addWidget(self.stop_button)

        main_layout.addLayout(buttons_layout)

        # Let's allow resizing
        self.resize(1200, 700)

    # Worker thread controls
    def play_video(self):
        if self.worker_thread is not None:
            # Already running
            return
        self.worker_thread = QThread()
        self.worker = VideoWorker(self.processor)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.frameReady.connect(self.update_video_label)
        self.worker.finished.connect(self.on_worker_finished)

        self.worker_thread.start()

    def pause_video(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None

    def stop_video(self):
        self.pause_video()
        self.seek_frame(0)

    def on_worker_finished(self):
        self.pause_video()

    # Seeking
    def seek_frame(self, frame_idx):
        frame = self.processor.seek_frame(frame_idx)
        if frame is not None:
            processed = self.processor.process_frame(frame)
            self.update_video_label(processed)
            self.timeline_slider.setValue(frame_idx)

    def on_timeline_slider_released(self):
        new_idx = self.timeline_slider.value()
        self.seek_frame(new_idx)

    # Display
    def update_video_label(self, frame_bgr):
        if frame_bgr is None:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimage = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        # Scale to label size
        pixmap = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)
        self.timeline_slider.setValue(self.processor.current_frame_idx)

    def closeEvent(self, event):
        self.pause_video()
        self.processor.release()
        super().closeEvent(event)

    # Callbacks
    def on_mode_changed(self, text):
        self.processor.set_mode(text)

    def on_ratio_changed(self, val):
        ratio = val / 100.0
        self.processor.set_ratio(ratio)

    def on_max_objects_changed(self, val):
        self.processor.set_max_objects(val)

    def on_kernel_changed(self, val):
        self.processor.set_morph_kernel(val)

    def on_iter_changed(self, val):
        self.processor.set_morph_iterations(val)

    def on_alpha_changed(self, val):
        self.alpha_slider.setValue(int(val * 100))
        self.processor.set_alpha(val)

    def on_alpha_slider_changed(self, slider_value):
        alpha = slider_value / 100.0
        self.alpha_spin.setValue(alpha)
        self.processor.set_alpha(alpha)

    def on_history_changed(self, state):
        flag = (state == Qt.Checked)
        self.processor.set_use_history(flag)

    def on_max_area_changed(self, val):
        self.processor.set_max_area(val)

    def on_stab_changed(self, state):
        flag = (state == Qt.Checked)
        self.processor.set_use_stabilization(flag)

    def on_smooth_radius_changed(self, val):
        self.processor.set_smoothing_radius(val)

    def on_ransac_changed(self, val):
        self.processor.set_ransac_thresh(val)

    def on_kmeans_changed(self, state):
        flag = (state == Qt.Checked)
        self.processor.set_use_kmeans(flag)

    def on_kmeans_clusters_changed(self, val):
        self.processor.set_kmeans_clusters(val)


def main():
    import argparse
    from PyQt5.QtWidgets import QApplication

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    args = parser.parse_args()

    QApplication.setStyle(QStyleFactory.create("Fusion"))

    app = QApplication(sys.argv)
    player = VideoPlayer()

    if args.video and os.path.isfile(args.video):
        player.video_path = args.video
        player.processor = DroneVideoProcessor(video_path=args.video)

    player.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
