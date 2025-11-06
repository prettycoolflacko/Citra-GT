import sys
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import io

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QSizePolicy,
    QSlider, QFrame, QCheckBox
)
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtCore import Qt, QSize, QTimer

# DIPERBARUI: Menambahkan "Saturation"
ADJUSTABLE_ALGORITHMS = {
    "Brightness": {"min": 0, "max": 200, "default": 100},
    "Contrast": {"min": 0, "max": 200, "default": 100},
    "Sharpen": {"min": 0, "max": 200, "default": 100},
    "Saturation": {"min": 0, "max": 200, "default": 100}, # BARU
}

# DIPERBARUI: Menambahkan 12 filter baru
ONE_CLICK_FILTERS = [
    # Point/Color
    "Grayscale", "Negative", "Sepia", "Posterize", "Solarize",
    # Blur
    "Gaussian Blur", "Median Blur", "Bilateral Filter",
    # Edge/Artistic
    "Emboss", "Edge Enhance", "Contour",
    # Thresholding
    "Binary Threshold", "Adaptive Threshold",
    # Geometric
    "Putar 90° Kanan", "Putar 90° Kiri", "Flip Horizontal", "Flip Vertical",
    # Morphology (Basic)
    "Erode", "Dilate",
    # BARU: Advanced Morphology
    "Opening (Noise Removal)", "Closing (Fill Holes)", "Morphological Gradient",
    # BARU: Advanced Edge Detection
    "Canny Edges", "Sobel X (Vertical)", "Sobel Y (Horizontal)", "Laplacian Edges",
    # BARU: Color Space Analysis
    "View Hue Channel", "View Saturation Channel", "View Value Channel",
    # BARU: Stylization
    "Stylization (Art)", "Pencil Sketch"
]


class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Citra-GT")
        self.setGeometry(100, 100, 1400, 900)

        # State Gambar
        self.active_image = None
        self.original_image = None
        self.preview_image = None
        self.current_pixmap = None

        # State Filter
        self.active_slider_values = {}
        self.active_one_click_filters = set()
        self.committed_slider_values = {}
        self.committed_one_click_filters = set()

        # State Kamera
        self.camera = None
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_camera_frame)

        # UI
        self.sliders = {}
        self.filter_checkboxes = []
        
        self.init_ui()
        self.apply_stylesheet()
        # Inisialisasi state filter default
        self.reset_filters_and_commit()

    def init_ui(self):
        # (Fungsi init_ui tidak berubah dari v4.0)
        # UI (slider dan checkbox) dibuat secara dinamis dari
        # daftar ADJUSTABLE_ALGORITHMS dan ONE_CLICK_FILTERS.
        # Tidak perlu modifikasi di sini.
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Panel Kiri (Kontrol) ---
        left_panel = QWidget()
        left_panel.setFixedWidth(250)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        left_layout.addWidget(QLabel("<h2>Kontrol File</h2>"))
        self.btn_upload = QPushButton("  Upload Gambar")
        self.btn_upload.clicked.connect(self.upload_image)
        self.btn_upload.setIcon(QIcon.fromTheme("document-open"))
        self.btn_upload.setIconSize(QSize(24, 24))

        self.btn_save = QPushButton("  Simpan Gambar")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setIcon(QIcon.fromTheme("document-save"))
        self.btn_save.setEnabled(False)

        self.btn_reset = QPushButton("  Reset ke Asli")
        self.btn_reset.clicked.connect(self.reset_image)
        self.btn_reset.setIcon(QIcon.fromTheme("edit-undo"))
        self.btn_reset.setEnabled(False)
        
        left_layout.addWidget(self.btn_upload)
        left_layout.addWidget(self.btn_save)
        left_layout.addWidget(self.btn_reset)
        
        left_layout.addWidget(QLabel("<h2>Kontrol Kamera</h2>"))
        self.btn_start_cam = QPushButton("  Mulai Kamera")
        self.btn_start_cam.clicked.connect(self.start_camera)
        self.btn_start_cam.setIcon(QIcon.fromTheme("camera-video"))
        
        self.btn_stop_cam = QPushButton("  Stop Kamera")
        self.btn_stop_cam.clicked.connect(self.stop_camera)
        self.btn_stop_cam.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.btn_stop_cam.setEnabled(False)
        
        self.btn_snapshot = QPushButton("  Ambil Snapshot")
        self.btn_snapshot.clicked.connect(self.take_snapshot)
        self.btn_snapshot.setIcon(QIcon.fromTheme("camera-photo"))
        self.btn_snapshot.setEnabled(False)
        
        left_layout.addWidget(self.btn_start_cam)
        left_layout.addWidget(self.btn_stop_cam)
        left_layout.addWidget(self.btn_snapshot)
        left_layout.addStretch()

        # --- Panel Tengah (Display & Histogram) ---
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        
        self.image_label = QLabel("Silakan upload gambar atau mulai kamera...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        center_layout.addWidget(self.image_label, 7) 
        
        self.histogram_label = QLabel("Histogram akan muncul di sini")
        self.histogram_label.setAlignment(Qt.AlignCenter)
        self.histogram_label.setMinimumHeight(150)
        self.histogram_label.setMaximumHeight(200)
        center_layout.addWidget(self.histogram_label, 3) 
        
        # --- Panel Kanan (Algoritma) ---
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)

        # --- Bagian Adjustments (Sliders) ---
        scroll_layout.addWidget(QLabel("<h2>Adjustments</h2>"))
        self.sliders = {}
        for name, props in ADJUSTABLE_ALGORITHMS.items():
            scroll_layout.addWidget(QLabel(name))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(props["min"], props["max"])
            slider.setValue(props["default"])
            slider.valueChanged.connect(self.on_slider_changed)
            slider.setEnabled(False)
            scroll_layout.addWidget(slider)
            self.sliders[name] = slider

        self.btn_apply_adjust = QPushButton("Apply")
        self.btn_apply_adjust.clicked.connect(self.commit_adjustments)
        self.btn_apply_adjust.setEnabled(False)
        
        self.btn_cancel_adjust = QPushButton("Cancel")
        self.btn_cancel_adjust.clicked.connect(self.cancel_adjustments)
        self.btn_cancel_adjust.setEnabled(False)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_apply_adjust)
        btn_layout.addWidget(self.btn_cancel_adjust)
        scroll_layout.addLayout(btn_layout)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        scroll_layout.addWidget(line)

        # --- Bagian Filters (QCheckBox) ---
        scroll_layout.addWidget(QLabel("<h2>Filters</h2>"))
        self.filter_checkboxes = []
        for algo_name in ONE_CLICK_FILTERS:
            cb = QCheckBox(algo_name)
            cb.toggled.connect(lambda checked, name=algo_name: self.on_one_click_toggled(checked, name))
            cb.setEnabled(False)
            scroll_layout.addWidget(cb)
            self.filter_checkboxes.append(cb)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_panel, 1)
        main_layout.addWidget(right_panel)


    def apply_stylesheet(self):
        # (Fungsi stylesheet tidak berubah dari v4.0)
        self.setStyleSheet("""
            QMainWindow { background-color: #2E2E2E; }
            QWidget { color: #FFFFFF; font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; }
            QLabel { font-size: 16px; }
            QLabel[setAlignment="132"] { color: #AAAAAA; font-size: 18px; font-style: italic; }
            h2 { font-size: 20px; font-weight: bold; color: #55AAFF;
                 border-bottom: 2px solid #444444; padding-bottom: 5px; }
            QPushButton { background-color: #555555; color: #FFFFFF; border: none;
                          padding: 12px 15px; border-radius: 5px; text-align: left; }
            QPushButton:hover { background-color: #666666; }
            QPushButton:pressed { background-color: #777777; }
            QPushButton:disabled { background-color: #444444; color: #888888; }
            QPushButton#btn_apply_adjust { background-color: #2D8C2D; text-align: center; }
            QPushButton#btn_apply_adjust:hover { background-color: #3AA83A; }
            QPushButton#btn_cancel_adjust { background-color: #B82C2C; text-align: center; }
            QPushButton#btn_cancel_adjust:hover { background-color: #D33636; }
            QScrollArea { border: none; }
            QScrollBar:vertical { background: #444444; width: 12px; margin: 0; }
            QScrollBar::handle:vertical { background: #666666; min-height: 20px; border-radius: 6px; }
            QSlider::groove:horizontal { border: 1px solid #444; background: #333; height: 8px; border-radius: 4px; }
            QSlider::handle:horizontal { background: #55AAFF; border: 1px solid #55AAFF;
                                         width: 18px; margin: -5px 0; border-radius: 9px; }
            QSlider::sub-page:horizontal { background: #55AAFF; }
            QFrame[frameShape="4"] { border-top: 1px solid #444444; }
            QCheckBox { spacing: 10px; padding: 5px 0; }
            QCheckBox::indicator { width: 18px; height: 18px; }
            QCheckBox::indicator:unchecked { border: 1px solid #888; background-color: #444; border-radius: 4px; }
            QCheckBox::indicator:checked { background-color: #55AAFF; border: 1px solid #55AAFF; border-radius: 4px; }
            QCheckBox:disabled { color: #888888; }
        """)
        self.btn_apply_adjust.setObjectName("btn_apply_adjust")
        self.btn_cancel_adjust.setObjectName("btn_cancel_adjust")

    # --- Fungsi Konversi (Tidak Berubah) ---

    def pil_to_cv(self, pil_img):
        pil_rgb = pil_img.convert('RGB')
        cv_bgr = np.array(pil_rgb)
        return cv_bgr[:, :, ::-1]

    def cv_to_pil(self, cv_img):
        if len(cv_img.shape) == 2:
            return Image.fromarray(cv_img, 'L')
        else:
            cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(cv_rgb)

    def pil_to_pixmap(self, pil_img):
        try:
            if pil_img.mode == "RGBA":
                data = pil_img.tobytes("raw", "RGBA")
                qimage = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
            elif pil_img.mode == "L":
                data = pil_img.tobytes("raw", "L")
                qimage = QImage(data, pil_img.width, pil_img.height, QImage.Format_Grayscale8)
            else:
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                data = pil_img.tobytes("raw", "RGB")
                qimage = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGB888)
            return QPixmap.fromImage(qimage)
        except Exception as e:
            print(f"Error konversi PIL ke QPixmap: {e}")
            return QPixmap()

    # --- Fungsi Inti Aplikasi (Tidak Berubah) ---

    def set_all_controls_enabled(self, enabled):
        self.btn_save.setEnabled(enabled)
        self.btn_reset.setEnabled(enabled)
        self.btn_apply_adjust.setEnabled(enabled)
        self.btn_cancel_adjust.setEnabled(enabled)
        for slider in self.sliders.values():
            slider.setEnabled(enabled)
        for cb in self.filter_checkboxes:
            cb.setEnabled(enabled)

    def upload_image(self):
        if self.camera_timer.isActive():
            self.stop_camera()
        fname, _ = QFileDialog.getOpenFileName(self, 'Buka Gambar', '.', 'File Gambar (*.jpg *.jpeg *.png *.bmp)')
        if fname:
            try:
                self.active_image = Image.open(fname)
                self.original_image = self.active_image.copy()
                self.reset_filters_and_commit()
                self.update_preview()
                self.set_all_controls_enabled(True)
            except Exception as e:
                self.image_label.setText(f"Gagal membuka gambar: {e}")

    def save_image(self):
        if self.preview_image:
            fname, _ = QFileDialog.getSaveFileName(self, 'Simpan Gambar', '.', 'PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)')
            if fname:
                try:
                    save_img = self.preview_image.convert('RGB')
                    save_img.save(fname)
                except Exception as e:
                    print(f"Error saat menyimpan: {e}")

    def reset_image(self):
        if self.original_image:
            self.active_image = self.original_image.copy()
            self.reset_filters_and_commit()
            self.update_preview()

    def display_image(self, img_to_display):
        if img_to_display:
            self.current_pixmap = self.pil_to_pixmap(img_to_display)
            self.update_image_display()
            self.update_histogram(img_to_display) # DIPERBARUI
        else:
            self.image_label.setText("Tidak ada gambar.")
            self.histogram_label.clear()
            
    def update_image_display(self):
        if self.current_pixmap:
            scaled_pixmap = self.current_pixmap.scaled(self.image_label.size(),
                                                       Qt.KeepAspectRatio,
                                                       Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_image_display()
        super().resizeEvent(event)
        
    def closeEvent(self, event):
        self.stop_camera()
        super().closeEvent(event)

    # --- Fungsi Kamera (Tidak Berubah) ---

    def start_camera(self):
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened(): raise Exception("Tidak dapat membuka webcam.")
            self.active_image = None
            self.original_image = None
            self.camera_timer.start(30)
            self.btn_start_cam.setEnabled(False)
            self.btn_upload.setEnabled(False)
            self.btn_stop_cam.setEnabled(True)
            self.btn_snapshot.setEnabled(True)
            self.set_all_controls_enabled(True)
        except Exception as e:
            self.image_label.setText(f"Error Kamera: {e}")
            if self.camera: self.camera.release()

    def stop_camera(self):
        if self.camera_timer.isActive():
            self.camera_timer.stop()
            self.camera.release()
            self.camera = None
        self.btn_start_cam.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)
        self.btn_snapshot.setEnabled(False)
        if not self.active_image:
            self.set_all_controls_enabled(False)
            self.image_label.setText("Kamera berhenti. Upload gambar atau mulai lagi.")
            self.histogram_label.clear()

    def update_camera_frame(self):
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                base_pil_frame = self.cv_to_pil(frame)
                slider_vals = self.active_slider_values
                filter_set = self.active_one_click_filters
                self.preview_image = self.apply_filter_pipeline(base_pil_frame, slider_vals, filter_set)
                self.display_image(self.preview_image)
            else:
                self.stop_camera()
                self.image_label.setText("Error membaca frame.")

    def take_snapshot(self):
        if self.preview_image:
            self.active_image = self.preview_image.copy()
            self.original_image = self.active_image.copy()
            self.stop_camera()
            self.reset_filters_and_commit()
            self.update_preview()
            
    # --- BARU: Fungsi Histogram (DIPERBARUI) ---
    
    def update_histogram(self, pil_img):
        try:
            # BARU: Dapatkan info pixel
            width, height = pil_img.size
            total_pixels = width * height
            
            # Konversi ke array numpy
            img = np.array(pil_img.convert('RGB'))
            
            plt.figure(figsize=(4, 2), dpi=100)
            plt.gca().set_facecolor('#2E2E2E')
            
            colors = ('r', 'g', 'b')
            is_grayscale = (len(img.shape) == 2) or (img.shape[2] == 1)

            if is_grayscale:
                 hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                 plt.plot(hist, color='white', linewidth=1)
            else:
                for i, col in enumerate(colors):
                    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                    plt.plot(hist, color=col, linewidth=1)
                
            plt.xlim([0, 256])
            plt.yticks([])
            plt.xticks(color='white')
            # DIPERBARUI: Judul dengan info pixel
            plt.title(f"Histogram (Total Pixels: {total_pixels:,})", color='white', fontsize=10)
            plt.gca().tick_params(axis='x', colors='white')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_color('white')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#2E2E2E')
            buf.seek(0)
            
            qimage = QImage()
            qimage.loadFromData(buf.getvalue(), 'PNG')
            pixmap = QPixmap.fromImage(qimage)
            
            self.histogram_label.setPixmap(pixmap)
            plt.close()

        except Exception as e:
            print(f"Error update histogram: {e}")
            self.histogram_label.setText("Error histogram")


    # --- Mesin Pengolah Citra (DIPERBARUI) ---

    def on_slider_changed(self):
        """Callback saat slider digerakkan."""
        self.active_slider_values = {name: s.value() for name, s in self.sliders.items()}
        if not self.camera_timer.isActive():
            self.update_preview()

    def on_one_click_toggled(self, checked, name):
        """Callback saat checkbox di-klik."""
        if checked:
            self.active_one_click_filters.add(name)
        else:
            self.active_one_click_filters.discard(name)
        
        if not self.camera_timer.isActive():
            self.update_preview()

    def update_preview(self):
        """
        Memperbarui preview berdasarkan 'active_image' dan state filter TREN.
        """
        if not self.active_image:
            return
            
        slider_vals = self.active_slider_values
        filter_set = self.active_one_click_filters
        
        self.preview_image = self.apply_filter_pipeline(self.active_image, slider_vals, filter_set)
        self.display_image(self.preview_image)

    def apply_filter_pipeline(self, base_image, slider_vals, filter_set):
        """
        Fungsi 'murni' yang mengambil gambar dasar dan state filter,
        lalu mengembalikan gambar yang sudah diproses.
        (DIPERBARUI dengan filter baru)
        """
        img = base_image.copy()
        
        try:
            # --- Terapkan Filter Geometri (PIL) ---
            if "Putar 90° Kanan" in filter_set: img = img.rotate(-90, expand=True)
            if "Putar 90° Kiri" in filter_set: img = img.rotate(90, expand=True)
            if "Flip Horizontal" in filter_set: img = ImageOps.mirror(img)
            if "Flip Vertical" in filter_set: img = ImageOps.flip(img)
            
            # --- Terapkan Filter Point/Artistik (PIL) ---
            if "Grayscale" in filter_set: img = ImageOps.grayscale(img)
            if "Negative" in filter_set: img = ImageOps.invert(img.convert('RGB'))
            if "Sepia" in filter_set: img = self.apply_sepia(img)
            if "Posterize" in filter_set: img = ImageOps.posterize(img.convert('RGB'), 4)
            if "Solarize" in filter_set: img = ImageOps.solarize(img.convert('RGB'), 128)
            if "Emboss" in filter_set: img = img.filter(ImageFilter.EMBOSS)
            if "Edge Enhance" in filter_set: img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
            if "Contour" in filter_set: img = img.filter(ImageFilter.CONTOUR)

            # --- Terapkan Filter OpenCV (membutuhkan konversi) ---
            cv_img = self.pil_to_cv(img)
            kernel = np.ones((5, 5), np.uint8) # Kernel umum untuk morfologi

            # Blur
            if "Gaussian Blur" in filter_set: cv_img = cv2.GaussianBlur(cv_img, (15, 15), 0)
            if "Median Blur" in filter_set: cv_img = cv2.medianBlur(cv_img, 15)
            if "Bilateral Filter" in filter_set: cv_img = cv2.bilateralFilter(cv_img, 15, 75, 75)
            
            # BARU: Stylization
            if "Stylization (Art)" in filter_set: cv_img = cv2.stylization(cv_img, sigma_s=60, sigma_r=0.6)
            if "Pencil Sketch" in filter_set:
                _ , cv_img = cv2.pencilSketch(cv_img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
                # pencilSketch mengembalikan (dst_gray, dst_color), kita ambil yg warna

            # Morfologi (Basic & Advanced)
            if "Erode" in filter_set: cv_img = cv2.erode(cv_img, kernel, iterations=1)
            if "Dilate" in filter_set: cv_img = cv2.dilate(cv_img, kernel, iterations=1)
            if "Opening (Noise Removal)" in filter_set: cv_img = cv2.morphologyEx(cv_img, cv2.MORPH_OPEN, kernel)
            if "Closing (Fill Holes)" in filter_set: cv_img = cv2.morphologyEx(cv_img, cv2.MORPH_CLOSE, kernel)
            if "Morphological Gradient" in filter_set: cv_img = cv2.morphologyEx(cv_img, cv2.MORPH_GRADIENT, kernel)

            # --- Filter yang menghasilkan Grayscale ---
            # (Thresholding, Edge Detection, Color Channel Views)
            # Buat gambar grayscale 'dasar' jika diperlukan
            gray_img = None
            if any(f in filter_set for f in ["Binary Threshold", "Adaptive Threshold", "Canny Edges", 
                                            "Sobel X (Vertical)", "Sobel Y (Horizontal)", "Laplacian Edges"]):
                gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) > 2 else cv_img

            if "Binary Threshold" in filter_set:
                _, cv_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
            elif "Adaptive Threshold" in filter_set:
                cv_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            elif "Canny Edges" in filter_set:
                cv_img = cv2.Canny(gray_img, 100, 200)
            elif "Sobel X (Vertical)" in filter_set:
                cv_img = cv2.convertScaleAbs(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5))
            elif "Sobel Y (Horizontal)" in filter_set:
                cv_img = cv2.convertScaleAbs(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5))
            elif "Laplacian Edges" in filter_set:
                cv_img = cv2.convertScaleAbs(cv2.Laplacian(gray_img, cv2.CV_64F))
            
            # BARU: Color Space Views
            elif any(f in filter_set for f in ["View Hue Channel", "View Saturation Channel", "View Value Channel"]):
                hsv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv_img)
                if "View Hue Channel" in filter_set: cv_img = h
                elif "View Saturation Channel" in filter_set: cv_img = s
                elif "View Value Channel" in filter_set: cv_img = v
            
            # Konversi kembali ke PIL
            img = self.cv_to_pil(cv_img)
            
            # --- Terapkan Sliders (ImageEnhance) Terakhir ---
            brightness = slider_vals.get("Brightness", 100) / 100.0
            contrast = slider_vals.get("Contrast", 100) / 100.0
            sharpness = slider_vals.get("Sharpen", 100) / 100.0
            saturation = slider_vals.get("Saturation", 100) / 100.0 # BARU

            if brightness != 1.0: img = ImageEnhance.Brightness(img).enhance(brightness)
            if contrast != 1.0: img = ImageEnhance.Contrast(img).enhance(contrast)
            if sharpness != 1.0: img = ImageEnhance.Sharpness(img).enhance(sharpness)
            if saturation != 1.0: img = ImageEnhance.Color(img).enhance(saturation) # BARU
            
            return img

        except Exception as e:
            print(f"Error di pipeline filter: {e}")
            return base_image # Kembalikan gambar asli jika error

    def commit_adjustments(self):
        """Tombol 'Apply': Menyimpan state filter TREN sebagai state 'committed'."""
        self.committed_slider_values = self.active_slider_values.copy()
        self.committed_one_click_filters = self.active_one_click_filters.copy()

    def cancel_adjustments(self):
        """Tombol 'Cancel': Mengembalikan UI filter ke state 'committed' terakhir."""
        # 1. Kembalikan nilai slider
        for name, s in self.sliders.items():
            default = ADJUSTABLE_ALGORITHMS[name]["default"]
            s.setValue(self.committed_slider_values.get(name, default))
        
        # 2. Kembalikan state checkbox
        # Matikan sinyal sementara agar tidak memicu update_preview berulang kali
        for cb in self.filter_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(cb.text() in self.committed_one_click_filters)
            cb.blockSignals(False)
        
        # PENTING: Perbarui state aktif secara manual setelah reset
        self.active_slider_values = {name: s.value() for name, s in self.sliders.items()}
        self.active_one_click_filters = self.committed_one_click_filters.copy()

        # Panggil update_preview sekali saja
        if not self.camera_timer.isActive():
            self.update_preview()

    def reset_filters_and_commit(self):
        """Helper untuk mereset filter ke default dan langsung 'Apply' state itu."""
        self.committed_slider_values = {}
        for name, props in ADJUSTABLE_ALGORITHMS.items():
             self.committed_slider_values[name] = props["default"]
        
        self.committed_one_click_filters = set()
        
        self.cancel_adjustments()
        self.commit_adjustments()

    def apply_sepia(self, img):
        # (Fungsi Sepia tidak berubah)
        img = img.convert('RGB')
        width, height = img.size
        pixels = img.load()
        for py in range(height):
            for px in range(width):
                r, g, b = img.getpixel((px, py))
                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                pixels[px, py] = (min(tr, 255), min(tg, 255), min(tb, 255))
        return img


# --- Main execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())