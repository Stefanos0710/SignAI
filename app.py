import sys
import cv2
import mediapipe as mp
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QTextEdit,
                               QStackedWidget, QFrame, QScrollArea, QCheckBox)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QFont
import numpy as np
import preprocessing_live_data
import inference
import json


class SidebarButton(QPushButton):
    def __init__(self, text, icon_unicode, parent=None):
        super().__init__(parent)
        self.setText(f"{icon_unicode}  {text}")
        self.setCheckable(True)
        self.setFixedHeight(50)
        self.setFont(QFont("Segoe UI", 11))
        self.setStyleSheet("""
            QPushButton {
                border: none;
                border-radius: 5px;
                text-align: left;
                padding: 10px;
                margin: 5px;
                color: #E0E0E0;
                background-color: rgba(255, 255, 255, 0.1);
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
            }
            QPushButton:checked {
                background-color: #2196F3;
                color: white;
            }
        """)


class Sidebar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 20, 10, 20)
        layout.setSpacing(5)

        # Logo area
        logo_label = QLabel("Sign Language\nTranslator")
        logo_label.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
                padding: 20px;
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
            }
        """)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        layout.addSpacing(20)

        # Navigation buttons
        self.translator_btn = SidebarButton("Translator", "🎥")
        self.dictionary_btn = SidebarButton("Dictionary", "📚")
        self.contribute_btn = SidebarButton("Contribute", "🤝")
        self.settings_btn = SidebarButton("Settings", "⚙️")
        self.help_btn = SidebarButton("Help", "❔")

        layout.addWidget(self.translator_btn)
        layout.addWidget(self.dictionary_btn)
        layout.addWidget(self.contribute_btn)
        layout.addSpacing(20)
        layout.addWidget(self.settings_btn)
        layout.addWidget(self.help_btn)

        layout.addStretch()

        self.setStyleSheet("""
            Sidebar {
                background-color: #1a1a2e;
                min-width: 250px;
                max-width: 250px;
            }
        """)


class ActionButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)


class TranslatorPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_gesture = ""
        # Kamera- und MediaPipe-Initialisierung direkt hier
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_camera_running = False
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Title
        title = QLabel("Sign Language Translator")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        layout.addWidget(title)

        # Camera view
        camera_frame = QFrame()
        camera_frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        camera_layout = QVBoxLayout(camera_frame)

        # Camera label
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                border-radius: 5px;
            }
        """)
        camera_layout.addWidget(self.camera_label)

        # Countdown overlay label (hidden by default)
        self.countdown_overlay = QLabel("")
        self.countdown_overlay.setAlignment(Qt.AlignCenter)
        self.countdown_overlay.setStyleSheet("font-size: 96px; color: #2196F3; background: rgba(255,255,255,0.3); border-radius: 20px;")
        self.countdown_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.countdown_overlay.setVisible(False)
        camera_layout.addWidget(self.countdown_overlay, alignment=Qt.AlignCenter)

        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        self.record_translate_btn = ActionButton("Start Recording")
        self.record_translate_btn.clicked.connect(self.handle_record_translate)

        controls_layout.addWidget(self.record_translate_btn)
        controls_layout.addStretch()

        # Translation output
        translation_frame = QFrame()
        translation_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        translation_layout = QVBoxLayout(translation_frame)

        translation_label = QLabel("Translation")
        translation_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")

        self.translation_output = QTextEdit()
        self.translation_output.setReadOnly(True)
        self.translation_output.setPlaceholderText("Translation will appear here...")
        self.translation_output.setMinimumHeight(100)
        self.translation_output.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 10px;
                font-size: 16px;
                color: #333;
            }
        """)

        translation_layout.addWidget(translation_label)
        translation_layout.addWidget(self.translation_output)

        layout.addWidget(camera_frame)
        layout.addLayout(controls_layout)
        layout.addWidget(translation_frame)

        # Aufnahme-Status
        self.is_recording = False
        self.is_countdown = False
        self.video_writer = None
        self.recorded_video_path = "data/live/video/recorded_video.mp4"
        self.countdown_timer = QTimer()
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_value = 3

        # Für stylische Übersetzungsanimation
        self.translation_progress = 0
        self.translation_progress_timer = QTimer()
        self.translation_progress_timer.setInterval(50)
        self.translation_progress_timer.timeout.connect(self.update_translation_progress)

        # Kamera direkt starten
        self.capture = None
        self.is_camera_running = False
        self.start_camera()

    def handle_record_translate(self):
        if not self.is_camera_running:
            self.translation_output.setText("Please start the camera before recording a video.")
            return
        if not self.is_recording and not self.is_countdown:
            # Countdown starten
            self.is_countdown = True
            self.countdown_value = 3
            self.record_translate_btn.setEnabled(False)
            self.translation_output.clear()
            self.translation_output.setText(f"Recording starts in {self.countdown_value}...")
            self.countdown_overlay.setText(str(self.countdown_value))
            self.countdown_overlay.setVisible(True)
            self.countdown_timer.start()
        elif self.is_recording:
            # Aufnahme stoppen und übersetzen
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.record_translate_btn.setEnabled(False)
            self.translation_output.clear()
            self.show_stylish_loading_animation(0)
            QApplication.processEvents()
            # Starte Preprocessing mit echtem Fortschritt
            def progress_callback(percent):
                self.show_stylish_loading_animation(percent)
            try:
                import preprocessing_live_data
                debug_mode = False
                mw = self.parent()
                while mw and not hasattr(mw, 'get_debug_mode'):
                    mw = mw.parent() if hasattr(mw, 'parent') else None
                if mw and hasattr(mw, 'get_debug_mode'):
                    debug_mode = mw.get_debug_mode()
                preprocessing_live_data.main(self.recorded_video_path, progress_callback=progress_callback, show_windows=debug_mode)
                self.translation_output.setText("Running AI translation...")
                QApplication.processEvents()
                result = inference.main_inference("models/trained_model_v21.keras")
                if not result:
                    result = "No translation found."
                self.translation_output.setText(result)
            except Exception as e:
                self.translation_output.setText(f"Error: {str(e)}")
            self.record_translate_btn.setText("Start Recording")
            self.record_translate_btn.setEnabled(True)

    def update_countdown(self):
        self.countdown_value -= 1
        if self.countdown_value > 0:
            self.translation_output.setText(f"Recording starts in {self.countdown_value}...")
            self.countdown_overlay.setText(str(self.countdown_value))
            self.countdown_overlay.setVisible(True)
        else:
            self.countdown_timer.stop()
            self.is_countdown = False
            self.countdown_overlay.setVisible(False)
            # Aufnahme starten
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 20.0
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_writer = cv2.VideoWriter(self.recorded_video_path, fourcc, fps, (width, height))
            self.is_recording = True
            self.record_translate_btn.setText("Translate Now")
            self.record_translate_btn.setEnabled(True)
            self.translation_output.setText("Recording... Press 'Translate Now' to save and translate.")

    def update_frame(self):
        if self.capture is not None and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                # Wenn Aufnahme läuft, Frame speichern
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)
                # Konvertiere das Bild von BGR (OpenCV) zu RGB (Qt)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.camera_label.setPixmap(pixmap)
            else:
                self.camera_label.setText("No frame")
        else:
            self.camera_label.setText("Camera not available")

    def start_camera(self):
        if self.capture is None or not (self.capture and self.capture.isOpened()):
            self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened():
            self.is_camera_running = True
            self.timer.start(30)
        else:
            self.show_error("Could not access the camera.")

    def stop_camera(self):
        # Kamera bleibt immer an, daher leer
        pass

    def handle_clear_translation(self):
        """Handler für Clear Translation Button"""
        self.clear_translation()

    def clear_translation(self):
        """Setzt das Übersetzungsfeld zurück."""
        self.translation_output.clear()

    def handle_toggle_camera(self):
        # Kamera wird beim Start automatisch gestartet, daher nicht mehr benötigt
        pass

    def record_video(self, filename, duration=3):
        if self.capture is None or not self.capture.isOpened():
            self.translation_output.setText("Bitte starte die Kamera, bevor du ein Video aufnimmst.")
            return False
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        num_frames = int(fps * duration)
        for _ in range(num_frames):
            ret, frame = self.capture.read()
            if not ret:
                break
            out.write(frame)
            cv2.waitKey(int(1000 / fps))
        out.release()
        return True

    def translate_video(self):
        try:
            # Loading-Animation anzeigen
            self.show_loading_animation()
            QApplication.processEvents()
            video_path = self.recorded_video_path
            preprocessing_live_data.main(video_path)
            self.translation_output.setText("Running AI translation...")
            QApplication.processEvents()
            result = inference.main_inference("models/trained_model_v21.keras")
            if not result:
                result = "No translation found."
            self.translation_output.setText(result)
        except Exception as e:
            self.translation_output.setText(f"Error: {str(e)}")

    def recognize_gesture(self, hand_landmarks):
        # Hier könnte eine Geste erkannt werden, aber wir implementieren die echte Übersetzung:
        pass

    def show_error(self, message):
        print(f"Error: {message}")

    def show_loading_animation(self):
        # CSS-animierter Spinner
        spinner_html = '''
        <style>
        .spinner {
          margin: 30px auto;
          width: 60px;
          height: 60px;
          border: 8px solid #e0e0e0;
          border-top: 8px solid #2196F3;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        </style>
        <div style="text-align:center;">
          <div class="spinner"></div>
          <div style="font-size:18px; margin-top:20px;">Übersetzung läuft...</div>
        </div>
        '''
        self.translation_output.setHtml(spinner_html)
        QApplication.processEvents()

    def show_stylish_loading_animation(self, percent):
        percent = int(percent)
        spinner_html = f'''
        <style>
        .orbit-loader {{
          position: relative;
          width: 120px;
          height: 120px;
          margin: 40px auto 10px auto;
        }}
        .orbit-loader .orbit {{
          position: absolute;
          border-radius: 50%;
          border: 4px solid transparent;
          border-top: 4px solid #21CBF3;
          border-right: 4px solid #2196F3;
          border-bottom: 4px solid #1DE9B6;
          border-left: 4px solid #fff;
          animation: orbit-spin 1.2s linear infinite;
          box-shadow: 0 0 16px #21CBF3, 0 0 32px #2196F3 inset;
        }}
        .orbit-loader .orbit.orbit2 {{
          width: 80px;
          height: 80px;
          top: 20px;
          left: 20px;
          border-width: 3px;
          border-top: 3px solid #1DE9B6;
          border-right: 3px solid #21CBF3;
          border-bottom: 3px solid #2196F3;
          border-left: 3px solid #fff;
          animation-duration: 1.7s;
          animation-direction: reverse;
          opacity: 0.7;
        }}
        .orbit-loader .orbit.orbit3 {{
          width: 50px;
          height: 50px;
          top: 35px;
          left: 35px;
          border-width: 2px;
          border-top: 2px solid #fff;
          border-right: 2px solid #1DE9B6;
          border-bottom: 2px solid #21CBF3;
          border-left: 2px solid #2196F3;
          animation-duration: 2.2s;
          opacity: 0.5;
        }}
        @keyframes orbit-spin {{
          0% {{ transform: rotate(0deg) scale(1); }}
          50% {{ transform: rotate(180deg) scale(1.08); }}
          100% {{ transform: rotate(360deg) scale(1); }}
        }}
        .progress-bar {{
          width: 80%;
          height: 20px;
          background: #e0e0e0;
          border-radius: 10px;
          margin: 30px auto 0 auto;
          overflow: hidden;
          box-shadow: 0 0 12px #21CBF3, 0 0 8px #1DE9B6 inset;
        }}
        .progress {{
          height: 100%;
          background: linear-gradient(90deg, #2196F3, #21CBF3, #1DE9B6);
          border-radius: 10px;
          width: {percent}%;
          transition: width 0.2s;
          box-shadow: 0 0 16px #21CBF3;
        }}
        .progress-text {{
          font-size: 2.2em;
          font-weight: bold;
          margin-top: 18px;
          background: linear-gradient(90deg, #2196F3, #21CBF3, #1DE9B6);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          text-fill-color: transparent;
          text-shadow: 0 2px 8px #21CBF3;
        }}
        .translating-label {{
          font-size: 1.2em;
          color: #2196F3;
          margin-bottom: 10px;
          letter-spacing: 2px;
        }}
        </style>
        <div style="text-align:center;">
          <div class="translating-label">Translating your sign...</div>
          <div class="orbit-loader">
            <div class="orbit" style="width:120px;height:120px;"></div>
            <div class="orbit orbit2"></div>
            <div class="orbit orbit3"></div>
          </div>
          <div class="progress-bar">
            <div class="progress"></div>
          </div>
          <div class="progress-text">{percent}%</div>
        </div>
        '''
        self.translation_output.setHtml(spinner_html)
        QApplication.processEvents()

    def update_translation_progress(self):
        self.translation_progress += 2
        if self.translation_progress > 100:
            self.translation_progress = 100
        # Setze den Fortschritt im HTML (über JS, falls unterstützt, sonst Text ersetzen)
        # PySide6 QTextEdit unterstützt kein echtes JS, daher Text ersetzen:
        html = self.translation_output.toHtml()
        html = html.replace(r'Translating... \d+%', f'Translating... {self.translation_progress}%')
        html = html.replace('width: 0%;', f'width: {self.translation_progress}%;')
        self.translation_output.setHtml(html)
        QApplication.processEvents()
        if self.translation_progress >= 100:
            self.translation_progress_timer.stop()
            # Jetzt wirklich übersetzen
            try:
                video_path = self.recorded_video_path
                preprocessing_live_data.main(video_path)
                self.translation_output.setText("Running AI translation...")
                QApplication.processEvents()
                result = inference.main_inference("models/trained_model_v21.keras")
                if not result:
                    result = "No translation found."
                self.translation_output.setText(result)
            except Exception as e:
                self.translation_output.setText(f"Error: {str(e)}")
            self.record_translate_btn.setText("Start Recording")
            self.record_translate_btn.setEnabled(True)


class  DictionaryPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Title
        title = QLabel("Sign Language Dictionary")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        layout.addWidget(title)

        # Search bar
        search_frame = QFrame()
        search_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
            }
        """)
        search_layout = QHBoxLayout(search_frame)

        self.search_input = QTextEdit()
        self.search_input.setMaximumHeight(40)
        self.search_input.setPlaceholderText("Search signs...")
        self.search_input.setStyleSheet("""
            QTextEdit {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
                color: #333;
            }
        """)

        self.search_btn = ActionButton("Search")
        self.search_btn.clicked.connect(self.handle_search_signs)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_btn)

        # Dictionary content
        content_frame = QFrame()
        content_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
            }
        """)

        self.content_area = QScrollArea()
        self.content_area.setWidgetResizable(True)
        self.content_area.setStyleSheet("""
            QScrollArea {
                border: none;
            }
        """)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)

        # Sample dictionary entries
        for i in range(10):
            entry = QFrame()
            entry.setStyleSheet("""
                QFrame {
                    background-color: #f8f9fa;
                    border: 1px solid #e0e0e0;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 5px;
                }
                QFrame:hover {
                    background-color: #e9ecef;
                }
            """)
            entry_layout = QHBoxLayout(entry)

            # Sign symbol
            sign_label = QLabel(f"🤚")
            sign_label.setStyleSheet("font-size: 32px;")

            # Sign details
            details_layout = QVBoxLayout()
            sign_name = QLabel(f"Sign {i + 1}")
            sign_name.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
            sign_desc = QLabel("Description of the sign and its usage in everyday communication.")
            sign_desc.setStyleSheet("color: #666; font-size: 14px;")
            sign_desc.setWordWrap(True)

            details_layout.addWidget(sign_name)
            details_layout.addWidget(sign_desc)

            entry_layout.addWidget(sign_label)
            entry_layout.addLayout(details_layout)
            entry_layout.setStretch(1, 1)

            self.content_layout.addWidget(entry)

        self.content_layout.addStretch()
        self.content_area.setWidget(self.content_widget)

        main_content_layout = QVBoxLayout(content_frame)
        main_content_layout.addWidget(self.content_area)

        layout.addWidget(search_frame)
        layout.addWidget(content_frame)
        # Load signs from tokenizer
        self.signs = self.load_signs_from_tokenizer()
        self.display_signs(self.signs)

    def load_signs_from_tokenizer(self):
        try:
            with open("tokenizers/gloss_tokenizer.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, str):
                data = json.loads(data)
            config = data.get("config", data)
            def is_real_word(word):
                return not (word.startswith('<') and word.endswith('>'))
            if "index_word" in config:
                index_word = config["index_word"]
                if isinstance(index_word, str):
                    index_word = json.loads(index_word)
                return sorted(set(w for w in index_word.values() if is_real_word(w)))
            elif "word_index" in config:
                word_index = config["word_index"]
                if isinstance(word_index, str):
                    word_index = json.loads(word_index)
                return sorted(set(w for w in word_index.keys() if is_real_word(w)))
            else:
                return sorted(set(str(v) for v in config.values() if is_real_word(str(v))))
        except Exception as e:
            print(f"Could not load signs: {e}")
            return []

    def display_signs(self, signs):
        # Clear layout
        for i in reversed(range(self.content_layout.count())):
            widget = self.content_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        # Add entries
        for sign in signs:
            entry = QFrame()
            entry.setStyleSheet("""
                QFrame {
                    background-color: #f8f9fa;
                    border: 1px solid #e0e0e0;
                    border-radius: 5px;
                    padding: 10px;
                    margin: 5px;
                }
                QFrame:hover {
                    background-color: #e9ecef;
                }
            """)
            entry_layout = QHBoxLayout(entry)
            sign_label = QLabel(f"🤚")
            sign_label.setStyleSheet("font-size: 32px;")
            details_layout = QVBoxLayout()
            sign_name = QLabel(sign)
            sign_name.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
            sign_desc = QLabel("No description available.")
            sign_desc.setStyleSheet("color: #666; font-size: 14px;")
            sign_desc.setWordWrap(True)
            details_layout.addWidget(sign_name)
            details_layout.addWidget(sign_desc)
            entry_layout.addWidget(sign_label)
            entry_layout.addLayout(details_layout)
            entry_layout.setStretch(1, 1)
            self.content_layout.addWidget(entry)
        self.content_layout.addStretch()

    def handle_search_signs(self):
        query = self.search_input.toPlainText().strip().lower()
        if not query:
            filtered = self.signs
        else:
            filtered = [s for s in self.signs if query in s.lower()]
        self.display_signs(filtered)


class ContributeDashboardPage(QWidget):
    def __init__(self, parent=None, switch_page_callback=None):
        super().__init__(parent)
        self.switch_page_callback = switch_page_callback
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        # Title
        title = QLabel("Contribute Dashboard")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        layout.addWidget(title)
        # Stats area
        stats_layout = QHBoxLayout()
        self.stats_boxes = []
        stats = [
            ("Uploaded", 12, "#2196F3"),
            ("Accepted", 7, "#43A047"),
            ("Pending", 3, "#FFC107"),
            ("Rejected", 2, "#E53935")
        ]
        for label, value, color in stats:
            box = QFrame()
            box.setStyleSheet(f"background:{color};border-radius:10px;padding:20px;")
            v = QVBoxLayout(box)
            l1 = QLabel(str(value))
            l1.setStyleSheet("font-size:32px;font-weight:bold;color:white;")
            l2 = QLabel(label)
            l2.setStyleSheet("font-size:16px;color:white;")
            v.addWidget(l1)
            v.addWidget(l2)
            stats_layout.addWidget(box)
            self.stats_boxes.append(box)
        layout.addLayout(stats_layout)
        # Main buttons
        btn_layout = QHBoxLayout()
        record_btn = ActionButton("Record a New Sign")
        record_btn.clicked.connect(lambda: self.switch_page_callback(1) if self.switch_page_callback else None)
        uploads_btn = ActionButton("My Uploads")
        uploads_btn.clicked.connect(lambda: self.switch_page_callback(2) if self.switch_page_callback else None)
        terms_btn = ActionButton("Read Terms & Rules")
        terms_btn.clicked.connect(lambda: self.switch_page_callback(3) if self.switch_page_callback else None)
        btn_layout.addWidget(record_btn)
        btn_layout.addWidget(uploads_btn)
        btn_layout.addWidget(terms_btn)
        layout.addLayout(btn_layout)
        layout.addStretch()

class ContributeRecordPage(QWidget):
    def __init__(self, parent=None, word="Hello", on_change_word=None):
        super().__init__(parent)
        self.word = word
        self.on_change_word = on_change_word
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        # Word display
        word_layout = QHBoxLayout()
        word_label = QLabel(self.word)
        word_label.setStyleSheet("font-size:36px;font-weight:bold;color:#2196F3;")
        word_layout.addWidget(word_label, alignment=Qt.AlignCenter)
        change_btn = ActionButton("↻ Change Word")
        if self.on_change_word:
            change_btn.clicked.connect(self.on_change_word)
        word_layout.addWidget(change_btn, alignment=Qt.AlignRight)
        layout.addLayout(word_layout)
        # Camera preview placeholder
        camera_label = QLabel("[Camera Preview Here]")
        camera_label.setAlignment(Qt.AlignCenter)
        camera_label.setMinimumSize(480, 320)
        camera_label.setStyleSheet("background:#f5f5f5;border-radius:10px;")
        layout.addWidget(camera_label)
        # Countdown placeholder
        countdown_label = QLabel("")
        countdown_label.setAlignment(Qt.AlignCenter)
        countdown_label.setStyleSheet("font-size:48px;color:#2196F3;")
        layout.addWidget(countdown_label)
        # After recording: preview and buttons
        preview_label = QLabel("[Video Preview Here]")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setVisible(False)
        layout.addWidget(preview_label)
        btns = QHBoxLayout()
        self.upload_btn = ActionButton("Upload Video")
        self.upload_btn.setEnabled(False)
        again_btn = ActionButton("Record Again")
        btns.addWidget(self.upload_btn)
        btns.addWidget(again_btn)
        layout.addLayout(btns)
        layout.addStretch()

class ContributeUploadsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        title = QLabel("My Uploads")
        title.setStyleSheet("font-size:24px;font-weight:bold;color:#333;")
        layout.addWidget(title)
        # Table header
        header = QHBoxLayout()
        for h in ["Word", "Date", "Status", "Preview", "Delete"]:
            l = QLabel(h)
            l.setStyleSheet("font-weight:bold;color:#2196F3;")
            header.addWidget(l)
        layout.addLayout(header)
        # Example uploads
        uploads = [
            ("Hello", "2024-06-01", "Accepted"),
            ("Thank you", "2024-06-02", "Pending"),
            ("Sorry", "2024-06-03", "Rejected")
        ]
        for word, date, status in uploads:
            row = QHBoxLayout()
            row.addWidget(QLabel(word))
            row.addWidget(QLabel(date))
            status_label = QLabel()
            if status == "Accepted":
                status_label.setText("✅ Accepted")
                status_label.setStyleSheet("color:#43A047;font-weight:bold;")
            elif status == "Pending":
                status_label.setText("🕐 Pending")
                status_label.setStyleSheet("color:#FFC107;font-weight:bold;")
            else:
                status_label.setText("❌ Rejected")
                status_label.setStyleSheet("color:#E53935;font-weight:bold;")
            row.addWidget(status_label)
            preview_btn = ActionButton("Preview")
            delete_btn = ActionButton("Delete")
            row.addWidget(preview_btn)
            row.addWidget(delete_btn)
            layout.addLayout(row)
        layout.addStretch()

class ContributeTermsPage(QWidget):
    def __init__(self, parent=None, on_agree=None):
        super().__init__(parent)
        self.on_agree = on_agree
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        title = QLabel("Terms & Rules")
        title.setStyleSheet("font-size:24px;font-weight:bold;color:#333;")
        layout.addWidget(title)
        # Sections
        sections = [
            ("Was darf ich hochladen?", "⚠️", "Nur eigene, klare Gebärdenvideos. Keine beleidigenden Inhalte."),
            ("Datenschutz", "🔒", "Deine Videos werden nur für Trainingszwecke verwendet."),
            ("Nutzungsrechte", "📜", "Mit dem Upload gibst du uns das Recht zur Nutzung für KI-Training.")
        ]
        for sec, icon, desc in sections:
            h = QHBoxLayout()
            h.addWidget(QLabel(icon))
            v = QVBoxLayout()
            l1 = QLabel(sec)
            l1.setStyleSheet("font-weight:bold;font-size:16px;")
            l2 = QLabel(desc)
            l2.setStyleSheet("color:#666;")
            v.addWidget(l1)
            v.addWidget(l2)
            h.addLayout(v)
            layout.addLayout(h)
        # Agree button
        self.agree_checkbox = QCheckBox("✔ I Agree to These Terms")
        if self.on_agree:
            self.agree_checkbox.stateChanged.connect(self.on_agree)
        layout.addWidget(self.agree_checkbox)
        layout.addStretch()

class ContributePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stacked = QStackedWidget()
        self.dashboard = ContributeDashboardPage(switch_page_callback=self.switch_page)
        self.record = ContributeRecordPage()
        self.uploads = ContributeUploadsPage()
        self.terms = ContributeTermsPage()
        self.stacked.addWidget(self.dashboard)
        self.stacked.addWidget(self.record)
        self.stacked.addWidget(self.uploads)
        self.stacked.addWidget(self.terms)
        # Modern sticky nav bar
        nav_bar = QFrame()
        nav_bar.setObjectName("contributeNavBar")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(24, 16, 24, 0)
        nav_layout.setSpacing(0)
        self.nav_buttons = []
        nav_items = [
            ("📊 Dashboard", 0),
            ("🎥 Record Sign", 1),
            ("📁 My Uploads", 2),
            ("⚖️ Terms & Rules", 3)
        ]
        for label, idx in nav_items:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(48)
            btn.setMinimumWidth(160)
            btn.setFont(QFont("Segoe UI", 13, QFont.Bold))
            btn.setStyleSheet(self._nav_btn_style(False))
            btn.clicked.connect(lambda checked, i=idx: self.switch_page(i))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)
        nav_layout.addStretch()
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(nav_bar)
        main_layout.addSpacing(8)
        main_layout.addWidget(self.stacked)
        self.setStyleSheet(self._nav_bar_qss())
        self.switch_page(0)
    def switch_page(self, idx):
        self.stacked.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == idx)
            btn.setStyleSheet(self._nav_btn_style(i == idx))
    def _nav_btn_style(self, active):
        if active:
            return (
                "background: white; color: #1976D2; font-weight: bold; "
                "border: none; border-radius: 16px 16px 0 0; "
                "box-shadow: 0 4px 16px rgba(33,150,243,0.10); "
                "padding: 12px 36px; margin-bottom: -2px; "
                "font-size: 18px; outline: none;"
            )
        else:
            return (
                "background: transparent; color: #2196F3; font-weight: normal; "
                "border: none; border-radius: 16px 16px 0 0; "
                "padding: 12px 36px; margin-bottom: 0; "
                "font-size: 17px; outline: none;"
                "transition: background 0.2s;"
            )
    def _nav_bar_qss(self):
        return (
            "#contributeNavBar { "
            "  background: #f8fbff; "
            "  border-radius: 18px 18px 0 0; "
            "  border-bottom: 2px solid #e3eaf3; "
            "  margin: 18px 18px 0 18px; "
            "  box-shadow: 0 6px 24px rgba(33,150,243,0.07); "
            "  min-height: 64px; "
            "  position: sticky; "
            "  top: 0; "
            "  z-index: 10; "
            "} "
            "QPushButton:hover { "
            "  background: #e3f2fd; "
            "  color: #1565C0; "
            "} "
            "QPushButton:pressed { "
            "  background: #bbdefb; "
            "  color: #0D47A1; "
            "} "
        )


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Title
        title = QLabel("Settings")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        layout.addWidget(title)

        # Debug Mode Toggle
        self.debug_checkbox = QCheckBox("Debug Mode (show hand/face windows)")
        self.debug_checkbox.setChecked(False)
        layout.addWidget(self.debug_checkbox)

        # Settings groups
        # Camera Settings
        camera_group = QFrame()
        camera_group.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        camera_layout = QVBoxLayout(camera_group)
        camera_title = QLabel("🎥 Camera Settings")
        camera_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        camera_desc = QLabel("Configure your camera settings")
        camera_desc.setStyleSheet("color: #666; margin-bottom: 15px;")

        camera_layout.addWidget(camera_title)
        camera_layout.addWidget(camera_desc)
        camera_layout.addWidget(ActionButton("Select Camera"))
        camera_layout.addWidget(ActionButton("Calibrate Camera"))

        # Translation Settings
        translation_group = QFrame()
        translation_group.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        translation_layout = QVBoxLayout(translation_group)
        translation_title = QLabel("🔤 Translation Settings")
        translation_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        translation_desc = QLabel("Customize translation preferences")
        translation_desc.setStyleSheet("color: #666; margin-bottom: 15px;")

        translation_layout.addWidget(translation_title)
        translation_layout.addWidget(translation_desc)
        translation_layout.addWidget(ActionButton("Language Preferences"))
        translation_layout.addWidget(ActionButton("Recognition Sensitivity"))

        layout.addWidget(camera_group)
        layout.addWidget(translation_group)
        layout.addStretch()
        self.save_btn = ActionButton("Save Settings")
        self.save_btn.clicked.connect(self.handle_save_settings)
        self.reset_btn = ActionButton("Reset Settings")
        self.reset_btn.clicked.connect(self.handle_reset_settings)

    def handle_save_settings(self):
        """Handler für Save Settings Button"""
        # Save debug mode to main window
        mw = self.parent().parent().parent()
        if hasattr(mw, 'set_debug_mode'):
            mw.set_debug_mode(self.debug_checkbox.isChecked())
    def handle_reset_settings(self):
        """Handler für Reset Settings Button"""
        self.debug_checkbox.setChecked(False)
        self.handle_save_settings()


class HelpPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Title
        title = QLabel("Help & Documentation")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        layout.addWidget(title)

        help_frame = QFrame()
        help_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        help_layout = QVBoxLayout(help_frame)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setStyleSheet("""
            QTextEdit {
                border: none;
                color: #333;
                font-size: 14px;
                background-color: transparent;
            }
        """)
        help_text.setHtml("""
            <h2>📖 Sign Language Translator Help</h2>
            <p>Welcome to the Sign Language Translator application. Here's how to use it:</p>

            <h3>🎥 Translator</h3>
            <ul>
                <li>Click 'Start Camera' to begin translation</li>
                <li>Position your hands clearly in front of the camera</li>
                <li>Make signs and see the translation appear below</li>
            </ul>

            <h3>📚 Dictionary</h3>
            <ul>
                <li>Browse through common signs</li>
                <li>Use the search bar to find specific signs</li>
                <li>Click on signs to see detailed information</li>
            </ul>

            <h3>✏️ Practice</h3>
            <ul>
                <li>Choose between Tutorial and Quiz modes</li>
                <li>Follow the instructions on screen</li>
                <li>Track your progress over time</li>
            </ul>
        """)

        help_layout.addWidget(help_text)
        layout.addWidget(help_frame)
        self.contact_btn = ActionButton("Contact Support")
        self.contact_btn.clicked.connect(self.handle_contact_support)

    def handle_contact_support(self):
        """Handler für Contact Support Button"""
        # Hier Logik für Support-Kontakt einfügen
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Translator")
        self.setMinimumSize(1200, 800)
        self.debug_mode = False

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create sidebar
        self.sidebar = Sidebar()
        layout.addWidget(self.sidebar)

        # Create main content area
        content_container = QWidget()
        content_container.setStyleSheet("""
            QWidget {
                background-color: #f0f2f5;
            }
        """)
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Create stacked widget for different pages
        self.stacked_widget = QStackedWidget()
        self.translator_page = TranslatorPage()
        self.dictionary_page = DictionaryPage()
        self.contribute_page = ContributePage()
        self.settings_page = SettingsPage()
        self.help_page = HelpPage()

        self.stacked_widget.addWidget(self.translator_page)
        self.stacked_widget.addWidget(self.dictionary_page)
        self.stacked_widget.addWidget(self.contribute_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.help_page)

        content_layout.addWidget(self.stacked_widget)
        layout.addWidget(content_container)

        # Connect sidebar buttons
        self.sidebar.translator_btn.clicked.connect(lambda: self.show_page(0))
        self.sidebar.dictionary_btn.clicked.connect(lambda: self.show_page(1))
        self.sidebar.contribute_btn.clicked.connect(lambda: self.show_page(2))
        self.sidebar.settings_btn.clicked.connect(lambda: self.show_page(3))
        self.sidebar.help_btn.clicked.connect(lambda: self.show_page(4))

        # Set initial page
        self.sidebar.translator_btn.setChecked(True)
        self.show_page(0)

    def show_page(self, index):
        self.stacked_widget.setCurrentIndex(index)
        buttons = [
            self.sidebar.translator_btn,
            self.sidebar.dictionary_btn,
            self.sidebar.contribute_btn,
            self.sidebar.settings_btn,
            self.sidebar.help_btn
        ]
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)

    def set_debug_mode(self, value):
        self.debug_mode = value

    def get_debug_mode(self):
        return getattr(self, 'debug_mode', False)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
