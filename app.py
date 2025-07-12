import sys
import os
import cv2
import tempfile
from PySide6 import QtCore, QtWidgets, QtGui
import importlib.util
import pyqtgraph as pg
import pandas as pd
from PySide6.QtCore import QPropertyAnimation, QEasingCurve

# Hilfsfunktion: Pipeline dynamisch importieren
spec_pre = importlib.util.spec_from_file_location("preprocessing_live_data", os.path.join(os.getcwd(), "preprocessing_live_data.py"))
if spec_pre is not None and spec_pre.loader is not None:
    preprocessing_live_data = importlib.util.module_from_spec(spec_pre)
    spec_pre.loader.exec_module(preprocessing_live_data)
else:
    preprocessing_live_data = None

spec_inf = importlib.util.spec_from_file_location("inference", os.path.join(os.getcwd(), "inference.py"))
if spec_inf is not None and spec_inf.loader is not None:
    inference = importlib.util.module_from_spec(spec_inf)
    spec_inf.loader.exec_module(inference)
else:
    inference = None

APP_ACCENT = "#2196F3"
APP_ACCENT_GRAD = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2196F3, stop:1 #9C27B0)"
FONT_FAMILY = "'Segoe UI', 'Arial', sans-serif"
BG_CARD = "#fff"
BG_GREY = "#f4f6fb"

# --- Theme Stylesheets ---
LIGHT_THEME = f'''
QWidget {{ background: #f4f6fb; color: #232946; font-family: {FONT_FAMILY}; }}
QFrame, QGroupBox {{ background: #fff; border-radius: 18px; }}
QPushButton {{ background: {APP_ACCENT_GRAD}; color: #fff; }}
QLabel {{ color: #232946; }}
'''
DARK_THEME = f'''
QWidget {{ background: #181c24; color: #eaf1fa; font-family: {FONT_FAMILY}; }}
QFrame, QGroupBox {{ background: #232946; border-radius: 18px; }}
QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #232946, stop:1 #2196F3); color: #fff; }}
QLabel {{ color: #eaf1fa; }}
'''

class Sidebar(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(230)
        self.setStyleSheet(f"""
            background: {APP_ACCENT_GRAD};
            color: #fff;
            border-top-right-radius: 18px;
            border-bottom-right-radius: 18px;
            box-shadow: 2px 0 16px rgba(33,150,243,0.10);
        """)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        # Professionelles Logo-Branding
        logo_hbox = QtWidgets.QHBoxLayout()

        appname = QtWidgets.QLabel("SignAI")
        appname.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        appname.setStyleSheet(f"""
            font-size: 2.5rem;
            font-weight: 900;
            letter-spacing: 2.5px;
            font-family: 'Montserrat', {FONT_FAMILY};
            margin-bottom: 2.2rem;
            margin-top: 1.5rem;
            color: #fff;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2196F3, stop:1 #9C27B0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-fill-color: transparent;
            text-shadow: 0 6px 24px rgba(33,150,243,0.18), 0 1px 0 #fff2;
        """)
        logo_hbox.addWidget(appname)
        logo_hbox.addStretch()
        logo_hbox.setContentsMargins(0, 18, 0, 10)
        layout.addLayout(logo_hbox)
        # Trennlinie
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #fff; background: #fff; min-height: 2px; margin: 1.2rem 0 1.2rem 0; opacity:0.18;")
        layout.addWidget(line)
        self.buttons = {}
        self.button_names = ["Home", "Live-Übersetzung", "Modell", "Einstellungen"]
        for i, name in enumerate(self.button_names):
            btn = QtWidgets.QPushButton(name)
            btn.setStyleSheet(f"""
                padding: 1.1rem 1.2rem 1.1rem 1.7rem;
                font-size: 1.13rem;
                border: none;
                background: #232946;
                color: #eaf1fa;
                border-radius: 10px;
                margin-bottom: 0.7rem;
                font-family: {FONT_FAMILY};
                text-align: left;
                position: relative;
                transition: background 0.2s, color 0.2s, box-shadow 0.2s;
            """)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.installEventFilter(self)
            layout.addWidget(btn)
            self.buttons[name] = btn
        layout.addStretch()
        self.set_active("Home")

    def eventFilter(self, obj, event):
        if isinstance(obj, QtWidgets.QPushButton):
            if event.type() == QtCore.QEvent.Type.Enter:
                if obj.styleSheet().find("background: #fff;") == -1:
                    obj.setStyleSheet(obj.styleSheet() + "background: #2d3142; color: #fff; box-shadow: 0 4px 16px rgba(33,150,243,0.10);")
            elif event.type() == QtCore.QEvent.Type.Leave:
                if obj.styleSheet().find("background: #fff;") == -1:
                    obj.setStyleSheet(obj.styleSheet().replace("background: #2d3142; color: #fff; box-shadow: 0 4px 16px rgba(33,150,243,0.10);", "background: #232946; color: #eaf1fa; box-shadow: none;"))
        return super().eventFilter(obj, event)

    def set_active(self, name):
        for btn_name, btn in self.buttons.items():
            if btn_name == name:
                btn.setStyleSheet(f"""
                    padding: 1.1rem 1.2rem 1.1rem 1.7rem;
                    font-size: 1.13rem;
                    border: none;
                    background: #fff;
                    color: {APP_ACCENT};
                    border-radius: 10px;
                    margin-bottom: 0.7rem;
                    font-family: {FONT_FAMILY};
                    text-align: left;
                    font-weight: bold;
                    box-shadow: 0 4px 18px rgba(33,150,243,0.13);
                    position: relative;
                    border-left: 6px solid {APP_ACCENT};
                    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
                """)
            else:
                btn.setStyleSheet(f"""
                    padding: 1.1rem 1.2rem 1.1rem 1.7rem;
                    font-size: 1.13rem;
                    border: none;
                    background: #232946;
                    color: #eaf1fa;
                    border-radius: 10px;
                    margin-bottom: 0.7rem;
                    font-family: {FONT_FAMILY};
                    text-align: left;
                    font-weight: normal;
                    box-shadow: none;
                    position: relative;
                    border-left: 6px solid transparent;
                    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
                """)

class Card(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {BG_CARD}; border-radius: 18px; box-shadow: 0 4px 24px rgba(33,150,243,0.07); padding: 2.2rem 2.2rem 1.7rem 2.2rem; margin: 2.5rem 2.5rem 0 0;")

# Entferne KeypointPlotWidget und alle Verwendungen

class VideoCaptureWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vbox = QtWidgets.QVBoxLayout()
        self.setLayout(self.vbox)
        # --- Video Preview ---
        self.video_label = QtWidgets.QLabel("Webcam-Feed erscheint hier")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(640, 360)
        self.video_label.setStyleSheet("border-radius: 22px; background: #181c24; box-shadow: 0 6px 32px rgba(33,150,243,0.13); border: 2.5px solid #e3e8f0; margin-bottom: 1.5rem;")
        self.vbox.addWidget(self.video_label, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        # --- Aufnahme-Buttons ---
        btn_hbox = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("● Aufnahme starten")
        self.btn_stop = QtWidgets.QPushButton("■ Aufnahme stoppen")
        self.btn_stop.setEnabled(False)
        self.btn_translate = QtWidgets.QPushButton("→ Übersetzen")
        self.btn_translate.setEnabled(False)
        for btn in [self.btn_start, self.btn_stop, self.btn_translate]:
            btn.setMinimumHeight(48)
            btn.setMinimumWidth(180)
            btn.setStyleSheet(f"""
                font-size: 1.18rem;
                font-family: {FONT_FAMILY};
                border-radius: 12px;
                margin: 0 0.7rem 0 0;
                background: {APP_ACCENT_GRAD};
                color: #fff;
                font-weight: 700;
                box-shadow: 0 2px 12px rgba(33,150,243,0.10);
                transition: background 0.2s, color 0.2s, box-shadow 0.2s, transform 0.2s;
            """)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.setGraphicsEffect(self._make_shadow())
        self.btn_start.setStyleSheet(self.btn_start.styleSheet() + "margin-top: 0.5rem; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #43a047, stop:1 #2196F3);")
        self.btn_stop.setStyleSheet(self.btn_stop.styleSheet() + "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #c62828, stop:1 #f44336);")
        btn_hbox.addWidget(self.btn_start)
        btn_hbox.addWidget(self.btn_stop)
        btn_hbox.addWidget(self.btn_translate)
        self.vbox.addLayout(btn_hbox)
        # --- Statusanzeige ---
        self.status = QtWidgets.QLabel("Status: Bereit")
        self.status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet(f"font-size: 1.13rem; margin: 1.5rem 0 0.7rem 0; padding: 0.9rem 1.2rem; border-radius: 10px; background: #e3f2ff; color: #1976d2; font-family: {FONT_FAMILY}; box-shadow: 0 2px 8px rgba(33,150,243,0.07);")
        self.status_icon = QtWidgets.QLabel()
        self.status_icon.setFixedWidth(36)
        self.status_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_icon.setStyleSheet("font-size: 1.7rem;")
        status_hbox = QtWidgets.QHBoxLayout()
        status_hbox.addWidget(self.status_icon)
        status_hbox.addWidget(self.status)
        self.vbox.addLayout(status_hbox)
        # --- Fortschrittsbalken ---
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedHeight(18)
        self.progress.setStyleSheet(f"QProgressBar {{ border-radius: 9px; background: #f4f6fb; color: #1976d2; font-size: 1.05rem; }} QProgressBar::chunk {{ background: {APP_ACCENT_GRAD}; border-radius: 9px; }}")
        self.vbox.addWidget(self.progress)
        # --- Ergebnis-Card (zunächst versteckt) ---
        self.result_card = QtWidgets.QFrame()
        self.result_card.setStyleSheet(f"background: #fff; border-radius: 22px; box-shadow: 0 6px 32px rgba(33,150,243,0.13); padding: 2.5rem 2.5rem 2.2rem 2.5rem; margin: 2.5rem 0 0 0;")
        self.result_card.setVisible(False)
        result_vbox = QtWidgets.QVBoxLayout(self.result_card)
        self.result_icon = QtWidgets.QLabel("✅")
        self.result_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.result_icon.setStyleSheet("font-size: 3.2rem; margin-bottom: 0.7rem;")
        self.result_text = QtWidgets.QLabel("")
        self.result_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.result_text.setStyleSheet(f"font-size: 2.1rem; font-weight: 700; color: #2196F3; font-family: {FONT_FAMILY}; margin-bottom: 0.7rem;")
        self.keypoint_preview = QtWidgets.QLabel("[Keypoint-Preview folgt]")
        self.keypoint_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.keypoint_preview.setStyleSheet("font-size: 1.1rem; color: #888; margin-bottom: 1.2rem;")
        self.copy_btn = QtWidgets.QPushButton("Ergebnis kopieren")
        self.copy_btn.setStyleSheet(f"font-size: 1.08rem; border-radius: 8px; background: {APP_ACCENT_GRAD}; color: #fff; font-weight: 600; padding: 0.7rem 1.2rem; margin-bottom: 0.5rem;")
        self.copy_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.copy_btn.clicked.connect(self.copy_result)
        self.again_btn = QtWidgets.QPushButton("Nochmal aufnehmen")
        self.again_btn.setStyleSheet(f"font-size: 1.08rem; border-radius: 8px; background: #f4f6fb; color: #1976d2; font-weight: 600; padding: 0.7rem 1.2rem; margin-bottom: 0.5rem;")
        self.again_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.again_btn.clicked.connect(self.reset_ui)
        result_vbox.addWidget(self.result_icon)
        result_vbox.addWidget(self.result_text)
        result_vbox.addWidget(self.keypoint_preview)
        result_vbox.addWidget(self.copy_btn)
        result_vbox.addWidget(self.again_btn)
        self.vbox.addWidget(self.result_card, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
        # Entferne KeypointPlotWidget(), self.keypoint_plot.setVisible(False), result_vbox.insertWidget(0, self.keypoint_plot)
        # --- Timer, Kamera, Events ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.recording = False
        self.out = None
        self.temp_video_path = None
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        self.btn_translate.clicked.connect(self.run_translation)
        # --- UX Verbesserungen ---
        self.setToolTip("Leertaste: Start/Stop Aufnahme, Q: Beenden, → Übersetzen")
        self.result = None
        # --- Verlauf der letzten Übersetzungen ---
        self.history = []
        self.history_box = QtWidgets.QHBoxLayout()
        self.history_widget = QtWidgets.QWidget()
        self.history_widget.setLayout(self.history_box)
        self.history_widget.setVisible(False)
        self.vbox.addWidget(self.history_widget)
        # --- Lade-Spinner ---
        self.spinner = QtWidgets.QLabel()
        self.spinner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.spinner.setVisible(False)
        self.spinner.setStyleSheet("font-size: 2.2rem; margin: 1.2rem 0;")
        self.spinner.setText("⏳")
        self.vbox.addWidget(self.spinner)

    def _make_shadow(self):
        effect = QtWidgets.QGraphicsDropShadowEffect()
        effect.setBlurRadius(14)
        effect.setColor(QtGui.QColor(33,150,243, 70))
        effect.setOffset(0, 3)
        return effect

    def start_recording(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.set_status("Fehler beim Öffnen der Kamera", "❌", error=True)
            return
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        fd, self.temp_video_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        self.out = cv2.VideoWriter(self.temp_video_path, fourcc, 20.0, (640, 360))
        self.recording = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_translate.setEnabled(False)
        self.set_status("Aufnahme läuft...", "●", color="#43a047")
        self.progress.setVisible(False)
        self.result_card.setVisible(False)
        self.timer.start(30)

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (640, 360))
                rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                if self.recording and self.out is not None:
                    self.out.write(frame_resized)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap)
            else:
                self.set_status("Kein Kamerabild", "❌", error=True)

    def stop_recording(self):
        self.recording = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_translate.setEnabled(True)
        self.set_status(f"Aufnahme beendet. Datei: {self.temp_video_path}", "✔️", color="#1976d2")
        self.progress.setVisible(False)
        self.result_card.setVisible(False)
        self.show_video_preview()

    def show_video_preview(self):
        if not self.temp_video_path:
            self.set_status("Kein Video zum Anzeigen", "❌", error=True)
            return
        cap = cv2.VideoCapture(self.temp_video_path)
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (640, 360))
            rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)
        cap.release()

    def run_translation(self):
        if preprocessing_live_data is None or inference is None:
            self.set_status("Fehler: KI-Pipeline nicht gefunden!", "❌", error=True)
            return
        self.set_status("Preprocessing läuft...", "⏳", color="#fbc02d")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.spinner.setVisible(True)
        self.fade_in(self.spinner)
        QtWidgets.QApplication.processEvents()
        try:
            preprocessing_live_data.main(self.temp_video_path)
            self.set_status("Inference läuft...", "🤖", color="#2196F3")
            QtWidgets.QApplication.processEvents()
            result = inference.main_inference("models/trained_model_v21.keras")
            self.result = result
            self.set_status(f"Fertig! Ergebnis: {result}", "✅", color="#43a047")
            self.progress.setVisible(False)
            self.spinner.setVisible(False)
            self.show_result(result)
        except Exception as e:
            self.set_status(f"Fehler: {str(e)}", "❌", error=True)
            self.progress.setVisible(False)
            self.spinner.setVisible(False)

    def show_result(self, result):
        self.result_card.setVisible(True)
        self.result_text.setText(result.upper())
        self.result_icon.setText("✅")
        # Entferne Keypoint-Visualisierung laden und die if/else-Logik für keypoint_plot
        self.keypoint_preview.setText("[Keypoint-Preview folgt]")
        self.keypoint_preview.setVisible(True)
        # --- Verlauf aktualisieren ---
        if result:
            self.history.insert(0, result)
            self.history = self.history[:5]
            self.update_history()

    def copy_result(self):
        if self.result:
            clipboard = QtWidgets.QApplication.clipboard()
            clipboard.setText(self.result)
            self.set_status("Ergebnis kopiert!", "📋", color="#2196F3")

    def reset_ui(self):
        self.result_card.setVisible(False)
        self.result = None
        self.set_status("Bereit", "", color="#1976d2")
        self.video_label.setText("Webcam-Feed erscheint hier")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_translate.setEnabled(False)
        self.progress.setVisible(False)
        self.history = [] # Reset history on reset
        self.update_history() # Hide history widget if no history

    def set_status(self, text, icon, color="#1976d2", error=False):
        self.status.setText(f"Status: {text}")
        self.status_icon.setText(icon)
        if error:
            self.status.setStyleSheet(f"font-size: 1.13rem; margin: 1.5rem 0 0.7rem 0; padding: 0.9rem 1.2rem; border-radius: 10px; background: #ffebee; color: #c62828; font-family: {FONT_FAMILY}; box-shadow: 0 2px 8px rgba(33,150,243,0.07);")
        else:
            self.status.setStyleSheet(f"font-size: 1.13rem; margin: 1.5rem 0 0.7rem 0; padding: 0.9rem 1.2rem; border-radius: 10px; background: #e3f2ff; color: {color}; font-family: {FONT_FAMILY}; box-shadow: 0 2px 8px rgba(33,150,243,0.07);")

    def update_history(self):
        # Verlauf als Cards anzeigen
        for i in reversed(range(self.history_box.count())):
            widget = self.history_box.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        for word in self.history:
            card = QtWidgets.QLabel(word.upper())
            card.setStyleSheet(f"background: #e3f2ff; color: #1976d2; border-radius: 10px; font-size: 1.1rem; font-weight: 600; padding: 0.7rem 1.2rem; margin-right: 0.7rem;")
            card.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.history_box.addWidget(card)
        self.history_widget.setVisible(len(self.history) > 0)
        self.fade_in(self.history_widget)

    def fade_in(self, widget):
        anim = QPropertyAnimation(widget, b"windowOpacity")
        anim.setDuration(500)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        widget.setWindowOpacity(0.0)
        widget.setVisible(True)
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

# Entferne die Methode load_keypoints_from_csv

class MainArea(QtWidgets.QStackedWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.home = QtWidgets.QWidget()
        home_layout = QtWidgets.QVBoxLayout(self.home)
        card = Card()
        card_layout = QtWidgets.QVBoxLayout(card)
        headline = QtWidgets.QLabel("Willkommen bei SignAI!")
        headline.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        headline.setStyleSheet(f"font-size: 2.2rem; font-weight: 700; font-family: {FONT_FAMILY}; margin-bottom: 0.7rem;")
        akzent = QtWidgets.QFrame()
        akzent.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        akzent.setStyleSheet(f"background: {APP_ACCENT}; min-height: 4px; max-width: 60px; border-radius: 2px; margin-bottom: 1.5rem; margin-left: 0;")
        card_layout.addWidget(headline)
        card_layout.addWidget(akzent)
        info = QtWidgets.QLabel("SignAI ist eine professionelle KI-Anwendung zur Übersetzung von Gebärdensprache in Text.\n\nNimm ein Video auf oder lade eines hoch, lasse es von der KI analysieren und erhalte eine direkte Übersetzung.\n\nFeatures:\n• Echtzeit-Übersetzung\n• Moderne KI-Modelle\n• Keypoint-Visualisierung\n• Datenschutz: Alles bleibt auf deinem Gerät\n\nTeste die Live-Übersetzung oder erfahre mehr im Modell-Tab.")
        info.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        info.setStyleSheet(f"font-size: 1.18rem; color: #444; font-family: {FONT_FAMILY}; margin-bottom: 0.5rem;")
        card_layout.addWidget(info)
        home_layout.addWidget(card)
        home_layout.addStretch()
        self.addWidget(self.home)

        self.live = VideoCaptureWidget()
        self.addWidget(self.live)

        self.model = QtWidgets.QWidget()
        model_layout = QtWidgets.QVBoxLayout(self.model)
        card2 = Card()
        card2_layout = QtWidgets.QVBoxLayout(card2)
        headline2 = QtWidgets.QLabel("Modellarchitektur & Pipeline-Flow")
        headline2.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        headline2.setStyleSheet(f"font-size: 1.6rem; font-weight: 600; font-family: {FONT_FAMILY}; margin-bottom: 0.7rem;")
        akzent2 = QtWidgets.QFrame()
        akzent2.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        akzent2.setStyleSheet(f"background: {APP_ACCENT}; min-height: 3px; max-width: 40px; border-radius: 2px; margin-bottom: 1.2rem; margin-left: 0;")
        card2_layout.addWidget(headline2)
        card2_layout.addWidget(akzent2)
        modelinfo = QtWidgets.QLabel("Das Modell nutzt Deep Learning (Seq2Seq, LSTM, Attention) und verarbeitet Keypoints aus Video-Frames.\n\nPipeline:\n1. Videoaufnahme/-upload\n2. Keypoint-Extraktion\n3. Preprocessing\n4. Modell-Inferenz\n5. Übersetzung\n\nAlle Schritte laufen lokal und datenschutzfreundlich.")
        modelinfo.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        modelinfo.setStyleSheet(f"font-size: 1.13rem; color: #444; font-family: {FONT_FAMILY}; margin-bottom: 0.5rem;")
        card2_layout.addWidget(modelinfo)
        model_layout.addWidget(card2)
        model_layout.addStretch()
        self.addWidget(self.model)

        self.settings = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(self.settings)
        settings_card = Card()
        settings_card_layout = QtWidgets.QVBoxLayout(settings_card)
        settings_head = QtWidgets.QLabel("Einstellungen")
        settings_head.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        settings_head.setStyleSheet(f"font-size: 1.4rem; color: {APP_ACCENT}; font-family: {FONT_FAMILY}; font-weight: bold; margin-bottom: 1.2rem;")
        settings_card_layout.addWidget(settings_head)
        # Dark/Light Mode Umschalter
        theme_hbox = QtWidgets.QHBoxLayout()
        theme_icon = QtWidgets.QLabel("🌙")
        theme_icon.setFixedWidth(32)
        theme_label = QtWidgets.QLabel("Theme:")
        theme_label.setStyleSheet(f"font-size: 1.1rem; font-family: {FONT_FAMILY}; margin-right: 1rem;")
        theme_switch = QtWidgets.QCheckBox("Dark Mode")
        theme_switch.setStyleSheet(f"font-size: 1.1rem; font-family: {FONT_FAMILY};")
        theme_hbox.addWidget(theme_icon)
        theme_hbox.addWidget(theme_label)
        theme_hbox.addWidget(theme_switch)
        theme_hbox.addStretch()
        settings_card_layout.addLayout(theme_hbox)
        # Sprache (mit Flaggen)
        lang_hbox = QtWidgets.QHBoxLayout()
        lang_icon = QtWidgets.QLabel("🌐")
        lang_icon.setFixedWidth(32)
        lang_label = QtWidgets.QLabel("Sprache:")
        lang_label.setStyleSheet(f"font-size: 1.1rem; font-family: {FONT_FAMILY}; margin-right: 1rem;")
        lang_combo = QtWidgets.QComboBox()
        lang_combo.addItem("🇩🇪 Deutsch")
        lang_combo.addItem("🇬🇧 Englisch")
        lang_combo.addItem("🇫🇷 Französisch")
        lang_combo.setStyleSheet(f"font-size: 1.1rem; font-family: {FONT_FAMILY};")
        lang_hbox.addWidget(lang_icon)
        lang_hbox.addWidget(lang_label)
        lang_hbox.addWidget(lang_combo)
        lang_hbox.addStretch()
        settings_card_layout.addLayout(lang_hbox)
        # Info
        info_icon = QtWidgets.QLabel("ℹ️")
        info_icon.setFixedWidth(32)
        info_label = QtWidgets.QLabel("SignAI ist ein Open-Source-Projekt.\nAlle Daten werden lokal verarbeitet.\nFeedback und Ideen gerne an: signai@projekt.de")
        info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        info_label.setStyleSheet(f"font-size: 1.05rem; color: #555; margin-top: 1.5rem; font-family: {FONT_FAMILY};")
        info_hbox = QtWidgets.QHBoxLayout()
        info_hbox.addWidget(info_icon)
        info_hbox.addWidget(info_label)
        settings_card_layout.addLayout(info_hbox)
        settings_layout.addWidget(settings_card)
        settings_layout.addStretch()
        self.addWidget(self.settings)

class TopBar(QtWidgets.QWidget):
    def __init__(self, parent=None, theme_callback=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self.setStyleSheet(f"background: #fff; border-bottom: 1px solid #e3e8f0; font-family: {FONT_FAMILY}; box-shadow: 0 2px 12px rgba(33,150,243,0.04);")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.app_title = QtWidgets.QLabel("SignAI Desktop")
        self.app_title.setStyleSheet(f"font-size: 1.25rem; font-weight: 600; color: {APP_ACCENT}; font-family: {FONT_FAMILY}; letter-spacing: 1px;")
        layout.addStretch()
        layout.addWidget(self.app_title, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        # Theme Toggle Button
        self.theme_btn = QtWidgets.QPushButton("🌙")
        self.theme_btn.setFixedSize(38, 38)
        self.theme_btn.setStyleSheet(f"font-size: 1.3rem; border-radius: 19px; background: #e3e8f0; color: #232946; margin-right: 12px;")
        self.theme_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.theme_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.login_btn = QtWidgets.QPushButton("Anmelden  ⎈")
        self.login_btn.setStyleSheet(f"font-size: 1.1rem; padding: 0.7rem 1.7rem; border-radius: 8px; background: {APP_ACCENT_GRAD}; color: #fff; font-weight: 600; border: none; box-shadow: 0 2px 8px rgba(33,150,243,0.08); letter-spacing: 0.5px; margin-right: 18px;")
        self.login_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.login_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.theme_callback = theme_callback
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.is_dark = False
    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self.theme_btn.setText("☀️" if self.is_dark else "🌙")
        if self.theme_callback:
            self.theme_callback(self.is_dark)

class SignAIApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignAI - Desktop")
        self.resize(1280, 850)
        self.theme_is_dark = False
        vlayout = QtWidgets.QVBoxLayout(self)
        vlayout.setContentsMargins(0,0,0,0)
        self.topbar = TopBar(theme_callback=self.set_theme)
        vlayout.addWidget(self.topbar)
        hbox = QtWidgets.QHBoxLayout()
        self.sidebar = Sidebar()
        self.mainarea = MainArea()
        hbox.addWidget(self.sidebar)
        hbox.addWidget(self.mainarea, 1)
        vlayout.addLayout(hbox)
        # Navigation
        for name, idx in zip(self.sidebar.button_names, range(4)):
            self.sidebar.buttons[name].clicked.connect(lambda checked, n=name, i=idx: self._navigate(n, i))
        self.sidebar.set_active("Home")
        self.set_theme(False)
    def set_theme(self, is_dark):
        self.theme_is_dark = is_dark
        if is_dark:
            self.setStyleSheet(DARK_THEME)
        else:
            self.setStyleSheet(LIGHT_THEME)

    def _navigate(self, name, idx):
        self.mainarea.setCurrentIndex(idx)
        self.sidebar.set_active(name)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = SignAIApp()
    window.show()
    sys.exit(app.exec())
