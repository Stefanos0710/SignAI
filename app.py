import sys
import cv2
import mediapipe as mp
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QTextEdit,
                               QStackedWidget, QFrame, QScrollArea)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QImage, QPixmap, QFont
import numpy as np
import preprocessing_live_data
import inference


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
        self.practice_btn = SidebarButton("Practice", "✏️")
        self.settings_btn = SidebarButton("Settings", "⚙️")
        self.help_btn = SidebarButton("Help", "❔")

        layout.addWidget(self.translator_btn)
        layout.addWidget(self.dictionary_btn)
        layout.addWidget(self.practice_btn)
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

    def update_frame(self):
        # Platzhalter für Kamera-Frame-Update-Logik
        pass

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

        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)

        self.toggle_camera_btn = ActionButton("Start Camera")
        self.toggle_camera_btn.clicked.connect(self.handle_toggle_camera)

        self.clear_btn = ActionButton("Clear Translation")
        self.clear_btn.clicked.connect(self.handle_clear_translation)

        self.translate_btn = ActionButton("Translate Video")
        self.translate_btn.clicked.connect(self.handle_translate_video)

        controls_layout.addWidget(self.toggle_camera_btn)
        controls_layout.addWidget(self.clear_btn)
        controls_layout.addWidget(self.translate_btn)
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

    def handle_toggle_camera(self):
        """Handler für Start/Stop Camera Button"""
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Startet die Kamera und das Frame-Update."""
        # Hier Kamera-Start-Logik einfügen
        self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened():
            self.is_camera_running = True
            self.toggle_camera_btn.setText("Stop Camera")
            self.timer.start(30)
        else:
            self.show_error("Could not access the camera.")

    def stop_camera(self):
        """Stoppt die Kamera und das Frame-Update."""
        # Hier Kamera-Stop-Logik einfügen
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
        self.is_camera_running = False
        self.toggle_camera_btn.setText("Start Camera")
        self.camera_label.clear()
        self.camera_label.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                border-radius: 5px;
            }
        """)

    def handle_clear_translation(self):
        """Handler für Clear Translation Button"""
        self.clear_translation()

    def clear_translation(self):
        """Setzt das Übersetzungsfeld zurück."""
        self.translation_output.clear()

    def handle_translate_video(self):
        """Handler für Translate Video Button"""
        self.translate_video()

    def translate_video(self):
        """Führt die echte Übersetzung durch (Preprocessing + Inference)."""
        try:
            self.translation_output.setText("Processing video...")
            QApplication.processEvents()
            # 1. Preprocessing
            preprocessing_live_data.main("data/live/video/recorded_video.mp4")
            self.translation_output.setText("Running AI translation...")
            QApplication.processEvents()
            # 2. Inference
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


class DictionaryPage(QWidget):
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

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

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

            content_layout.addWidget(entry)

        content_layout.addStretch()
        self.content_area.setWidget(content_widget)

        main_content_layout = QVBoxLayout(content_frame)
        main_content_layout.addWidget(self.content_area)

        layout.addWidget(search_frame)
        layout.addWidget(content_frame)

    def handle_search_signs(self):
        """Handler für Search Signs Button"""
        # Hier Logik für die Suche nach Gebärden einfügen
        pass


class PracticePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Title
        title = QLabel("Practice Your Signs")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        layout.addWidget(title)

        # Practice modes
        modes_layout = QHBoxLayout()

        # Tutorial mode
        tutorial_card = QFrame()
        tutorial_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
            }
            QFrame:hover {
                background-color: #f8f9fa;
            }
        """)
        tutorial_layout = QVBoxLayout(tutorial_card)
        tutorial_title = QLabel("📚 Tutorials")
        tutorial_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        tutorial_desc = QLabel("Learn new signs\nstep by step")
        tutorial_desc.setStyleSheet("color: #666; margin-top: 10px;")
        self.tutorial_btn = ActionButton("Start Tutorial")
        self.tutorial_btn.clicked.connect(self.handle_start_tutorial)

        tutorial_layout.addWidget(tutorial_title)
        tutorial_layout.addWidget(tutorial_desc)
        tutorial_layout.addStretch()
        tutorial_layout.addWidget(self.tutorial_btn)

        # Quiz mode
        quiz_card = QFrame()
        quiz_card.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
            }
            QFrame:hover {
                background-color: #f8f9fa;
            }
        """)
        quiz_layout = QVBoxLayout(quiz_card)
        quiz_title = QLabel("✏️ Quiz")
        quiz_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        quiz_desc = QLabel("Test your knowledge\nwith interactive quizzes")
        quiz_desc.setStyleSheet("color: #666; margin-top: 10px;")
        self.quiz_btn = ActionButton("Start Quiz")
        self.quiz_btn.clicked.connect(self.handle_start_quiz)

        quiz_layout.addWidget(quiz_title)
        quiz_layout.addWidget(quiz_desc)
        quiz_layout.addStretch()
        quiz_layout.addWidget(self.quiz_btn)

        modes_layout.addWidget(tutorial_card)
        modes_layout.addWidget(quiz_card)

        # Progress section
        progress_frame = QFrame()
        progress_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        progress_layout = QVBoxLayout(progress_frame)

        progress_title = QLabel("📊 Your Progress")
        progress_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        progress_desc = QLabel("Track your learning journey")
        progress_desc.setStyleSheet("color: #666; margin-top: 10px;")

        progress_layout.addWidget(progress_title)
        progress_layout.addWidget(progress_desc)

        layout.addLayout(modes_layout)
        layout.addWidget(progress_frame)
        layout.addStretch()

    def handle_start_tutorial(self):
        """Handler für Start Tutorial Button"""
        # Hier Logik für Tutorial-Start einfügen
        pass
    def handle_start_quiz(self):
        """Handler für Start Quiz Button"""
        # Hier Logik für Quiz-Start einfügen
        pass


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
        # Hier Logik für das Speichern der Einstellungen einfügen
        pass
    def handle_reset_settings(self):
        """Handler für Reset Settings Button"""
        # Hier Logik für das Zurücksetzen der Einstellungen einfügen
        pass


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
        self.practice_page = PracticePage()
        self.settings_page = SettingsPage()
        self.help_page = HelpPage()

        self.stacked_widget.addWidget(self.translator_page)
        self.stacked_widget.addWidget(self.dictionary_page)
        self.stacked_widget.addWidget(self.practice_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.help_page)

        content_layout.addWidget(self.stacked_widget)
        layout.addWidget(content_container)

        # Connect sidebar buttons
        self.sidebar.translator_btn.clicked.connect(lambda: self.show_page(0))
        self.sidebar.dictionary_btn.clicked.connect(lambda: self.show_page(1))
        self.sidebar.practice_btn.clicked.connect(lambda: self.show_page(2))
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
            self.sidebar.practice_btn,
            self.sidebar.settings_btn,
            self.sidebar.help_btn
        ]
        for i, btn in enumerate(buttons):
            btn.setChecked(i == index)


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
