from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

# QApplication starten
app = QApplication([])

# UI-Datei laden
ui_file = QFile("updater_app.ui")
ui_file.open(QFile.ReadOnly)

loader = QUiLoader()
window = loader.load(ui_file)
ui_file.close()

# QSS laden
with open("style.qss", "r") as f:
    qss = f.read()
    app.setStyleSheet(qss)

window.show()
app.exec()
