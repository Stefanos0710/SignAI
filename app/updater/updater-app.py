from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

# start application
app = QApplication([])

# load ui
ui_file = QFile("updater_app.ui")
ui_file.open(QFile.ReadOnly)

loader = QUiLoader()
window = loader.load(ui_file)
ui_file.close()

# load qss
with open("style.qss", "r") as f:
    qss = f.read()
    app.setStyleSheet(qss)

window.show()
app.exec()