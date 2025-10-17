from flask import Flask
from flask import request, render_template, url_for, redirect

# Initialisiere die Flask-Anwendung
app = Flask(__name__)

# DELETE THIS IN PRODUKTION
app.config.update(
    DEBUG=True,
    TEMPLATES_AUTO_RELOAD=True
)

# main route
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/download")
def download():
    return render_template("download.html")

# start the server
if __name__ == '__main__':
    app.run(debug=True)