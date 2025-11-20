from flask import Flask
from flask import request, render_template, url_for, redirect

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

@app.route("/download/instructions")
def instructions():
    return render_template("instructions.html")

@app.route("/favicon.ico")
def favicon():
    return redirect(url_for('static', filename='icon/icon.png'))

# start the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
