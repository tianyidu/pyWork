#encoding:utf-8

from flask import Flask,render_template,url_for
import config
from exts import db
from models import User

app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)

@app.route("/")
def index():
	user = User(id="1")
	db.session.add(user)
	db.session.commit() 
	return render_template("index.html")

if __name__=="__main__":
	app.run(port=8080)