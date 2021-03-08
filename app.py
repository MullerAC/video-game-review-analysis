from flask import Flask, request, render_template

from scripts import WebInterface

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route('/', methods = ['GET'])
def render_homepage():
    examples = WebInterface.get_examples()
    return render_template('homepage.html', data=examples)

@app.route('/', methods = ['POST'])
def predict():
    search = request.form.get('search')
    data = WebInterface.get_web_output(search)
    return render_template('output.html', data=data, source=search)

if __name__ == '__main__':
    app.run()