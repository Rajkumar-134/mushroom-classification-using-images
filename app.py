from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
app = Flask(__name__)

dic = {0 : 'Edible', 1 : 'Poisonous'}
model = load_model('model.h5')

model.make_predict_function()

def predict_label(img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predicted_probabilities = model.predict(img_array)
    predicted_class = np.argmax(predicted_probabilities)
    predicted_label = dic[predicted_class]
    return predicted_label


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)