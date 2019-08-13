from flask import Flask,render_template,url_for,request
import Review_Ana_App
# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
	print('GET /')
	return render_template('index.html',message=None,QL=0,DE=0,PK=0,SE=0)


@app.route('/', methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		# corpus1 = message
		#print(message)
		my_QL=0
		my_DE=0
		my_PK=0
		my_SE=0
		my_QL,my_DE,my_PK,my_SE=Review_Ana_App.getResults(message)
		#print("Printing this")
		#print(Review_Ana_App.getResults(message))
		#x = [message,my_QL,my_DM,my_DM]
		#print(message)
		#print(my_QL,my_DE,my_PK,my_SE)
	return render_template('index.html', message=message,QL=my_QL,DE=my_DE,PK=my_PK, SE=my_SE)
	
	
if __name__ == '__main__':
	app.run(debug=True)