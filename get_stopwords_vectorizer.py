from flask_api import FlaskAPI
from flask import request
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

import json


app = FlaskAPI(__name__)


@app.route('/get_word_count_vectorizer/', methods=['GET', 'POST'])
def get_word_count_vectorizer():
	if request.method == 'GET':
		return 'Invalid Request'
	

	if request.method == 'POST':
		data =request.get_json()
		output = {}
		if not data:
			output = {'message': 'No Text Received'}
			return output
		for key in data.keys():
			text = data[key]
			text = text.split(' ')
			non_stop_words = []
			en_stops = set(stopwords.words('english'))
			for word in text: 
				if word not in en_stops:
					non_stop_words.append(word)
			output[key] = non_stop_words
			cv=CountVectorizer()
			word_count_vector=cv.fit_transform(text)
			tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
			tfidf_transformer.fit(word_count_vector)
		
			#import pdb; pdb.set_trace()
			df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["tf_idf_weights"])
			return df_idf.to_json()

			#df_idf.sort_values(by=['tf_idf_weights'])

		 
		



if __name__ == '__main__':
	app.run()
