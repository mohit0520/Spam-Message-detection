import joblib
import string
import nltk
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import sklearn
print(sklearn.__version__)

model = joblib.load('extra_trees_model.pkl')



loaded_text = pd.read_pickle('cleaned_text.pkl')




ps = PorterStemmer()

def preprocess_text(text):
  
    text = nltk.word_tokenize(text.lower())
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            stemming = ps.stem(i)
            y.append(stemming)

    return " ".join(y)




def main():
  
    st.title('Spam Classifier')
    st.markdown('Enter a message to check if it is spam or not.')

  
    user_input = st.text_input('Enter a message')

    # Create a button to trigger the prediction
    if st.button('Predict'):
        # Preprocess the input message
        preprocessed_text = preprocess_text(user_input)

        # Transform the preprocessed text using the pre-trained TfidfVectorizer
        tf = TfidfVectorizer()
        # %%
        tf.fit(loaded_text)

        vectorized_text = tf.transform([preprocessed_text]).toarray()


        prediction = model.predict(vectorized_text)

      
        if prediction == 1:
            st.error('The message is classified as spam.')
        else:
            st.success('The message is not spam.')


if __name__ == '__main__':
    main()
    pass