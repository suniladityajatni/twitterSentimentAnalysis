import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')

import streamlit as st
import tensorflow as tf
import string
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from nltk.stem.snowball import SnowballStemmer

stop_words = set(stopwords.words('english'))

def clean_text(text):
    
    s=text.split(' ')
    
    s=[w.lower() for w in s]
    
    table = str.maketrans('', '', string.punctuation)
    s = [w.translate(table) for w in s]
    s = [word for word in s if word.isalpha()]
    
    s = [w for w in s if not w in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    s = [lemmatizer.lemmatize(word) for word in s]
    
    final_text=' '.join(s)
    return final_text



def padding_seq(tokenizer,seq,maxlen,vocab_size):
    seq=tokenizer.texts_to_sequences(seq)
    padded_seq=tf.keras.preprocessing.sequence.pad_sequences(seq,truncating='post',padding='post',maxlen=maxlen)
    return padded_seq

file="tokenizer.pkl"
fileobj=open(file,"rb")
tokenizer=pickle.load(fileobj)

model=tf.keras.models.load_model("TwitterSentimentAnalysisLSTMNetwork.h5")

def Predict(text):
    global tokenizer,model
    text=clean_text(text)
    text=[text]
    maxlen=45           
    vocab_size=30000
    padded_text=padding_seq(tokenizer,text,maxlen,vocab_size)
    y_pred=model.predict(padded_text)
    y_pred=np.argmax(y_pred,axis=1)

    if(y_pred):
        return "Positive"
    else:
        return "Negative"



def main():
    st.title("Twitter Sentiment Analysis")
    tweet=st.text_input("Tweet","Type Here")
    result="Failed"
    if st.button("Predict"):
        result=Predict(tweet)
        st.success("The sentiment is {}".format(result))

if __name__=="__main__":
    main()
