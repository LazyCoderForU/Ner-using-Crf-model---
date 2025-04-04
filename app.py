import streamlit as st
import spacy
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn_crfsuite import CRF

# Load trained CRF model
with open("crf_model.pkl", "rb") as f:
    crf = pickle.load(f)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")  # If you have a custom model, change to 'my_spacy_model'

# Function to extract NER features
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag
    }
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

# Streamlit UI
st.title("Named Entity Recognition (NER) using CRF")

sentence = st.text_area("Enter a sentence:", "India is playing against Australia in the 2023 World Cup.")

if st.button("Analyze"):
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    
    features = sent2features(pos_tags)
    ner_tags = crf.predict([features])[0]

    # Display results
    st.subheader("NER Results:")
    result = [(token, tag) for token, tag in zip(tokens, ner_tags)]
    st.write(result)

    # Highlight entities using spaCy visualization
    doc = nlp(sentence)
    for token, ner_tag in zip(doc, ner_tags):
        token.ent_type_ = ner_tag

    st.subheader("Visualization:")
    st.write("**Named Entities:**")
    for word, entity in result:
        st.write(f"{word} → {entity}")
