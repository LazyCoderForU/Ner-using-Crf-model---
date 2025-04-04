import joblib
import pandas as pd
import nltk
import spacy
from spacy import displacy
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import (
    flat_f1_score, flat_classification_report,
    flat_precision_score, flat_recall_score, flat_accuracy_score,
    sequence_accuracy_score
)
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('ner_dataset.csv', encoding="ISO-8859-1")
df = df.fillna(method='ffill')

# Sentence class to group words and tags
class Sentence:
    def __init__(self, df):
        agg = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values, s['POS'].values, s['Tag'].values)]
        self.grouped = df.groupby("Sentence #").apply(agg)
        self.sentences = [s for s in self.grouped]

# Extract sentences
getter = Sentence(df)
sentences = getter.sentences

def word2features(sent, i):
    word, postag = sent[i][0], sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word.isalpha()': word.isalpha(),
        'word.isalnum()': word.isalnum(),
        'word.startswith.upper()': word[0].isupper(),
        'word.endswith.s': word.endswith('s'),
        'word.length': len(word)
    }
    if i > 0:
        word1, postag1 = sent[i-1][0], sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:word.isalpha()': word1.isalpha(),
            '-1:word.isalnum()': word1.isalnum(),
            '-1:word.startswith.upper()': word1[0].isupper(),
            '-1:word.endswith.s': word1.endswith('s'),
            '-1:word.length': len(word1)
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1, postag1 = sent[i+1][0], sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:word.isalpha()': word1.isalpha(),
            '+1:word.isalnum()': word1.isalnum(),
            '+1:word.startswith.upper()': word1[0].isupper(),
            '+1:word.endswith.s': word1.endswith('s'),
            '+1:word.length': len(word1)
        })
    else:
        features['EOS'] = True
    
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for _, _, label in sent]

def sent2tokens(sent):
    return [token for token, _, _ in sent]

# Feature extraction
X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train CRF model
crf = CRF(algorithm='l2sgd', c2=0.1, max_iterations=100, all_possible_transitions=False)
crf.fit(X_train, y_train)

# Evaluate model
y_pred = crf.predict(X_test)
print("F1 Score:", flat_f1_score(y_test, y_pred, average='weighted'))
print("Precision:", flat_precision_score(y_test, y_pred, average='weighted'))
print("Recall:", flat_recall_score(y_test, y_pred, average='weighted'))
print("Accuracy:", flat_accuracy_score(y_test, y_pred))
print("Sequence Accuracy:", sequence_accuracy_score(y_test, y_pred))
print(flat_classification_report(y_test, y_pred))

# Save model
joblib.dump(crf, 'crf_model.pkl')
print("Model saved successfully!")

# Test on a new sentence
sentence = "India is going to win the Apple stocks and can get a profit of 2 billion dollars in the next year 2020 with 2kg of apples"
nltk.download('averaged_perceptron_tagger_eng')
tokens = nltk.word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)

# Predict tags
ner_tags = crf.predict([sent2features(pos_tags)])[0]
nlp = spacy.load("en_core_web_sm")
doc = nlp(sentence)
for token, ner_tag in zip(doc, ner_tags):
    token.ent_type_ = ner_tag

displacy.render(doc, style="ent", jupyter=False)

# Dependency parsing
options = {"compact": True, "bg": "#09a3d5", "color": "white", "font": "Source Sans Pro", "fine_grained": True}
displacy.render(doc, style='dep', jupyter=False, options=options)

# Print token-wise NER predictions
for token, tag in zip(tokens, ner_tags):
    print(f"{token} -> {tag}")