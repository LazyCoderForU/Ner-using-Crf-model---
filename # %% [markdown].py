# %% [markdown]
# # Developing CRF model for Named Entity Recognition 
# 
# * geo = Geographical Entity
# * org = Organization
# * per = Person
# * gpe = Geopolitical Entity
# * tim = Time indicator
# * art = Artifact
# * eve = Event
# * nat = Natural Phenomenon

# %% [markdown]
# #### Importing Libraries

# %%

from sklearn_crfsuite.metrics import flat_f1_score, flat_classification_report, flat_precision_score, flat_recall_score, flat_accuracy_score, sequence_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
import pandas as pd

# %%
#Reading the csv file
df = pd.read_csv('ner_dataset.csv', encoding = "ISO-8859-1")

# %%
df[df['Tag'] != 'O'].head(20)

# %%
df.describe()

# %%
#Displaying the unique Tags
df['Tag'].unique()

# %%
df

# %%
#Checking null values, if any.
df.isnull().sum()

# %% [markdown]
# There are lots of missing values in 'Sentence #' attribute. So we will use pandas fillna technique and use 'ffill' method which propagates last valid observation forward to next.

# %%
df = df.fillna(method = 'ffill')

# %% [markdown]
# ## Converting df to tuple to give input to --> crf model  

# %%
# This is a class te get sentence. The each sentence will be list of tuples with its tag and pos.
class sentence(object):
    def __init__(self, df):
        self.n_sent = 1
        self.df = df
        self.empty = False
        agg = lambda s : [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),s['POS'].values.tolist(),s['Tag'].values.tolist())]
        self.grouped = self.df.groupby("Sentence #").apply(agg)
        self.sentences = [s for s in self.grouped]
        
    def get_text(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent +=1
            return s
        except:
            return None

# %%
#Displaying one full sentence
#getter = sentence(df)
sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
sentences[0]

# %%
#sentence with its pos and tag.
sent = getter.get_text()
print(sent)
sentences = getter.sentences

# %%
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

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
        'word.length': len(word),
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
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
            '-1:word.length': len(word1),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
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
            '+1:word.length': len(word1),
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


# %%
X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# %%

crf = CRF(algorithm = 'l2sgd',
         c2 = 0.1,
         max_iterations = 100,
         all_possible_transitions = False)
crf.fit(X_train, y_train)

# %%
#Predicting on the test set.
y_pred = crf.predict(X_test)

# %% [markdown]
# #### Evaluating the model performance.
# 

# %%
f1_score = flat_f1_score(y_test, y_pred, average = 'weighted')
print(f1_score)

# %%
flat_f1_score(y_test, y_pred, average = 'weighted')

# %%
flat_precision_score(y_test, y_pred, average = 'weighted')

# %%
sequence_accuracy_score(y_test, y_pred)

# %%
flat_recall_score(y_test, y_pred, average = 'weighted')

# %%
flat_accuracy_score(y_test, y_pred)

# %%
report = flat_classification_report(y_test, y_pred)
print(report)

# %% [markdown]
# ## change the input to any sentence, for which you want NER

# %%

sentence = "India is going to win the Apple stocks and can get a profit of 2 billion dollars in the next year 2020 with 2kg of apples"  


# %%
import nltk
nltk.download('averaged_perceptron_tagger_eng')
tokens = nltk.word_tokenize(sentence)
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)


crf.predict([sent2features(pos_tags)])

# %%
import spacy
from spacy import displacy

ner_tags = crf.predict([sent2features(pos_tags)])[0]

nlp = spacy.load("en_core_web_sm")

doc = nlp(sentence)
for token, ner_tag in zip(doc, ner_tags):
    token.ent_type_ = ner_tag

displacy.render(doc, style="ent", jupyter=True)

# %%
options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro", "fine_grained": True}
displacy.render(doc, style='dep', jupyter=True, options=options)

# %%
for a,b in zip(tokens, crf.predict([sent2features(pos_tags)])[0]):
    print(f"{a} -> {b}") 


