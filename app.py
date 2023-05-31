from urllib import request
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import model_selection
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load and preprocess the data
df = pd.read_csv('train.csv')

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)


def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


lemma = WordNetLemmatizer()


def cleanTweet(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = " ".join([lemma.lemmatize(word) for word in words if word not in (stop)])
    text = "".join(words)
    text = re.sub('[^a-z]', ' ', text)
    return text


df['cleaned_tweets'] = df['text'].apply(cleanTweet)

y = df.target
X = df.cleaned_tweets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=0)

# Train the models
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1, 2))
tfidf_train_2 = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_2 = tfidf_vectorizer.transform(X_test)

mnb_tf = MultinomialNB()
mnb_tf.fit(tfidf_train_2, y_train)

pass_tf = PassiveAggressiveClassifier()
pass_tf.fit(tfidf_train_2, y_train)

kfold = model_selection.KFold(n_splits=10)
scoring = 'accuracy'

acc_mnb2 = cross_val_score(estimator=mnb_tf, X=tfidf_train_2, y=y_train, cv=kfold, scoring=scoring)
acc_pass2 = cross_val_score(estimator=pass_tf, X=tfidf_train_2, y=y_train, cv=kfold, scoring=scoring)

pred_mnb2 = mnb_tf.predict(tfidf_test_2)
pred_pass2 = pass_tf.predict(tfidf_test_2)

CM_mnb = confusion_matrix(y_test, pred_mnb2)
CM_pass = confusion_matrix(y_test, pred_pass2)

TN_mnb = CM_mnb[0][0]
FN_mnb = CM_mnb[1][0]
TP_mnb = CM_mnb[1][1]
FP_mnb = CM_mnb[0][1]

TN_pass = CM_pass[0][0]
FN_pass = CM_pass[1][0]
TP_pass = CM_pass[1][1]
FP_pass = CM_pass[0][1]

specificity_mnb = TN_mnb / (TN_mnb + FP_mnb)
specificity_pass = TN_pass / (TN_pass + FP_pass)
acc_mnb = accuracy_score(y_test, pred_mnb2)
acc_pass = accuracy_score(y_test, pred_pass2)
prec_mnb = precision_score(y_test, pred_mnb2)
prec_pass = precision_score(y_test, pred_pass2)
rec_mnb = recall_score(y_test, pred_mnb2)
rec_pass = recall_score(y_test, pred_pass2)
f1_mnb = f1_score(y_test, pred_mnb2)
f1_pass = f1_score(y_test, pred_pass2)

model_results = pd.DataFrame([
    ['Multinommial Naive Bayes - TFIDF-Bigram', acc_mnb, prec_mnb, rec_mnb, specificity_mnb, f1_mnb],
    ['Passive Agressive - TFIDF-Bigram', acc_pass, prec_pass, rec_pass, specificity_pass, f1_pass]
], columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score'])

tfidf_vectorizer_3 = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1, 3))
tfidf_train_3 = tfidf_vectorizer_3.fit_transform(X_train)
tfidf_test_3 = tfidf_vectorizer_3.transform(X_test)

mnb_tf3 = MultinomialNB()
mnb_tf3.fit(tfidf_train_3, y_train)

pass_tf3 = PassiveAggressiveClassifier()
pass_tf3.fit(tfidf_train_3, y_train)

acc_mnb3 = cross_val_score(estimator=mnb_tf3, X=tfidf_train_3, y=y_train, cv=kfold, scoring=scoring)
acc_pass3 = cross_val_score(estimator=pass_tf3, X=tfidf_train_3, y=y_train, cv=kfold, scoring=scoring)

pred_mnb3 = mnb_tf3.predict(tfidf_test_3)
pred_pass3 = pass_tf3.predict(tfidf_test_3)

CM_mnb3 = confusion_matrix(y_test, pred_mnb3)
CM_pass3 = confusion_matrix(y_test, pred_pass3)

TN_mnb3 = CM_mnb3[0][0]
FN_mnb3 = CM_mnb3[1][0]
TP_mnb3 = CM_mnb3[1][1]
FP_mnb3 = CM_mnb3[0][1]

TN_pass3 = CM_pass3[0][0]
FN_pass3 = CM_pass3[1][0]
TP_pass3 = CM_pass3[1][1]
FP_pass3 = CM_pass3[0][1]

specificity_mnb3 = TN_mnb3 / (TN_mnb3 + FP_mnb3)
specificity_pass3 = TN_pass3 / (TN_pass3 + FP_pass3)
acc_mnb3 = accuracy_score(y_test, pred_mnb3)
acc_pass3 = accuracy_score(y_test, pred_pass3)
prec_mnb3 = precision_score(y_test, pred_mnb3)
prec_pass3 = precision_score(y_test, pred_pass3)
rec_mnb3 = recall_score(y_test, pred_mnb3)
rec_pass3 = recall_score(y_test, pred_pass3)
f1_mnb3 = f1_score(y_test, pred_mnb3)
f1_pass3 = f1_score(y_test, pred_pass3)

mod_results = pd.DataFrame([
    ['Multinomial Naive Bayes - TFIDF-Trigram', acc_mnb3, prec_mnb3, rec_mnb3, specificity_mnb3, f1_mnb3],
    ['Passive Agressive - TFIDF-Trigram', acc_pass3, prec_pass3, rec_pass3, specificity_pass3, f1_pass3]
], columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score'])

results = model_results.append(mod_results, ignore_index=True)


@app.route('/classify', methods=['POST'])
def classify_tweet():
    tweet = request.form.get('tweet')
    cleaned_tweet = cleanTweet(tweet)  # Clean the user-entered tweet
    tfidf_tweet = tfidf_vectorizer.transform([cleaned_tweet])  # Apply TF-IDF transformation
    prediction = mnb_tf.predict(tfidf_tweet)  # Classify the tweet using the trained model

    if prediction[0] == 0:
        result = "Normal tweet"
    else:
        result = "Disaster tweet"

    return render_template('index.html', result=result)


@app.route('/')
def home():
    return render_template('index.html', results=results)

@app.route('/predict', methods=['POST'])
def predict():
    tweet = request.form['tweet']  # Get the tweet entered by the user
    cleaned_tweet = cleanTweet(tweet)  # Clean the tweet using your existing cleanTweet function

    # Vectorize the cleaned tweet using the same TfidfVectorizer used for training
    tfidf_tweet = tfidf_vectorizer.transform([cleaned_tweet])

    # Make prediction using the trained models
    prediction_mnb = mnb_tf.predict(tfidf_tweet)
    prediction_pass = pass_tf.predict(tfidf_tweet)

    # Determine the final prediction based on model results
    if prediction_mnb == 1 and prediction_pass == 1:
        prediction = 'Disaster Tweet'
    else:
        prediction = 'Normal Tweet'

    return render_template('index.html', results=results, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
