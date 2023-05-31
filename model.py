import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import nltk, re, string
from string import punctuation
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pickle

df = pd.read_csv('train.csv')
df.head()

stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

#creating a list of possible stopwords

stop = stopwords.words('english')

def cleanTweet(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    #removing stop words & lemmatizing the words
    words = " ".join([lemma.lemmatize(word) for word in words if word not in (stop)])
    text = "".join(words)
    #removing non-alphabetic characters
    text = re.sub('[^a-z]', ' ', text)
    return text


df['cleaned_tweets'] = df['text'].apply(cleanTweet)
df.head()

y = df.target
X = df.cleaned_tweets


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=0)


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,2))
tfidf_train_2 = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_2 = tfidf_vectorizer.transform(X_test)

mnb_tf = MultinomialNB()
mnb_tf.fit(tfidf_train_2, y_train)


from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10)
scoring  = 'accuracy'

acc_mnb2 = cross_val_score(estimator = mnb_tf, X = tfidf_train_2, y = y_train, cv = kfold, scoring=scoring)
acc_mnb2.mean()


pred_mnb2 = mnb_tf.predict(tfidf_test_2)
CM = confusion_matrix(y_test, pred_mnb2)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

specificity = TN/(TN+FP)
acc = accuracy_score(y_test, pred_mnb2)
prec = precision_score(y_test, pred_mnb2)
rec = recall_score(y_test, pred_mnb2)
f1 = f1_score(y_test, pred_mnb2)

model_results = pd.DataFrame([['Multinommial Naive Bayes - TFIDF-Bigram', acc, prec, rec, specificity, f1]],
                             columns = ['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score'])
print(model_results)




pass_tf = PassiveAggressiveClassifier()
pass_tf.fit(tfidf_train_2, y_train)



kfold = model_selection.KFold(n_splits=10)
scoring = 'accuracy'
acc_pass2 = cross_val_score(estimator = pass_tf, X = tfidf_train_2, y = y_train, cv = kfold, scoring=scoring)
acc_pass2.mean()


pred_pass2 = pass_tf.predict(tfidf_test_2)
CM = confusion_matrix(y_test, pred_pass2)
acc = accuracy_score(y_test, pred_pass2)
prec = precision_score(y_test, pred_pass2)
rec = recall_score(y_test, pred_pass2)
f1 = f1_score(y_test, pred_pass2)

results = pd.DataFrame([['Passive Agressive - TFIDF-Bigram', acc, prec, rec, specificity, f1]],
                         columns = ['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score'])
results = model_results.append(results, ignore_index = True)
results
#
#
tfidf_vectorizer_3 = TfidfVectorizer(stop_words = 'english', max_df=0.8, ngram_range=(1,3))
tfidf_train_3 = tfidf_vectorizer_3.fit_transform(X_train)
tfidf_test_3 = tfidf_vectorizer_3.transform(X_test)


mnb_tf3 = MultinomialNB()
mnb_tf3.fit(tfidf_train_3, y_train)


kfold = model_selection.KFold(n_splits=10)
scoring = 'accuracy'
acc_mnb3 = cross_val_score(estimator=mnb_tf, X = tfidf_train_3, y=y_train, cv=kfold, scoring=scoring)
acc_mnb3.mean()


pred_mnb3 = mnb_tf3.predict(tfidf_test_3)
CM = confusion_matrix(y_test, pred_mnb3)

acc = accuracy_score(y_test, pred_mnb3)
prec = precision_score(y_test, pred_mnb3)
rec = recall_score(y_test, pred_mnb3)
f1 = f1_score(y_test, pred_mnb3)

mod_results = pd.DataFrame([['Multinomial Naive Bayes - TFIDF-Trigram', acc, prec, rec, specificity, f1]],
                           columns = ['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score'])
results = results.append(mod_results, ignore_index = True)
results


pass_tf3 = PassiveAggressiveClassifier()
pass_tf3.fit(tfidf_train_3, y_train)


kfold = model_selection.KFold(n_splits=10)
scoring = 'accuracy'
acc_pass3 = cross_val_score(estimator=pass_tf3, X = tfidf_train_3, y=y_train, cv=kfold, scoring=scoring)
acc_pass3.mean()


pred_pass3 = pass_tf3.predict(tfidf_test_3)
CM = confusion_matrix(y_test, pred_pass3)

acc = accuracy_score(y_test, pred_pass3)
prec = precision_score(y_test, pred_pass3)
rec = recall_score(y_test, pred_pass3)
f1 = f1_score(y_test, pred_pass3)

mod1_results = pd.DataFrame([['Passive Agressive - TFIDF-Trigram', acc, prec, rec, specificity, f1]],
                            columns = ['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1 Score'])
results = results.append(mod1_results, ignore_index = True)
results
#
#
def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names_out()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        pass
        # print(class_labels, coef, feat)
    print()
    for coef, feat in reversed(topn_class2):
        pass
        # print(class_labels[1], coef, feat)


most_informative_feature_for_binary_classification(tfidf_vectorizer_3, pass_tf3, n=10)
#
# sentences = [
#     "Just happened a terrible car crash",
#     "Heard about #earthquake is different cities, stay safe everyone.",
#     "I am ILL",
#     "@RosieGray Now in all sincerety do you think the UN would to Israel if there was a fraction of chance of being annihilated"
# ]
#
# tfidf_trigram = tfidf_vectorizer_3.transform(sentences)
#
# predictions = pass_tf3.predict(tfidf_trigram)
#
# for text, label in zip(sentences, predictions):
#     if label==1:
#         target = "Disaster Tweet"
#         print("text: ", text, "\nClass", target)
#         print()
#     else:
#         target = "Normal Tweet"
#         print("text: ", text,"\nClass", target)
#         print()
#
