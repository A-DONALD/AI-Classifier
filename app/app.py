import numpy as np
import re
import nltk
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
nltk.download('stopwords')
import pickle
import os

# Load files
data_dir = os.path.dirname(os.path.abspath(__file__))
sport_data = load_files(os.path.join(data_dir, "data_sets/sports"))
politics_data = load_files(os.path.join(data_dir, "data_sets/politics"))
economy_data = load_files(os.path.join(data_dir, "data_sets/economy"))

X_sport, y_sport = sport_data.data, np.zeros(len(sport_data.data))
X_politics, y_politics = politics_data.data, np.ones(len(politics_data.data))
X_economy, y_economy = economy_data.data, np.twos(len(economy_data.data))

X = np.concatenate((X_sport, X_politics, X_economy))
y = np.concatenate((y_sport, y_politics, y_economy))

# Text preprocessing
documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

# Convert word to number: type bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

# Find TFIDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Algorithm of random forest
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

if os.path.exists(os.path.join(data_dir, "text_classifier")):
    # Open the model
    with open('text_classifier', 'rb') as training_model:
        model = pickle.load(training_model)

    # Evaluating the model save
    y_pred2 = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred2))
    print(classification_report(y_test, y_pred2))
    print(accuracy_score(y_test, y_pred2))
    print("...")

else:
    # Evaluating the model

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    # Save the model
    with open('text_classifier', 'wb') as picklefile:
        pickle.dump(classifier, picklefile)

text = "Hello world!, I don't like you, you always talk a bullshit"
text = vectorizer.transform([text]).toarray()
text = tfidfconverter.transform(text).toarray()
label = classifier.predict(text)[0]
print(label)