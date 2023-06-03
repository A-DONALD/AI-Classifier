import pickle
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from sklearn.datasets import load_files
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
nltk.download('stopwords')

data_dir = os.path.dirname(os.path.abspath(__file__))
data = []

# Load files
for filename in os.listdir('../data_sets'):
    if filename.endswith('.sgm'):
        try:
            with open(os.path.join('../data_sets', filename), 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            continue

        soup = BeautifulSoup(content, 'html.parser')
        reuters = soup.findAll('reuters')

        for reuter in reuters:
            if reuter['topics'] == "YES" and reuter.topics.text != '' and reuter.body is not None:
                data.append({ 'content': reuter.body.text,'target': reuter.topics.d.text,'lewissplit': reuter['lewissplit']})

X, y = [item['content'] for item in data], [item['target'] for item in data]
# Text preprocessing
documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

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
X_train, X_test, y_train, y_test = [], [], [], []
for i, x in enumerate(X):
    if data[i]['lewissplit'].lower() == 'train':
        X_train.append(x)
        y_train.append(y[i])
    elif data[i]['lewissplit'].lower() == 'test':
        X_test.append(x)
        y_test.append(y[i])

# Algorithm of random forest
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

if os.path.exists(os.path.join(data_dir, 'model', "text_classifier.pkl")):
    # Open the model
    with open(os.path.join(data_dir, 'model', 'text_classifier.pkl'), 'rb') as training_model:
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
    with open(os.path.join(data_dir, 'model', 'text_classifier.pkl'), 'wb') as picklefile:
        pickle.dump(classifier, picklefile)


def classify_text():
    text = entry.get("1.0", tk.END)
    if len(text) == 0:
        messagebox.showwarning("Avertissement", "Veuillez saisir du texte.")
        return

    # Prétraitement du texte
    stemmer = WordNetLemmatizer()
    document = re.sub(r'\W', ' ', text)
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    document = re.sub(r'^b\s+', '', document)
    document = document.lower()
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    # Transformer le texte en features
    text_features = vectorizer.transform([document]).toarray()
    text_features = tfidfconverter.transform(text_features).toarray()

    # Prédire le label
    label = classifier.predict(text_features)[0]

    # Afficher le résultat
    messagebox.showinfo("Résultat", "Label prédit : {}".format(label))


# Initialiser les composants Tkinter
root = tk.Tk()
root.title("Classification de texte")
root.geometry("500x300")

model_label = ttk.Label(root, text="Select a model ...")
model_label.pack()
model_list = [filename for filename in os.listdir(data_dir + '\\model') if filename.endswith(".pkl")]

select_model = ttk.Combobox(root, values=model_list, width=40)
select_model.pack()

# Créer le composant d'entrée de texte
entry = tk.Text(root, height=10, width=50)
entry.pack(pady=20)

# Créer le bouton pour lancer la classification
button = ttk.Button(root, text="Classer le texte", command=classify_text)
button.pack()

# Lancer la boucle Tkinter
root.mainloop()