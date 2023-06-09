import pickle
import os
import tkinter as tk
import re
import colorama
import webbrowser
import nltk
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from tkinter import messagebox
from tkinter import ttk

nltk.download('wordnet')
nltk.download('stopwords')

class MainApp:
    def __init__(self):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataSets_dir = os.path.join(os.path.dirname(self.data_dir), 'data_sets')
        self.model_dir = os.path.join(self.data_dir, 'model')
        self.data = []
        self.documents = []

        self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []

        self.RandomForest_classifier = ''
        self.KNeighbors_classifier = ''
        self.multinomial_classifier = ''
        self.vectorizer = None
        self.tfidfconverter = None

        self.load_data()
        self.preprocess_data()
        self.extract_features()
        self.split_data()
        self.load_or_create_models()

        self.create_gui()

    def load_data(self):
        print("Loading files ", end=" ")
        for filename in os.listdir(self.dataSets_dir):
            if filename.endswith('.sgm'):
                try:
                    with open(os.path.join(self.dataSets_dir, filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(colorama.Back.GREEN + " " + colorama.Style.RESET_ALL, end=" ")
                except UnicodeDecodeError:
                    print(colorama.Back.RED + " " + colorama.Style.RESET_ALL, end=" ")
                    continue

                print(colorama.Back.GREEN + " " + colorama.Style.RESET_ALL, end=" ")
                soup = BeautifulSoup(content, 'html.parser')
                reuters = soup.findAll('reuters')

                for reuter in reuters:
                    if reuter['topics'] == "YES" and reuter.topics.text != '' and reuter.body is not None:
                        self.data.append({'content': reuter.body.text, 'target': reuter.topics.d.text, 'lewissplit': reuter['lewissplit']})

        self.X, self.y = [item['content'] for item in self.data], [item['target'] for item in self.data]

    def preprocess_data(self):
        from nltk.stem import WordNetLemmatizer
        stemmer = WordNetLemmatizer()

        for sen in range(0, len(self.X)):
            document = re.sub(r'\W', ' ', str(self.X[sen]))
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
            document = document.lower()
            document = document.split()
            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
            self.documents.append(document)

    def extract_features(self):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer

        self.vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        self.X = self.vectorizer.fit_transform(self.documents).toarray()

        self.tfidfconverter = TfidfTransformer()
        self.X = self.tfidfconverter.fit_transform(self.X).toarray()

    def split_data(self):
        for i, x in enumerate(self.X):
            if self.data[i]['lewissplit'].lower() == 'train':
                self.X_train.append(x)
                self.y_train.append(self.y[i])
            elif self.data[i]['lewissplit'].lower() == 'test':
                self.X_test.append(x)
                self.y_test.append(self.y[i])

    def load_or_create_models(self):
        print(" ")
        if not (os.path.exists(self.model_dir)):
            os.mkdir(self.model_dir)
        if os.path.exists('app/model/RandomForest_classifier.pkl'):
            self.RandomForest_classifier = pickle.load(open(os.path.join(self.model_dir, 'RandomForest_classifier.pkl'), 'rb'))
        else:
            self.RandomForest_classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
            self.RandomForest_classifier.fit(self.X_train, self.y_train)
            pickle.dump(self.RandomForest_classifier, open(os.path.join(self.model_dir, 'RandomForest_classifier.pkl'), 'wb'))
        print("Load Random Forest:", end=' ')
        print(round(accuracy_score(self.y_test, self.RandomForest_classifier.predict(self.X_test)), 3))

        if os.path.exists('app/model/KNeighbors_classifier.pkl'):
            self.KNeighbors_classifier = pickle.load(open(os.path.join(self.model_dir, 'KNeighbors_classifier.pkl'), 'rb'))
        else:
            self.KNeighbors_classifier = KNeighborsClassifier(n_neighbors=5)
            self.KNeighbors_classifier.fit(self.X_train, self.y_train)
            pickle.dump(self.KNeighbors_classifier, open(os.path.join(self.model_dir, 'KNeighbors_classifier.pkl'), 'wb'))
        print("Load K-Nearest Neighbors:", end=' ')
        print(round(accuracy_score(self.y_test, self.KNeighbors_classifier.predict(self.X_test)), 3))

        if os.path.exists('app/model/multinomial_classifier.pkl'):
            self.multinomial_classifier = pickle.load(open(os.path.join(self.model_dir, 'multinomial_classifier.pkl'), 'rb'))
        else:
            self.multinomial_classifier = MultinomialNB()
            self.multinomial_classifier.fit(self.X_train, self.y_train)
            pickle.dump(self.multinomial_classifier, open(os.path.join(self.model_dir, 'multinomial_classifier.pkl'), 'wb'))
        print("Load Multinomial Naives Bayes:", end=' ')
        print(round(accuracy_score(self.y_test, self.multinomial_classifier.predict(self.X_test)), 3))


    def classify_text(self):
        text = self.text_entry.get("1.0", 'end-1c')
        preprocessed_text = self.preprocess_text(text)
        features = self.vectorizer.transform([preprocessed_text]).toarray()
        selected_model = self.model_combo.get()

        if not selected_model or text.isspace():
            tk.messagebox.showwarning("Document Classification Tool",
                                      "Warning: Please select a model and enter some text")
            return

        if selected_model == "Multinomial Naive Bayes":
            predicted_label = self.multinomial_classifier.predict(features)
        elif selected_model == "K-Nearest Neighbors":
            predicted_label = self.KNeighbors_classifier.predict(features)
        elif selected_model == "Random Forest":
            predicted_label = self.RandomForest_classifier.predict(features)

        messagebox.showinfo("Prediction Result", f"The predicted label is: {predicted_label[0]}")

    def preprocess_text(self, text):
        from nltk.stem import WordNetLemmatizer

        stemmer = WordNetLemmatizer()

        document = re.sub(r'\W', ' ', text)
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)
        document = document.lower()
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        return document

    def open_github(self):
        webbrowser.open_new("https://github.com/A-DONALD/AI-Classifier")

    def create_gui(self):
        self.window = tk.Tk()
        self.window.title("Text Classification")
        self.window.geometry('500x350')

        self.model_label = tk.Label(self.window, text="Select Model:")
        self.model_label.pack()
        self.model_combo = ttk.Combobox(self.window, values=["Multinomial Naive Bayes", "K-Nearest Neighbors", "Random Forest"])
        self.model_combo.current(0)
        self.model_combo.pack()

        self.text_entry = tk.Text(self.window, height=10, width=50)
        self.text_entry.pack(pady=20)

        self.classify_label = tk.Label(self.window, text="Launch the classification")
        self.classify_label.pack()
        self.classify_button = tk.Button(self.window, text="Classify", command=self.classify_text)
        self.classify_button.pack()

        self.github_label = tk.Label(self.window, text="See the repository on Github")
        self.github_label.pack()
        self.github_button = tk.Button(self.window, text="GitHub", command=self.open_github)
        self.github_button.pack()

        self.window.mainloop()
