import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv(r"C:\Users\Lenovo\Desktop\Fake News Detector\WELFake_Dataset.csv")

data

data = data.dropna()

data['content'] = data['title'] + " " + data['text']

data

# stemming
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

X = data['content'].values
y = data['label'].values

vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=1)

model = LogisticRegression()
model.fit(X_train,y_train)

train_y_pred = model.predict(X_train)
print("train accurracy :",accuracy_score(train_y_pred,y_train))

import pickle

# Save the trained model
with open('fake_news_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vector, vectorizer_file)

print("✅ Model and vectorizer have been saved successfully!")
