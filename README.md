# 📰 Fake News Detector (Machine Learning Project)

This is a machine learning project that classifies news articles as **real** or **fake** using Logistic Regression and TF-IDF vectorization.

📁 Fake News Detector
├── 📁 templates
    ├── index.html
├── 📄 app.py                       
├── 📄 train_model.py              
├── 📄 fake_news_model.pkl         
├── 📄 vectorizer.pkl             
├── 📄 WELFake_Dataset.csv  

## 🚀 Features

- Preprocessing (stemming, cleaning)
- TF-IDF text vectorization
- Logistic Regression model
- Model accuracy: ~95%
- Confusion matrix & classification report
- Flask web app interface (optional)
- Model + Vectorizer saved using `pickle`

## 📁 Dataset

Dataset used: **WELFake_Dataset.csv**  
It contains labeled news articles with:
- `0 = Fake`
- `1 = Real`

> (Note: Dataset file not uploaded here due to size/privacy.)

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Flask (for web interface)
- Pickle

## 📊 Results

Model accuracy: **95%**

Classification report:

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Fake  | 0.96      | 0.94   | 0.95     |
| Real  | 0.94      | 0.96   | 0.95     |

## 🧪 How to Use

### 1. Train the Model

```bash
python train_model.py

