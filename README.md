# 📄 Resume Classification using Naive Bayes

This project demonstrates how to build a **resume classification system** using **Natural Language Processing (NLP)** and **Machine Learning**. The notebook processes resume text, extracts meaningful features using text vectorization techniques, and trains a **Multinomial Naive Bayes** classifier to automatically categorize resumes into different job domains.

The project provides a practical introduction to text preprocessing, feature extraction, and document classification using Python.

---

## 📌 Features

- Resume text preprocessing
- Natural Language Processing (NLP)
- Stopword removal
- Tokenization
- Lemmatization and stemming
- Word cloud visualization
- Text vectorization using CountVectorizer
- Label encoding
- Handling class imbalance with SMOTE
- Multinomial Naive Bayes classifier
- Model evaluation

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- Gensim
- WordCloud
- imbalanced-learn (SMOTE)
- Jupyter Notebook

---

## 📚 Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
gensim
wordcloud
imbalanced-learn
jupyterthemes
```

---

## 📊 Dataset

The project uses a **Resume Dataset** containing resumes from multiple professional fields.

Each resume belongs to a predefined category, allowing the model to learn how to classify unseen resumes based on their textual content.

Example categories may include:

- Data Science
- Web Development
- Java Developer
- Python Developer
- HR
- Testing
- Business Analyst
- DevOps
- Network Security
- Sales
- Mechanical Engineer
- Civil Engineer

---

## 🔍 Exploratory Data Analysis

The notebook performs several exploratory analysis tasks, including:

- Dataset inspection
- Class distribution analysis
- Resume length analysis
- Word frequency analysis
- Word cloud generation
- Visualization of category distribution

These analyses provide insights into the textual characteristics of the dataset.

---

## 📝 Natural Language Processing Pipeline

The notebook applies several NLP preprocessing techniques before training the model:

- Convert text to lowercase
- Remove punctuation
- Remove special characters
- Remove stopwords
- Tokenization
- Lemmatization
- Stemming
- Text cleaning using Gensim
- Feature extraction using **CountVectorizer**

These steps transform raw resume text into numerical features suitable for machine learning.

---

## ⚙️ Machine Learning Workflow

The notebook follows these steps:

1. Load the resume dataset
2. Explore the data
3. Clean and preprocess resume text
4. Encode target labels using `LabelEncoder`
5. Convert text into numerical features using **CountVectorizer**
6. Split the dataset into training and testing sets
7. Balance the training data using **SMOTE**
8. Train a **Multinomial Naive Bayes** classifier
9. Generate predictions
10. Evaluate model performance

---

## 📈 Model Evaluation

The classifier is evaluated using standard classification metrics, including:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score

These metrics measure how effectively the model classifies resumes into their respective categories.

---

## 🚀 Getting Started

### Install dependencies

Install them manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk gensim wordcloud imbalanced-learn jupyterthemes
```

---

### Download NLTK Resources

Before running the notebook, download the required NLTK datasets:

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
```

---

## ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the Jupyter file.

Run the notebook cells sequentially to reproduce the complete NLP and machine learning workflow.

---

## 🎯 Learning Objectives

This project demonstrates how to:

- Perform text preprocessing using NLP techniques
- Clean and normalize textual data
- Generate word clouds
- Convert text into numerical vectors
- Handle imbalanced datasets using SMOTE
- Train a Multinomial Naive Bayes classifier
- Build a document classification model
- Evaluate multi-class classification performance

---

## 🔮 Future Improvements

- Compare CountVectorizer with TF-IDF Vectorization
- Evaluate additional classifiers such as Logistic Regression, Random Forest, SVM, and XGBoost
- Apply word embeddings using Word2Vec or GloVe
- Fine-tune a transformer model such as BERT for resume classification
- Develop a web application using Streamlit for real-time resume prediction
