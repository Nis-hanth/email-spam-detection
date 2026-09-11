# 📧 Email Spam Detection Using Machine Learning

A Machine Learning project that detects whether an email is **Spam** or **Not Spam** using **TF-IDF** and **Multinomial Naive Bayes**.

## 📌 Project Overview

Email spam is a common problem where unwanted or harmful emails are sent to users.

This project uses Natural Language Processing (NLP) and Machine Learning to classify email messages into two categories:

* **0 → Not Spam**
* **1 → Spam**

The model learns patterns from email text and predicts whether a new email is spam or not.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Multinomial Naive Bayes
* Matplotlib
* Jupyter Notebook
* Streamlit
* Pickle

## 📂 Dataset

The project uses an email dataset containing **3,000 email records**.

The dataset contains the following columns:

| Column    | Description            |
| --------- | ---------------------- |
| `from`    | Sender email address   |
| `to`      | Receiver email address |
| `subject` | Email subject          |
| `date`    | Email date             |
| `body`    | Email message          |
| `label`   | Spam or Not Spam       |

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Remove Duplicates
   ↓
Handle Missing Values
   ↓
Combine Subject + Body
   ↓
EDA
   ↓
Train-Test Split
   ↓
TF-IDF Vectorization
   ↓
Multinomial Naive Bayes
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Save Model
   ↓
Streamlit Application
```

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Checked dataset shape and columns.
2. Checked missing values.
3. Checked duplicate records.
4. Removed duplicate records.
5. Filled missing values in text columns.
6. Combined email subject and body into one text column.

## 🔤 TF-IDF Vectorization

TF-IDF converts email text into numerical values that can be understood by the Machine Learning model.

The vectorizer was configured with:

* Lowercase conversion
* English stop-word removal
* Maximum 5,000 features

The TF-IDF vectorizer was fitted only on the training data and then used to transform the test data.

## 🤖 Machine Learning Model

### Multinomial Naive Bayes

The project uses **Multinomial Naive Bayes**, which is commonly used for text classification problems.

The model is trained using the TF-IDF features.

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
```

## 📊 Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

These metrics help measure how well the model identifies spam and non-spam emails.

## 🌐 Streamlit Application

A Streamlit web application was created for testing new emails.

The user can:

1. Enter or paste an email.
2. Click **Check Email**.
3. The trained model predicts the result.
4. The application displays whether the email is **Spam** or **Not Spam**.

## 🖥️ Application Screenshot

![Email Spam Detection]("C:\Users\nisha\OneDrive\Pictures\Screenshots\email.png" "screenshots\email.png")

## 📁 Project Structure

```text
Email-Spam-Detection/
│
├── Email_spam_detection.ipynb
├── app.py
├── email_dataset.csv
├── email_spam_model.pkl
├── tfidf.pkl
├── README.md
│
└── screenshots/
    └── email.png
```

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Nis-hanth/email-spam-detection.git
```

### 2. Open the project folder

```bash
cd email-spam-detection
```

### 3. Install required libraries

```bash
pip install pandas numpy scikit-learn streamlit matplotlib
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Saved Model Files

The project contains two saved files:

* `email_spam_model.pkl` → Trained Multinomial Naive Bayes model
* `tfidf.pkl` → Trained TF-IDF vectorizer

These files are loaded by the Streamlit application to make predictions on new emails.

## 🎯 Project Objective

The main objective of this project is to build a simple and practical Machine Learning application that can automatically classify emails as spam or not spam using Natural Language Processing.

## 👨‍💻 Author

**Nishanth**

BCA – Bharatesh College of Computer Applications, Belagavi

## ⭐ Future Improvements

* Improve model accuracy with a larger dataset
* Try other Machine Learning algorithms
* Add email header analysis
* Add probability/confidence score
* Deploy the Streamlit application online
* Improve text preprocessing
* Add real-time email integration
