# 📧 SMS Spam Detection with Machine Learning

A Machine Learning project that detects whether a text message or email is **Spam** or **Ham**. This project includes a model training script and a user-friendly Graphical User Interface (GUI) built with Tkinter.

## 🚀 Features
* **Machine Learning:** Uses **Logistic Regression** to classify text with ~96% accuracy.
* **Text Processing:** Implements TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features.
* **GUI Application:** A clean desktop interface to test custom messages instantly.
* **Reproducibility:** Scripts to train the model from scratch and save it for inference.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Scikit-Learn, Pandas, NumPy, Pickle
* **GUI:** Tkinter (Standard Python Interface)

## 📂 Project Structure
```text
SPAMCLASSIFIER/
├── .venv/                   # Virtual environment (ignored in git)
├── app.py                   # GUI Application for testing
├── main.py                  # Script to train and evaluate the model
├── spam.csv                 # Dataset file
├── README.md                # Project documentation
└── .gitignore               # Files to ignore (pkl, venv, etc.)
