import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
mail_data = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# Removing unnecessary columns
columns_to_remove = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
mail_data = mail_data.drop(columns_to_remove, axis=1)

# Rename columns
mail_data.rename(columns={'v1': 'Category', 'v2': 'Message'}, inplace=True)

# spam=0, ham=1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separate the data (X) and the label (Y)
X = mail_data['Message']
Y = mail_data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Transform the text data to feature vectors that can be used as input to the Logistic Regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Fit and transform the training data
X_train_features = feature_extraction.fit_transform(X_train)

# Only transform the test data (do not fit, as we want to use the same rules as training)
X_test_features = feature_extraction.transform(X_test)

# Convert Y_train and Y_test values to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

model = LogisticRegression(class_weight='balanced', max_iter=1000)
# Train the model
model.fit(X_train_features, Y_train)

# Save the trained model
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(feature_extraction, vectorizer_file)

print("Model and vectorizer saved successfully!")

# Prediction on training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)

# Prediction on test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)

# Confusion matrix and classification report
print(confusion_matrix(Y_test, prediction_on_test_data))
print(classification_report(Y_test, prediction_on_test_data))