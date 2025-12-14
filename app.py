import pickle
import tkinter as tk
from tkinter import messagebox

# Load the saved model
with open('spam_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the saved TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

def check_spam():
    input_mail = text_input.get("1.0", "end-1c")
    
    if not input_mail.strip():
        messagebox.showwarning("Warning", "Please enter some text first!")
        return

    # Transform input and predict
    input_data_features = loaded_vectorizer.transform([input_mail])
    prediction = loaded_model.predict(input_data_features)

    # Update the result label
    if prediction[0] == 1:
        result_label.config(text="Result: HAM", fg="green")
    else:
        result_label.config(text="Result: SPAM", fg="red")

    messagebox.showinfo("Classification Result", f"The email is classified as: {result}")

# Create the main window
window = tk.Tk()
window.title("Email Spam Classifier")
window.geometry("500x400")

# Title Label
title_label = tk.Label(window, text="Email Spam Detector", font=("Helvetica", 18, "bold"))
title_label.pack(pady=10)

# Instruction Label
instr_label = tk.Label(window, text="Paste the email content below:", font=("Arial", 10))
instr_label.pack()

# Text Input Box (Height=10 lines)
text_input = tk.Text(window, height=10, width=50)
text_input.pack(pady=10)

# Check Button
check_button = tk.Button(window, text="Check Email", command=check_spam, font=("Arial", 12), bg="lightblue")
check_button.pack(pady=5)

# Result Label (Starts empty)
result_label = tk.Label(window, text="Result: ", font=("Arial", 14, "bold"))
result_label.pack(pady=20)

# Run the app loop
window.mainloop()