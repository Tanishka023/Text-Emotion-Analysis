import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Example Data: Replace with actual data
data = {
    'text': [
        "I am so happy today!, "I feel really sad", "This is terrifying", 
        "Such a joyful moment", "I am disgusted", "I'm feeling neutral", 
        "That was such a scary experience", "What a pleasant surprise"
    ],
    'emotion': ['happy', 'sad', 'fear', 'joy', 'disgust', 'neutral', 'fear', 'surprise']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the data into features and labels
X = df['text']
y = df['emotion']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline with a TfidfVectorizer and Logistic Regression model
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression())
])

# Train the model
pipe.fit(X_train, y_train)

# Save the trained model to a .pkl file
joblib.dump(pipe, 'model/emotion_model.pkl')

print("Model saved as model/emotion_model.pkl")

# Function to predict emotion for custom text input
def predict_emotion(text):
    # Load the trained model
    model = joblib.load('model/emotion_model.pkl')
    
    # Predict the emotion of the input text
    prediction = model.predict([text])
    
    # Return the predicted emotion
    return prediction[0]

# Test the prediction function with custom input
custom_text = input("Enter some text to predict its emotion: ")
predicted_emotion = predict_emotion(custom_text)

print(f"The predicted emotion for the text '{custom_text}' is: {predicted_emotion}")
