import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ✅ Balanced & fixed dataset
data = {
    'text': [
        # Happy
        "I am so happy today!",
        "This made my day wonderful.",
        "I'm feeling really good and cheerful.",
        
        # Sad
        "I feel really sad.",
        "My heart is heavy with sorrow.",
        "I just want to cry all day.",
        
        # Fear
        "This is terrifying.",
        "I'm scared of what might happen.",
        "That was such a scary experience.",
        
        # Joy
        "Such a joyful moment.",
        "I can't stop smiling from excitement.",
        "Everything feels so light and joyful.",
        
        # Disgust
        "I am disgusted.",
        "That's absolutely revolting.",
        "This makes me sick to my stomach.",
        
        # Neutral
        "I'm feeling neutral.",
        "Nothing much to say today.",
        "It's just an average day.",
        
        # Surprise
        "What a pleasant surprise!",
        "I didn't see that coming at all.",
        "That shocked me completely!"
    ],
    'emotion': [
        'happy', 'happy', 'happy',
        'sad', 'sad', 'sad',
        'fear', 'fear', 'fear',
        'joy', 'joy', 'joy',
        'disgust', 'disgust', 'disgust',
        'neutral', 'neutral', 'neutral',
        'surprise', 'surprise', 'surprise'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Split into features and labels
X = df['text']
y = df['emotion']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create pipeline
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lr', LogisticRegression(max_iter=200))
])

# Train the model
pipe.fit(X_train, y_train)

# Save the model
joblib.dump(pipe, 'model/emotion_model.pkl')
print("✅ Model saved as model/emotion_model.pkl")

# -------------------------
# 🔍 Predict custom input
# -------------------------
def predict_emotion(text):
    model = joblib.load('model/emotion_model.pkl')
    prediction = model.predict([text])
    return prediction[0]

# Get input from user
custom_text = input("\nEnter some text to predict its emotion: ")
predicted_emotion = predict_emotion(custom_text)

print(f"\n🔮 The predicted emotion for the text '{custom_text}' is: {predicted_emotion}")
