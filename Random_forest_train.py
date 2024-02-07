import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import make_pipeline

# Download NLTK resources (if not already downloaded)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load your dataset from an Excel file (replace 'your_dataset.xlsx' with the actual file path)
df = pd.read_csv('D:\\Mega_Project\\tweet_emotions\\tweet_emotions.csv')

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatization
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Apply preprocessing to the 'Text' column
df['content'] = df['content'].apply(preprocess_text)

# Assuming 'Emotion' is the target column
# If your target column has a different name, replace 'Emotion' with the actual name
X = df['content']
y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a pipeline with TF-IDF vectorizer and Random Forest classifier
pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier(random_state=42))

# Train the model on the training set
pipeline.fit(X_train, y_train)

# Make predictions on the test set
predictions = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display classification report
print('Classification Report:\n', classification_report(y_test, predictions, zero_division=1))  # Set zero_division=1 to handle the warning

# Example: Predict the emotion for a new sentence
new_sentence = "In the quiet aftermath, shattered fragments of trust lingered, leaving behind a silence heavy with the weight of unspoken grievances."

# Apply the same preprocessing to the new sentence
preprocessed_sentence = preprocess_text(new_sentence)

# Make predictions using the trained model
prediction = pipeline.predict([preprocessed_sentence])

# Display the predicted emotion
print(f'The predicted emotion for the sentence "{new_sentence}" is: {prediction[0]}')

# Save the trained model to a file
joblib.dump(pipeline, 'Random_forest_train.pkl')
