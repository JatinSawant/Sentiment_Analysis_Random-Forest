import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Download NLTK resources (if not already downloaded)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load your trained model from the saved file (replace 'your_model_filename.pkl' with the actual filename)
import joblib
pipeline = joblib.load('D:\Mega_Project\Random_forest_train.pkl')

# Load your new unlabeled dataset from an Excel file
new_df = pd.read_excel('D:\Mega_Project\Temporary data.xlsx')

# Manually labeled subset with true emotions
manually_labeled_subset = pd.read_excel('D:\Mega_Project\Temporary data.xlsx')

# Text preprocessing for the new dataset
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

# Apply preprocessing to the 'text' column in the new dataset
new_df['Text'] = new_df['Text'].apply(preprocess_text)

# Make predictions using the trained model
new_predictions = pipeline.predict(new_df['Text'])

# Assuming 'emotion' is the target column in the new dataset
# If your target column has a different name, replace 'emotion' with the actual name
true_labels_subset = manually_labeled_subset['Emotion']

# Calculate accuracy for the new unlabeled dataset
accuracy_subset = accuracy_score(true_labels_subset, new_predictions)
print(f'Accuracy on the new unlabeled dataset: {accuracy_subset * 100:.2f}%')

# Add the predicted emotions to the new dataset
new_df['predicted_emotion'] = new_predictions

# Save the new dataset with predicted emotions to a CSV file
new_df.to_csv('Random_predicted_emotions.csv', index=False)

# Display the new dataset with predicted emotions
print(new_df[['Text', 'predicted_emotion']])

# Example: Predict the emotion for a new sentence
new_sentence = "s twilight descended over the tranquil hamlet of Meadowhaven, a nuanced atmosphere enveloped the landscape, weaving together the threads of both somber reflection and the delicate resilience of the human spirit. The sky, painted in hues of lavender and muted indigo, mirrored the complexities that danced in the hearts of its denizens. Silhouetted against the fading light, the village elders gathered in a quiet ritual, their eyes reflecting the weight of accumulated memories and the passage of time.Meanwhile, the younger generation, bathed in the soft glow of lanterns, engaged in a communal activity that held a subtle undercurrent of hope. Laughter, though not exuberant, echoed through the air, threading its way through the melancholic undertones of the evening. As they shared stories and aspirations, there lingered an unspoken acknowledgment of the transient nature of happiness, like fragile petals that sway in the gentle breeze, aware of the inevitable descent into the arms of twilight.In the interplay of shadows and muted laughter, Meadowhaven found itself suspended between the poignant acknowledgment of the past and the tentative embrace of the present. The evening held a delicate balance of emotions, a quiet symphony where the notes of sorrow and joy intertwined, creating a tapestry that mirrored the complex dance of life in the quiet corners of the village."

# Apply the same preprocessing to the new sentence
preprocessed_sentence = preprocess_text(new_sentence)

# Make predictions using the trained model
prediction = pipeline.predict([preprocessed_sentence])

# Display the predicted emotion
print(f'The predicted emotion for the sentence "{new_sentence}" is: {prediction[0]}')
