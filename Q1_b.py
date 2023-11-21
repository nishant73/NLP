from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from datasets import load_dataset


# Load the dataset
dataset = load_dataset("carblacac/twitter-sentiment-analysis", split="train")
df = dataset.to_pandas()


# Use a subset for quick testing
df = df.sample(frac=0.1, random_state=42)


# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(df["text"], df["feeling"], test_size=0.2, random_state=42)


# Create a TfidfVectorizer with optimizations
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
train_data_vec = vectorizer.fit_transform(train_data)
test_data_vec = vectorizer.transform(test_data)


# Train an SVM classifier with parallel processing
model = SVC(kernel='linear', probability=True, random_state=42, verbose=True)  # Enable probability estimation for log_loss
model.fit(train_data_vec, train_labels)


# Make predictions on the test set
predictions = model.predict(test_data_vec)


# Print the classification report
print(metrics.classification_report(test_labels, predictions))


# Print the accuracy
accuracy = metrics.accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)
