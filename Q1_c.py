import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset and preprocess it
# Assuming you have 'text' and 'feeling' columns
dataset = load_dataset("carblacac/twitter-sentiment-analysis", split="train")
df = dataset.to_pandas()
texts = df['text'].values
labels = df['feeling'].values

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences and pad them
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    padded_sequences, encoded_labels, test_size=0.2, random_state=42
)

# Build the Bi-LSTM model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=padded_sequences.shape[1]))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
