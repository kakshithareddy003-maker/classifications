
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Get the folder where this Python file is saved
folder = os.path.dirname(os.path.abspath(__file__))

# Load the datasets
train_path = os.path.join(
    folder, "Bitext_Sample_Customer_Service_Training_Dataset.csv"
)
test_path = os.path.join(
    folder, "Bitext_Sample_Customer_Service_Testing_Dataset.csv"
)

try:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("Training dataset loaded successfully!")
    print("Testing dataset loaded successfully!")

except FileNotFoundError as e:
    print("ERROR: Dataset file not found!")
    print("Please place both CSV files in this folder:")
    print(folder)
    print("\nMissing file:", e.filename)
    raise SystemExit

# Check required columns
required_columns = ["utterance", "intent"]

if not all(col in train.columns for col in required_columns):
    print("ERROR: Training CSV must contain 'utterance' and 'intent' columns.")
    raise SystemExit

if not all(col in test.columns for col in required_columns):
    print("ERROR: Testing CSV must contain 'utterance' and 'intent' columns.")
    raise SystemExit

# Remove rows without an intent label
train = train.dropna(subset=["intent"])
test = test.dropna(subset=["intent"])

# Prepare input and output data
X_train = train["utterance"].fillna("").astype(str)
y_train = train["intent"].astype(str)

X_test = test["utterance"].fillna("").astype(str)
y_test = test["intent"].astype(str)

# Create the classification model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LinearSVC())
])

# Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

print("Model training completed!")

# Evaluate the model
predictions = model.predict(X_test)

print("\nTest Accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nCustomer Support Intent Classification Model is Ready!")

# Predict the intent of user input
while True:
    sentence = input(
        "\nEnter a customer-support sentence (or type exit): "
    ).strip()

    if sentence.lower() == "exit":
        print("Program ended.")
        break

    if not sentence:
        print("Please enter a sentence.")
        continue

    prediction = model.predict([sentence])[0]

    print("Predicted Intent:", prediction)
