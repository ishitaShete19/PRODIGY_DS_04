import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the CSV file
df = pd.read_csv("twitter_sentiment_500.csv")

# Clean column names
df.columns = df.columns.str.strip()
print("✅ Cleaned Columns:", list(df.columns))

# Check for required columns
if 'Sentiment' not in df.columns or 'Tweet Content' not in df.columns:
    raise ValueError("Missing required columns: 'Sentiment' and 'Tweet Content'")

# Drop rows with missing data
df = df[['Tweet Content', 'Sentiment']].dropna()

# Normalize sentiment labels
df['Sentiment'] = df['Sentiment'].str.lower().str.strip()

# Split the data
X = df['Tweet Content']
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Plotting sentiment distribution
plt.figure(figsize=(8, 5))
df['Sentiment'].value_counts().plot(kind='bar', color=['red', 'green', 'orange'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
