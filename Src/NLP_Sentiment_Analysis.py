import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def preprocess_text(text):
    """
    Preprocesses the input text by lowercasing, removing special characters,
    extra spaces, stopwords, and lemmatizing the words.

    Args:
    text (str): The text to preprocess.

    Returns:
    str: The preprocessed text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Replace multiple spaces with a single space
    text = re.sub(r'^b\s+', '', text)  # Remove leading 'b ' that sometimes appears in online text
    tokens = text.split()  # Tokenize the text into words
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize each word
    return ' '.join(tokens)

def categorize_sentiment(row):
    """
    Categorizes the sentiment of a review based on the review score.

    Args:
    row (Series): A pandas Series containing 'review_comment_message' and 'review_score'.

    Returns:
    str: The sentiment category ('No Comment', 'positive', 'neutral', 'negative').
    """
    if row['review_comment_message'] == 'No Comment':
        return 'No Comment'
    elif row['review_score'] > 3:
        return 'positive'
    elif row['review_score'] == 3:
        return 'neutral'
    else:
        return 'negative'

def train_model(X, y):
    """
    Trains a Random Forest Classifier model on the TF-IDF vectorized text data.

    Args:
    X (iterable): The text data to vectorize and train on.
    y (iterable): The target labels for the text data.

    Returns:
    tuple: The test labels and predicted labels from the trained model.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train the Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions on the test data
    predictions = model.predict(X_test_tfidf)
    
    # Return the test labels and predictions for evaluation
    return y_test, predictions

def evaluate_model(y_test, predictions):
    """
    Evaluates the performance of the trained model using classification report,
    accuracy score, and confusion matrix.

    Args:
    y_test (iterable): The true labels for the test data.
    predictions (iterable): The predicted labels by the model.

    Prints the evaluation metrics to the output.
    """
    print(classification_report(y_test, predictions))
    print('Accuracy:', accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
