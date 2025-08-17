import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from textblob import TextBlob
import pandas as pd
import re

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class NLTKTextAnalyzer:
    def __init__(self, dataframe, text_column, date_column=None):
        """
        dataframe: pandas DataFrame containing the text data
        text_column: column name containing text
        date_column: optional column for trend analysis over time
        """
        self.df = dataframe
        self.text_column = text_column
        self.date_column = date_column
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.preprocessed_texts = []

    def preprocess_text(self, text):
        """Tokenize, remove stopwords, lemmatize, and clean text"""
        # Lowercase and remove non-alphanumeric
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return tokens

    def preprocess_corpus(self):
        """Preprocess all texts"""
        self.preprocessed_texts = self.df[self.text_column].apply(self.preprocess_text)
        return self.preprocessed_texts

    def frequent_keywords(self, n=20):
        """Return top n most frequent keywords"""
        all_words = [word for tokens in self.preprocessed_texts for word in tokens]
        counter = Counter(all_words)
        return counter.most_common(n)

    def sentiment_analysis(self):
        """Calculate sentiment polarity for each text"""
        self.df['polarity'] = self.df[self.text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
        if self.date_column:
            sentiment_trend = self.df.groupby(self.date_column)['polarity'].mean()
            return sentiment_trend
        return self.df['polarity']

    def readability_metrics(self):
        """Calculate simple readability metrics (average sentence length, word length)"""
        metrics = []
        for text in self.df[self.text_column]:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            avg_sentence_len = len(words) / len(sentences) if sentences else 0
            avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
            metrics.append({'avg_sentence_len': avg_sentence_len, 'avg_word_len': avg_word_len})
        return pd.DataFrame(metrics)


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Example dataset
    data = {
        'date': ['2025-08-01', '2025-08-02', '2025-08-03'],
        'text': [
            "I love this product! It is amazing and works perfectly.",
            "Not satisfied. The product broke after one use.",
            "Average experience. Could be better but okay for the price."
        ]
    }
    df = pd.DataFrame(data)

    analyzer = NLTKTextAnalyzer(df, text_column='text', date_column='date')
    analyzer.preprocess_corpus()

    print("Frequent Keywords:")
    print(analyzer.frequent_keywords(10))

    print("\nSentiment Analysis:")
    print(analyzer.sentiment_analysis())

    print("\nReadability Metrics:")
    print(analyzer.readability_metrics())
