import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import digits

# Function to preprocess the text, lowercasing, removing digits and stopwords, and lemmatizing
def text_preprocessing(raw_text):
    # Tokenize and lowercase text
    lower = CountVectorizer().build_tokenizer()(raw_text.lower())
    # Remove digits
    no_digits = [w for w in lower if not w.isdigit()]
    # Remove stopwords
    filtered = [w for w in no_digits if not w in stopwords.words('english')]
    # Lemmatize the text
    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(w) for w in filtered]
    return ' '.join(lemma)

# Function to tag the documents for training
def tag_documents(text):
    # Tag documents with index
    tagged_docs = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(text)]
    return tagged_docs


