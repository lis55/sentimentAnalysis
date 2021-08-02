import re

import nltk
import pandas as pd

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()


def preprocess(review):    

    # remove <br /> occurences
    review = re.sub(r'<br />', ' ', review)
    
    # remove non-word characters such as punctuation, numbers etc
    review = re.sub(r'\W', ' ', review)

    # remove all single characters
    review = re.sub(r'\s+[a-zA-Z]\s+', ' ', review)

    # remove single characters from the start
    review = re.sub(r'\^[a-zA-Z]\s+', ' ', review)

    # substitute multiple spaces with single space
    review = re.sub(r'\s+', ' ', review, flags=re.I)

    # convert to lowercase
    review = review.lower()

    # split the document in whitespaces (--> List of words)
    review = review.split()

    # remove stopwords before lemmatizing
    review = [word for word in review if word not in stopwords]

    # lemmatize each word in the list
    review = [lemmatizer.lemmatize(word) for word in review]

    # reconstruct the document by joining the words using whitespace
    review = ' '.join(review)

    return review


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['review'] = train['review'].apply(preprocess)
test['review'] = test['review'].apply(preprocess)
