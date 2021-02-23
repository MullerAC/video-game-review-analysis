import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from re import sub
from string import punctuation
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
    
def remove_markdown(x):
    # remove markdown tags, only needed for Steam reviews
    # todo: check if tag is a link, and remove url in parenthesis
    # and keep the text in the brackets (without the brackets)
    # link format: [text to keep](www.urltoremove.com)
    return sub(r'\[.*?\]', '', x)
    
def remove_punctuation(x):
    # remove all punctuation, which is often freely not use on these user reviews
    punctuation_list = list(punctuation) + ['`', '’', '…', '\n']
    return x.translate(str.maketrans('', '', ''.join(punctuation_list)))
    
def tokenize(x):
    # tokenize words with only numbers and latin characters
    # also turns everything to lowercase
    # input is a single string, output is a list of strings
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    return tokenizer.tokenize(x.lower())
    
def lemmatize(x):
    # expects list of strings as input
    lemmatizer = WordNetLemmatizer()
    return list(map(lemmatizer.lemmatize, x))
    
def make_bigrams(x):
    # expects list of strings as input
    # adds bigrams onto existing tokens
    grams = []
    for i in range(len(x)-(n-1)):
        gram = []
        for j in range(i, i+n):
            gram.append(x[j])
        grams.append(' '.join(gram))

    return x + grams
    
def remove_stopwords(x):
    # expects list of strings as input
    stopwords_list = stopwords.words('english') + self.punctuation_list
    return [word for word in x if word not in stopwords_list]
    
def unsplit(x):
    # recombines list of strings into single string
    # needed for TF-IDF vectorizer
    # not needed with doc2vec
    # do not use with make_bigrams
    return ' '.join(x)