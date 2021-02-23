import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from re import sub
from string import punctuation
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class Preprocessor:
    
    def __init__(self, remove_markdown=True, remove_punctuation=True, tokenize=True,
                 lemmatize=True, make_bigrams=False, remove_stopwords=True, split=False):
        self.remove_markdown = remove_markdown
        self.remove_punctuation = remove_punctuation
        self.tokenize = tokenize
        self.lemmatize = lemmatize
        self.make_bigrams = make_bigrams
        self.remove_stopwords = remove_stopwords
        self.split = split
        
        self.punctuation_list = list(punctuation) + ['`', '’', '…']
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y=None, **fit_params):
        for x in X:
            if self.remove_markdown:
                x = self.transform_remove_markdown(x)
            if self.remove_punctuation:
                x = self.transform_remove_punctuation(x)
            if self.tokenize:
                x = self.transform_tokenize(x)
            if self.lemmatize:
                x = self.transform_lemmatize(x)
            if self.make_bigrams:
                x = self.transform_make_bigrams(x)
            if self.remove_stopwords:
                x = self.transform_remove_stopwords(x)
            if not self.split:
                x = self.transform_unsplit(x)
        return X
    
    def transform_remove_markdown(self, x):
        # remove markdown tags, only needed for Steam reviews
        # todo: check if tag is a link, and remove url in parenthesis
        # and keep the text in the brackets (without the brackets)
        # link format: [text to keep](www.urltoremove.com)
        return sub(r'\[.*?\]', '', x)
    
    def transform_remove_punctuation(self, x):
        # remove all punctuation, which is often freely not use on these user reviews
        punctuation_list = list(punctuation) + ['`', '’', '…']
        return x.translate(str.maketrans('', '', ''.join(punctuation_list)))
    
    def transform_tokenize(self, x):
        # tokenize words with only numbers and latin characters
        # also turns everything to lowercase
        # input is a single string, output is a list of strings
        tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
        return tokenizer.tokenize(x.lower())
    
    def transform_lemmatize(self, x):
        # expects list of strings as input
        lemmatizer = WordNetLemmatizer()
        return list(map(lemmatizer.lemmatize, x))
    
    def transform_make_bigrams(self, x):
        # expects list of strings as input
        # adds bigrams onto existing tokens
        grams = []
        for i in range(len(x)-(n-1)):
            gram = []
            for j in range(i, i+n):
                gram.append(x[j])
            grams.append(' '.join(gram))

        return x + grams
    
    def transform_remove_stopwords(self, x):
        # expects list of strings as input
        stopwords_list = stopwords.words('english') + self.punctuation_list
        return [word for word in x if word not in stopwords_list]
    
    def transform_unsplit(self, x):
        # recombines list of strings into single string
        # needed for TF-IDF vectorizer
        # not needed with doc2vec
        # do not use with make_bigrams
        return ' '.join(x)

class DenseTransformer:
    
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()