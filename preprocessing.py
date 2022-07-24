import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
import pandas as pd

class preprocess_csv:

    def __init__(self,csv_file):
        self.df = pd.read_csv(csv_file)
        self.df = self.getsentiment()

    def rem_meaningless(self, tweet):
        tweet = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", tweet)
        tweet = re.sub(r"(@[A-Za-z0-9_]+)", "", tweet)
        tweet = re.sub(r"[\#\:\)\n\/\_\-\&]", "", tweet)
        return tweet

    def stemming(self, filtered):
        stemmer = PorterStemmer()
        cleaned = [stemmer.stem(word) for word in filtered]
        return cleaned

    def getsentiment(self):
        self.df.dropna(axis = 0, inplace = True)
        self.df['sentiment'] = 0
        for i in range(len(self.df)):
            if self.df.loc[i,'Label'] in [1,2]:
                self.df.loc[i,'sentiment'] = 'negative'
            if self.df.loc[i, 'Label'] in [4, 5]:
                self.df.loc[i, 'sentiment'] = 'positive'
            else:
                self.df.loc[i, 'sentiment'] = 'neutral'
        return self.df



    def cleaning(self, lowercase = True, remove_special = True, stemming = True, stop_words = None):
        if not stop_words:
            stop_words = set(stopwords.words('english'))
        else:
            tmp = pd.read_csv('stopwords.txt', header = None)
            stop_words = list(tmp[0])
        train_data = []
        self.df.dropna(axis = 0, inplace = True)
        print('The sample size is:{}'.format(len(self.df)))
        for tweet in self.df['Review']:
            if lowercase:
                tweet = tweet.lower()
            else:
                continue
            if remove_special:
                tweet = self.rem_meaningless(tweet)
            else:
                continue
            tokens = word_tokenize(tweet)
            filtered = []
            for word in tokens:
                if word.isalpha() and word not in stop_words:
                    filtered.append(word)
            if stemming:
                filtered = self.stemming(filtered)
            else:
                continue
            train_data.append(' '.join(filtered))
        self.df['tokens'] = train_data

        return  self.df
if __name__ == '__main__':
    csv_file = 'reviews.csv'
    preprocess = preprocess_csv(csv_file)
    data = preprocess.cleaning(stop_words='stopwords.txt')
    print(data.head())







