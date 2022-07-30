import re

import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


class Preprocessing:

    def __init__(self, csv_file, positive_num):
        self.df = pd.read_csv(csv_file)
        self.df = self.get_sentiment()

        self.get_sample(positive_num=positive_num)

    @staticmethod
    def rem_meaningless(tweet):
        tweet = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", tweet)
        tweet = re.sub(r"(@[A-Za-z0-9_]+)", "", tweet)
        tweet = re.sub(r"[#:)\n/_\-&]", "", tweet)
        return tweet

    @staticmethod
    def stemming(filtered):
        stemmer = PorterStemmer()
        cleaned = [stemmer.stem(word) for word in filtered]
        return cleaned

    def get_sentiment(self):
        self.df.dropna(axis=0, inplace=True)
        self.df['sentiment'] = 0
        # map func?
        dict_ = {1: 'negative', 2: 'negative', 3: 'neutral', 4: 'positive', 5: 'positive', }
        self.df['sentiment'] = self.df['Label'].map(lambda x: dict_[x])
        return self.df

    def get_sample(self, positive_num=7000):
        data = self.df.copy()
        posindex = data[(data['sentiment'] == 'positive')].index.tolist()
        neuindex = data[(data['sentiment'] == 'neutral')].index.tolist()
        negindex = data[(data['sentiment'] == 'negative')].index.tolist()
        posidx = np.random.choice(posindex, positive_num, replace=False).tolist()

        self.df = data.iloc[posidx + negindex + neuindex].sample(frac=1)

    def cleaning(self, lowercase=True, remove_special=True, stemming=True, stop_words=None):
        '''

        :param lowercase:
        :param remove_special:
        :param stemming:
        :param stop_words:
        :return:
        '''
        print('Preprocessing the data')
        if not stop_words:
            stop_words = set(stopwords.words('english'))
        else:
            tmp = pd.read_csv('stopwords.txt', header=None)
            stop_words = list(tmp[0])
        train_data = []
        self.df.dropna(axis=0, inplace=True)
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

        return self.df


if __name__ == '__main__':
    csv_file = 'data/reviews.csv'
    preprocess = Preprocessing(csv_file, positive_num=5000)
    df = preprocess.cleaning(stop_words='stopwords.txt')
    print(df.shape[0])
