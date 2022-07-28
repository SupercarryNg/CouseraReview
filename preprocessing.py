import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import *
import pandas as pd


class Preprocessing:

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df = self.get_sentiment()

    @staticmethod
    def rem_meaningless(tweet):
        # remove the urls
        tweet = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", tweet)
        # remove @ and references
        tweet = re.sub(r"(@[A-Za-z0-9_]+)", "", tweet)
        # remove special charaecters
        tweet = re.sub(r"[#:)\n/_\-&]", "", tweet)
        return tweet

    @staticmethod
    def stemming(filtered):
        # return each word's token
        stemmer = PorterStemmer()
        cleaned = [stemmer.stem(word) for word in filtered]
        return cleaned

    def get_sentiment(self):
        # label 1,2 --> negative; label 3 --> neutral; label 4,5 --> positive;
        self.df.dropna(axis = 0, inplace = True)
        self.df['sentiment'] = 0
        dic = {1:'negative',2:'negative',3:'neutral',4:'positive',5:'positive',}
        self.df['sentiment'] = self.df['Label'].map(lambda x:dic[x])
        return self.df
    
    def getsample(self, positveNum = 7000, data = None):
        posindex = data[(data['sentiment'] == 'positive')].index.tolist()
        neuindex = data[(data['sentiment'] == 'neutral')].index.tolist()
        negindex = data[(data['sentiment'] == 'negative')].index.tolist()
        posidx = np.random.choice(posindex,positveNum, replace = False).tolist()

        return self.df.iloc[posidx + negindex + neuindex]
    
    def cleaning(self, lowercase=True, remove_special=True, stemming=True, stop_words=None):
        '''
        :param lowercase:
        :param remove_special:
        :param stemming:
        :param stop_words:
        :return: cleaned tokenized data with sentiment values (dataframe)
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
    csv_file = 'reviews.csv'
    preprocess = preprocess_csv(csv_file)
    cleaneddata = preprocess.cleaning(stop_words='stopwords.txt')
    samples = preprocess.getsample(data=cleaneddata).reset_index()
    print(samples.head())






