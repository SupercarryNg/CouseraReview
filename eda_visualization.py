import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_csv
from wordcloud import WordCloud


class edaAnalysis:

    def __init__(self, csv_file):
        self.preprocess = preprocess_csv(csv_file)
        self.data = self.preprocess.cleaning(stop_words='stopwords.txt')

        self.data['sentiment'] = self.data['sentiment'].map({'positive':1, 'neutral':2, 'negative':3})

    def categoryPlot(self, colName):
        if colName not in ['Label', 'sentiment']:
            raise Exception('Invalid column name')
        self.data[colName].value_counts().plot(kind="bar")
        plt.title("Value counts of the {} variable".format(colName))
        plt.xlabel("{} type".format(colName))
        plt.xticks(rotation=0)
        plt.ylabel("Count")
        plt.show()

    def wordcloudplot(self, colName):
        if colName not in ['Label', 'sentiment']:
            raise Exception('Invalid column name')
        for i in range(len(self.data[colName].value_counts().keys())):
            text = ' '.join(self.data.loc[self.data[colName] == i + 1]['tokens'])
            wordcloud = WordCloud().generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()

if __name__ == '__main__':
    csv_file = 'reviews.csv'
    eda = edaAnalysis(csv_file)
    col = 'sentiment'
    eda.wordcloudplot(colName=col)
