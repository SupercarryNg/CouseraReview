from preprocessing import Preprocessing
from topicmodel import TopicModel


def main():
    csv_file = 'data/reviews.csv'
    preprocess = Preprocessing(csv_file)
    df = preprocess.cleaning(stop_words='stopwords.txt')
    model = TopicModel(df=df, method='LDA_BERT', req_k=8)
    model.load_data()
    model.vectorization()
    model.fit()
    model.visualization()
    model.get_wordcloud()


if __name__ == '__main__':
    main()
