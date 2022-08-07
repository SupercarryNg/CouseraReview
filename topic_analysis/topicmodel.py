import os
import warnings
from collections import Counter

import gensim.models.ldamodel
import matplotlib.pyplot as plt
import numpy as np
import umap
from gensim import corpora
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

from autoencoder import autoencoder_reduce
from finch import FINCH

warnings.filterwarnings('ignore')


class TopicModel:
    def __init__(self, df, sentiment='all', method='TF-IDF',
                 gamma=15, cluster_method='KMeans', req_k=None):

        self.df = df
        self.sentiment = sentiment
        self.method = method
        self.gamma = gamma
        self.cluster_method = cluster_method
        self.req_k = req_k

        self.check_param()

        self.vec_dict = {'TF-IDF': None, 'LDA': None, 'BERT': None, 'LDA_BERT': None}
        self.label_dict = {'TF-IDF': None, 'LDA': None, 'BERT': None, 'LDA_BERT': None}
        self.texts = None
        self.common_texts = None

    def check_param(self):
        if set(self.df.columns) != {'Id', 'Review', 'Label', 'tokens', 'sentiment'}:
            raise Exception('Invalid dataframe attribute, check your dataframe')

        if self.sentiment not in ['positive', 'neutral', 'negative', 'all']:
            raise Exception('Invalid sentiment')

        if self.method not in ['TF-IDF', 'LDA', 'BERT', 'LDA_BERT']:
            raise Exception('Invalid vectorization method, please choose from TF-IDF, LDA, BERT, LDA_BERT.')

        if self.cluster_method not in ['FINCH', 'KMeans']:
            raise Exception('Invalid cluster method, please choose from FINCH, KMeans.')

        if self.gamma > 20:
            raise Warning('Gamma Hyper parameter is too large, use (0, 20) instead')

        if self.req_k >= self.df.shape[0]:
            raise Exception('Cluster number should be less than the number of example points')

    def load_data(self):
        print('Loading data...')
        if self.sentiment == 'all':
            self.df = self.df

        else:
            self.df = self.df.loc[self.df.sentiment == self.sentiment]
            self.df.reset_index(inplace=True)

        self.texts = self.df['tokens']
        self.common_texts = []
        for text in self.texts:
            tmp = text.split(' ')
            self.common_texts.append(list(filter(None, tmp)))

    def vectorization(self, method=None):
        if method is None:
            method = self.method
        print('Getting vector representation for {} method'.format(method))

        vec = None

        if method == 'TF-IDF':
            tfidf = TfidfVectorizer()
            vec = tfidf.fit_transform(self.texts)

        elif method == 'LDA':
            dictionary = corpora.Dictionary(self.common_texts)
            common_corpus = [dictionary.doc2bow(text) for text in self.common_texts]
            lda_model = gensim.models.ldamodel.LdaModel(common_corpus, num_topics=self.req_k,
                                                        alpha='auto', id2word=dictionary, passes=20)

            n_reviews = self.df.shape[0]
            vec = np.zeros((n_reviews, self.req_k))  # Initialize a matrix to store the topic weights for each review

            for i in range(n_reviews):
                for topic, prob in lda_model.get_document_topics(common_corpus[i]):
                    vec[i, topic] = prob

        elif method == 'BERT':
            model = SentenceTransformer('all-mpnet-base-v2')
            vec = np.array(model.encode(self.texts, show_progress_bar=True))
            self.vec_dict[method] = vec

        elif method == 'LDA_BERT':
            if self.vec_dict['LDA'] is None:
                self.vectorization(method='LDA')

            if self.vec_dict['BERT'] is None:
                self.vectorization(method='BERT')

            vec_lda = self.vec_dict['LDA']
            vec_bert = self.vec_dict['BERT']
            vec = np.c_[vec_lda * self.gamma, vec_bert]
            vec = autoencoder_reduce(vec)

        self.vec_dict[method] = vec
        print('Vectorization {} done.'.format(method))
        # (N, c) -> sentence

    def fit(self, method=None, cluster_method=None):
        if method is None:
            method = self.method

        if cluster_method is None:
            cluster_method = self.cluster_method

        # LDA method does not need clustering
        if method == 'LDA':
            return

        print('Using {} method for clustering...'.format(cluster_method))

        if cluster_method == 'KMeans':
            best_cluster = 10  # initialize best cluster number
            best_s_score = -1
            # if self.req_k:
            #     cluster_model = KMeans(self.req_k)
            #     cluster_model.fit(self.vec_dict[method])
            # else:
            for i in range(2, 20):
                cluster_model = KMeans(i)
                cluster_model.fit(self.vec_dict[method])
                tmp_res = cluster_model.labels_
                s_score = self.evaluation(res=tmp_res)
                if s_score > best_s_score:
                    best_cluster = i
                    best_s_score = s_score

            self.req_k = best_cluster
            print('Best number of topic clusters is {}, silhouette_score is {}'.format(best_cluster, best_s_score))
            cluster_model = KMeans(n_clusters=best_cluster)
            cluster_model.fit(self.vec_dict[method])

            lbs = cluster_model.labels_
            self.label_dict[method] = lbs

        elif cluster_method == 'FINCH':
            c, num_clust, req_c = FINCH(self.vec_dict[method], distance='euclidean', req_clust=None)
            # c -> shape (N, num of partition)
            best_silhouette_score = -1
            best_partition = 0
            for i in range(c.shape[1]):
                tmp_score = silhouette_score(self.vec_dict[method], c[:, i])
                if tmp_score > best_silhouette_score:
                    best_silhouette_score = tmp_score
                    best_partition = i

            lbs = c[:, best_partition]
            print('Best_partition: {}'.format(best_partition))
            self.label_dict[method] = lbs
            self.req_k = len(np.unique(lbs))

    def evaluation(self, method=None, res=None):
        if method is None:
            method = self.method

        vec = self.vec_dict[method]
        lbs = self.label_dict[method]

        if res is not None:
            s_score = silhouette_score(vec, res)
            return s_score  # This is for choosing the right number of topics

    @staticmethod
    def reduce(vec):
        print('Decreasing the data dimension for better visualization')
        reducer = umap.UMAP(min_dist=0.9, random_state=42, n_components=2, n_neighbors=100)
        embedding = reducer.fit_transform(vec)  # reduce the data dimension for visualization
        return embedding

    def visualization(self, method=None):
        if method is None:
            method = self.method
        if method == 'LDA':
            return

        vec = self.vec_dict[method]
        lbs = self.label_dict[method]

        embedding = self.reduce(vec)

        path = './Results/' + self.method + '/' + self.sentiment + '/' + self.cluster_method + '/'

        if not os.path.exists(path):
            os.makedirs(path)

        counter = Counter(lbs)
        n = embedding.shape[0]

        for i in range(len(np.unique(lbs))):
            plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', ms=1.5, alpha=0.5,
                     label='cluster {}: {:.2f}%'.format(i, counter[i] / n * 100))
        plt.legend()
        plt.savefig(path + 'Cluster_Plot')

    def wordcloudplot(self, topic, lbs):
        common_texts = self.common_texts
        tokens = ' '.join([' '.join(_) for _ in np.array(common_texts)[lbs == topic]])

        path = './Results/' + self.method + '/' + self.sentiment + '/' + self.cluster_method + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        wordcloud = WordCloud(width=800, height=560, background_color='white',
                              collocations=False, min_font_size=10).generate(tokens)

        # plot the WordCloud image
        plt.figure(figsize=(8, 5.6), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(path + str(topic))

    def get_wordcloud(self, method=None):

        #    Get word cloud of each topic from fitted model
        #    :param model: Topic_Model object
        #    :param sentences: preprocessed sentences from docs
        if method is None:
            method = self.method
        if self.method == 'LDA':
            return

        lbs = self.label_dict[method]
        for i in range(self.req_k):
            print('Getting wordcloud for topic {} ...'.format(i))
            self.wordcloudplot(i, lbs)
