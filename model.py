
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def print_nearest_neighbors(query_tf_idf, full_bill_dictionary, knn_model, k):
    """
    Inputs: a query tf_idf vector, the dictionary of bills, the knn model, and the number of neighbors
    Prints the k nearest neighbors
    """
    distances, indices = knn_model.kneighbors(query_tf_idf, n_neighbors=k + 1)
    nearest_neighbors = [full_bill_dictionary.keys()[x] for x in indices.flatten()]

    for bill in range(len(nearest_neighbors)):
        if bill == 0:
            print
            'Query Law: {0}\n'.format(nearest_neighbors[bill])
        else:
            print
            '{0}: {1}\n'.format(bill, nearest_neighbors[bill])


def create_vector(text):
    stemmer = PorterStemmer()
    def stem_words(words_list, stemmer):
        return [stemmer.stem(word) for word in words_list]

    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        stems = stem_words(tokens, stemmer)
        return stems

    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(text)
    return tfs


def knn(tfs):
    from sklearn.neighbors import NearestNeighbors
    model_tf_idf = NearestNeighbors(metric='cosine', algorithm='brute')
    model_tf_idf.fit(tfs)
    return model_tf_idf


def k_means(tfs, k):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=5,
                verbose=1)
    km.fit(tfs)
    return km
