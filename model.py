
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

def create_combined_vector(tweets, wordVec_model):
    stemmer = PorterStemmer()
    def stem_words(words_list, stemmer):
        return [stemmer.stem(word) for word in words_list]

    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        stems = stem_words(tokens, stemmer)
        return stems

    dataset = np.ndarray(shape=(len(tweets), 300),
                             dtype=np.float32)

    # vector_rep = []
    i = 0 
    x = 0
    for each_tweet in tweets:
        print(each_tweet)

        tokens = tokenize(each_tweet)  
        tokens = list(filter(lambda x: x in wordVec_model.vocab, tokens))
        if len(tokens)<1:
            continue
        print('filtered', len(tokens)) 
        each_tweet = ' '.join(tokens) 
        tokens2 = tokenize(each_tweet)  
        print('combined', len(tokens2)) 


        tfidf = TfidfVectorizer(tokenizer=tokenize)
        tfs = tfidf.fit_transform([each_tweet])  
        word_vec = wordVec_model[tokens]
        print(type(word_vec))
        word_vec_avg = np.ndarray(shape=(1, 300 ),
                            dtype=np.float32)
    
        for each_vec in word_vec:
            word_vec_avg = word_vec_avg + each_vec
    
        word_vec_avg = word_vec_avg / len(tokens)
        

        print(word_vec_avg.T.shape, tfs.shape)
        vector = word_vec_avg.T * tfs
        print(vector.T.shape)
        transposed_vec =  word_vec_avg.T[:, 0]
        dataset[i, :] = transposed_vec
        i = i+1
        print(each_tweet, np.where(np.isnan(dataset)))

        # vector_rep.append(vector.T)
           
    return dataset


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

if __name__ == '__main__':
    text = 'teaser photo i'
    from word_vec import Word2Vec
    word = Word2Vec()
    arr = create_combined_vector([text],word.model)
    print(np.where(np.isnan(arr )))