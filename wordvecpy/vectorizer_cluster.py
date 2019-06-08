
#VectokenizedClustering IS NOT currently workable and not included when importing massvecpy.  Will include in __init__.py
#when complete.

class VectokenizerClustering:

    def __init__(self, vectokenized_corpus, n_clusters, clustering_method = None, verbose = False):

        self.corpus = vectokenized_corpus
        self.integer_embedding = self.corpus.integer_embedding
        self.vector_dictionary = self.corpus.vector_dictionary
        self.word_dictionary = self.corpus.ranked_word_list
        if not (self.integer_embedding and self.vector_dictionary and self.word_dictionary):
            raise ValueError('Vectokenized Corpus must have a vector dictionary, word dictionary, and integer embedding.')
        self.n_clusters = n_clusters
        self.clustering = clustering_method
        if self.clustering:
            from sklearn.cluster import KMeans
            self.KMeans = KMeans(n_clusters=self.n_clusters, verbose=verbose)
            self.clf = self.KMeans.fit(self.vector_dictionary)
            self.clusters = self.clf.labels_
            num_clusters = len(set(self.clf.labels_))
            self.cluster_vectors = {i: clf.cluster_centers_[i] for i in range(num_clusters)}

    def new_clustering_method(self, cluster_method):
        if self.clustering:
            if self.need_vector_dict:
                self.clf = cluster_method.fit(self.vector_dictionary)
                self.clusters = clf.labels_
                num_clusters = len(set(clf.labels_))
                self.cluster_vectors = {i: clf.cluster_centers_[i] for i in range(num_clusters)}
            else:
                raise ValueError('Must have trained vector dictionary.')

    def cluster_transformed_integer_embedding(self, tokenize_unknown=False):
        if self.integer_token_embedding:
            if tokenize_unknown:
                adj = 1
                ext_labels = np.concatenate([[-1, 0], self.clusters])
            else:
                adj = 0
                ext_labels = np.concatenate([0], self.clusters)
            for i, j in self.integer_embedding:
                self.integer_embedding[i, j] = ext_labels[self.integer_embedding[i, j] + adj]
        else:
            raise ValueError('Must have integer embedding.')

    def cluster_transformed_vector_dictionary(self):
        for i in range(len(self.ranked_word_list)):
            self.vector_dictionary[i] = self.cluster_vectors[self.clusters[i]]

    def multi_index(self, index_list, label):
        return [i for i in range(len(index_list)) if index_list[i]==label]
