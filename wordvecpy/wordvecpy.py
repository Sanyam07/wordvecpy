class LoadVectorEmbeddedDoc:

    def __init__(self, matrix_name, file_dir='./', labels = True):

        from os import listdir
        import numpy as np
        self.np = np

        if labels:
            self.current_labels = np.array(1)
        if labels == False:
            self.current_labels = np.array(None)

        self.name = matrix_name + '__'

        self.file_dir = file_dir

        self.all_files = [f for f in listdir(file_dir)]

        self.slice_files = list(filter(lambda x: x[:len(matrix_name)] == matrix_name, self.all_files))

        self.all_indices_available = [int(x[len(self.name):-4]) for x in self.slice_files if x[len(self.name):-4].isdigit()]

        if 2*len(self.all_indices_available) != len(self.slice_files):
            raise ValueError('File missing label and/or corpus slice.')

        self.current_embedding = self.np.array(0)

    def load(self, index, save_internal=False, return_matrix=True, verbose=False):
        import pickle

        if index not in self.all_indices_available:
            raise ValueError('Indexed slice not found in current directory.')
        elif (save_internal == False) and (return_matrix == False):
            raise ValueError('Matrix needs to be returned or saved internally.')

        file = open(self.file_dir + self.name + str(index) + '.dat', "rb")

        if self.current_labels.any():
            label_file = open(self.file_dir + self.name + str(index) + '_l.dat', 'rb')
            labels = pickle.load(label_file)

        embedded_slice = pickle.load(file)

        if (save_internal == True):
            self.current_embedding = embedded_slice

            if verbose == True:
                print('{} successfully loaded.'.format(self.name + str(index) + '.dat'))

            if return_matrix == True:
                self.current_labels = labels
                return self.current_embedding, self.current_labels
        else:
            if verbose == True:
                print('{} successfully loaded.'.format(self.name + str(index) + '.dat'))
            return embedded_slice, labels


    def load_multiple(self, index_list, save_internal=True, return_matrix=True, verbose=False):
        concat_matrix, concat_labels = self.load(index_list[0], save_internal=False, verbose = True)

        for index in index_list[1:]:
            new_m, new_l =  self.load(index, save_internal=False, return_matrix=True, verbose=verbose)
            concat_matrix = self.np.concatenate([concat_matrix, new_m])
            concat_labels = self.np.concatenate([concat_labels, new_l])

        if (save_internal == True):
            self.current_embedding = concat_matrix
            self.current_labels = concat_labels
            if return_matrix == True:
                return self.current_embedding, self.current_labels
            else:
                return None
        elif (save_internal == False) and (return_matrix == True):
            return concat_matrix, concat_labels

    def stack_internal(self, index_to_add, return_matrix=False, verbose=False):

        new_m, new_l = self.load(index_to_add, save_internal = False, return_matrix = True, verbose = verbose)
        self.current_embedding = self.np.concatenate([self.current_embedding, new_m])
        self.current_labels = self.np.concatenate([self.current_labels, new_l])
        if verbose == True:
            print('Slice {} added to saved internal embedding'.format(index_to_add))
        if return_matrix == True:
            return self.current_embedding, self.current_labels

    def clear_memory(self):

        self.current_embedding = None
        self.current_embedding = self.np.array(0)

        return "Drop it like it's hot!"

class VectorEmbeddedDoc:
    def __init__(self, corpus, labels = None, name = None, num_of_slices = 1, file_dir='./', pymag_vectors = None, random_state = None):

        import sys
        import numpy as np
        self.np = np

        try:
            import tqdm
            self.tqdm = tqdm
        except:
            print('Install tqdm using command <!pip install tqdm> to utilize progress bars when converting words to vectors.')

        if name:
            self.name = name
        else:
            try:
               self.name = corpus.name
            except:
                self.name = 'unnamed'

        try:
            self.corpus = corpus.corpus
            self.vectors = corpus
        except:
            self.corpus = corpus
            if not pymag_vectors:
                self.vectors = pymag_vectors
            else:
                raise ValueError("pymag_vectors cannot be 'None' if corpus is not vectokenized.")

        self.split_factor = num_of_slices

        self.labels = np.asarray(labels)

        if self.labels.any():
            if len(self.corpus) != len(labels):
                raise ValueError('Dimensions are not equal for corpus and labels.')

        self.file_dir = file_dir

        self.split_factor = num_of_slices

        self.split_dist = int(len(self.corpus) / self.split_factor)

        self.shuffle_index = list(range(len(self.labels)))

        if random_state:
            np.random.seed(random_state)
            self.shuffle_index = np.random.choice(self.shuffle_index, len(self.labels), replace = False)

        self.corpus = [self.corpus[i] for i in self.shuffle_index]

        if self.labels.any():
            self.labels = np.asarray([self.labels[i] for i in self.shuffle_index])

        self.max_length = corpus.longest_sentence

        self.vector_dim = corpus.vector_dim

        self.embedded_matrix = np.zeros((self.split_dist, self.max_length, self.vector_dim))


        self.sliced_corpus = np.asarray([self.corpus[a * self.split_dist : (a + 1) * self.split_dist] for a in range(self.split_factor)])

        self.sliced_labels = np.asarray([self.labels[a * self.split_dist:(a + 1) * self.split_dist] for a in range(self.split_factor)])

        self.current_slice = 0

        self.bar = ('tqdm' in sys.modules)



    def fit(self, slice_index=None, return_matrix=True, restrict_search = False, verbose=False):

        '''
        DESCRIPTION:    This function will take a slice of a list of tokenized docs,
                        convert the words from tokens to vectors, save the generated embedding matrix to harddrive,
                        and delete the matrix from RAM.

        OUTPUTS:        Returns matrix embedding for this input slice if selected.  Loads transformed corpus slice
                        into self.embedded_matrix

        INPUT VARIABLES:

        slice_index:    (INT) choose specific slice index different from the current slice index to load

        input_slice:    (LIST) single slice (not necessarily indexed) from which to generate embedding.  If left blank,
                         will use the current slice index of object to pick the associated indexed slice.

        return_matrix:  (BOOL) choose whether to return the transformed corpus slice or just update internal embedding matrix

        verbose:        (BOOL) set to True to receive updates on processing
        '''
        self.clear_memory()

        if not slice_index:
            slice_index = self.current_slice


        input_slice = self.sliced_corpus[slice_index]

        slice_size = len(input_slice)

        if self.labels.any():
            self.labels = self.sliced_labels[self.current_slice]

        if (self.bar == True) and (verbose == True):
            for row_index in self.tqdm.trange(slice_size):
                for word_index in range(len(input_slice[row_index])):
                    self.embedded_matrix[row_index, word_index, :] = self.vectors.query(input_slice[row_index][word_index], restrict_search = restrict_search)
        else:
            for row_index in range(slice_size):
                for word_index in range(len(input_slice[row_index])):
                    self.embedded_matrix[row_index, word_index, :] = self.vectors.query(input_slice[row_index][word_index], restrict_search = restrict_search)

        if verbose:
            print("Slice {} successfully generated.".format(slice_index))

        if return_matrix:
            return self.embedded_matrix, self.labels





    def save(self, slice_name = None, verbose=False):
        '''
        DESCRIPTION:        Saves the current embedded matrix for current slice as a .dat file in the current
                            directory via pickle

        OUTPUTS:            Saves the current embedding matrix (self.embedded_matrix)
                            to file as matrix_name + '__' + current slice index + '.dat'

        INPUT VARIABLES:
            slice_name:    (STR) Filename to save current embedding matrix.  If NONE, file will
                            be saved in the filename format ./matrix_name + '__' + str(current_slice) + '.dat'

            verbose:       (BOOL) Set to True to receive updates on processing
        '''
        import pickle

        if (slice_name == None):
            slice_name = self.file_dir + self.name + '__' + str(self.current_slice) + '.dat'
            if self.labels.any():
                labels_name = self.file_dir + self.name + '__' + str(self.current_slice) + '_l.dat'
                pickle.dump(self.sliced_labels[self.current_slice], open(labels_name, 'wb'), protocol=4)
        else:
            slice_name = self.file_dir + slice_name + '.dat'

        pickle.dump(self.embedded_matrix, open(slice_name, 'wb'), protocol=4)

        if verbose == True:
            print("{} saved successfully.".format(self.name + '__' + str(self.current_slice)))
            if self.labels.any():
                print("{} labels saved successfully.".format(self.name + '__' + str(self.current_slice) + '_l'))


    def __update_slice(self, set_slice=None):
        '''
        DESCRIPTION:    This is an internal function to increment the slice index

        OUTPUTS:        Increments the current slice index by 1 or to specified slice

        INPUT VARIABLES:
            set_slice:  (INT) Enter the new slice index.  If none entered, increment current slice index by 1
        '''

        if set_slice == None:
            self.current_slice += 1
        else:
            self.current_slice = set_slice

        return ('Yes bro, the slice index has increased by one.')



    def clear_memory(self):
        '''
        DESCRIPTION:     Zeros out self.embedding_matrix
        '''

        self.embedded_matrix = None
        self.embedded_matrix = self.np.zeros((self.split_dist, self.max_length, self.vector_dim))

        return ("Faded.")

    def fit_and_save(self, indexed_slices, verbose=False):
        '''
        DESCRIPTION:        Transform sequential slices of corpus to word vector format and save transformed embedding matrix
                            to disk

        OUTPUTS:            Saves to disk the transformed embedding matrix for each slice of corpus listed on input. Final
                            embedding matrix saved to self.embedded_matrix is the word vector transformation of the last
                            slice given.

        INPUT VARIABLES:
            indexed_slices: (LIST/INT/None) Lists all slices to transform and save.  If variable left empty, transform and save
                            all slices of corpus.
        '''
        indexed_slices = self.np.asarray(indexed_slices)

        if not indexed_slices.any():
            if verbose == True:
                print("\nLet's go.  In and out, 20 minute adventure...\n")
            while (self.current_slice < self.split_factor):
                self.fit(return_matrix = False, verbose=verbose)
                self.save(verbose=verbose)
                self.__update_slice()
                self.clear_memory()
            if verbose == True:
                print("The entire dataset has been successfully split into {} parts and saved.".format(self.split_factor))
        else:

            if indexed_slices.size == 1:
                indexed_slices = self.np.asarray([indexed_slices])

            for index in indexed_slices:
                self.__update_slice(index)
                self.fit(slice_index=index, return_matrix = False, verbose=verbose)
                self.save(verbose=verbose)
                self.clear_memory()

class Vectokenizer:

    def __init__(self, corpus, vector_dict, test_corpus = None, max_words = None, max_sentence_length = None, tokenize_unknown = False, name  = None, verbose = False):

        import numpy as np
        self.np = np

        import sys

        import tqdm
        self.tqdm = tqdm

        if not name:
            self.name = name
        else:
            self.name = 'unnamed'

        self.fast_type = False
        self.bar = ('tqdm' in sys.modules)
        self.verbose = verbose
        self.vectors = vector_dict
        self.vector_dim = len(self.vectors.query('test'))
        self.tokenize_unknown = tokenize_unknown

        if test_corpus != None:
            try:
                self.test_corpus = self.test_corpus.transform()
            except:
                self.test_corpus = test_corpus
        else:
            self.test_corpus = None

        try:
            self.corpus = corpus.transform()
        except:
            self.corpus = corpus

        if not max_sentence_length:
            self.longest_sentence = np.max([len(x) for x in self.corpus])
        else:
            self.longest_sentence = max_sentence_length

        self.total_list = [x for sentence in self.corpus for x in sentence]

        self.freq_dict = None

        if not self.freq_dict:
            self.freq_dict = {}
            if self.verbose:
                print('Generating frequency dictionary (.freq_dict)...')
                for word in self.tqdm.tqdm(self.total_list):
                    self.freq_dict[word] = self.freq_dict.get(word, 0) + 1
            else:
                for word in self.total_list:
                    self.freq_dict[word] = self.freq_dict.get(word, 0) + 1

        if self.freq_dict:
            self.total_list = None

        self.ranked_word_list = np.concatenate([[' '], (sorted(self.freq_dict.keys(), key = self.freq_dict.__getitem__, reverse = True))])

        if not max_words:
            self.max_words = len(self.ranked_word_list)-1
        else:
            self.max_words = max_words

        self.lost_words = self.ranked_word_list[self.max_words+1:]
        self.ranked_word_list = self.ranked_word_list[:self.max_words+1]

        self.avg_missing_vector = np.zeros(self.vector_dim)

        if self.tokenize_unknown and len(self.lost_words) > 0:
            self.ranked_word_list = np.concatenate([self.ranked_word_list, ['UNKNOWN']])
            if verbose == True:
                print('Calculating vector average of discarded words...')
            self.avg_missing_vector = np.average(self.query(self.lost_words, verbose = verbose), axis = 0)

        for key in list(self.freq_dict):
            if key not in self.ranked_word_list: del self.freq_dict[key]

    def tokenized_word_index(self):
        return {word:self.integer_token_lookup(word) for word in self.ranked_word_list}

    def fit_vector_dict(self, verbose = None):
        if type(verbose)!= bool:
            verbose = self.verbose
        if verbose == True:
            print('Generating integer token to word vector dictionary...')
        vector_dict = self.query(self.ranked_word_list[1:], initial_zero = True, verbose = verbose)
        return vector_dict

    def fit_integer_embedding(self, verbose = None, pad_first = True):
        if verbose == None:
            verbose = self.verbose
        integer_embedding = self.np.zeros((len(self.corpus), self.longest_sentence), dtype=int)
        if verbose:
            print('Generating integer embedding (.integer_embedding)...')
            for index in self.tqdm.trange(len(self.corpus)):
                integer_embedding[index] = self.str_to_int_tokens(self.corpus[index], max_length=self.longest_sentence, pad_first = pad_first)
        else:
            for index in range(len(self.corpus)):
                integer_embedding[index] = self.str_to_int_tokens(self.corpus[index], max_length=self.longest_sentence, pad_first = pad_first)
        return integer_embedding

    def __query_str(self, word):
        if word == 'UNKNOWN':
            return self.avg_missing_vector
        else:
            return self.vectors.query(str(word))

    def query(self, word_list, restrict_search = False, initial_zero = False, verbose = False):
        if initial_zero == False:
            query_embedding = self.np.zeros((len(word_list), self.vector_dim))
            i_z = 0
        else:
            query_embedding = self.np.zeros((len(word_list)+1, self.vector_dim))
            i_z = 1
        if verbose == True and self.bar == True:
            for i in self.tqdm.trange(i_z, len(word_list)+i_z):
                if (restrict_search == True) and (word_list[i] in self.ranked_word_list):
                    query_embedding[i] = self.__query_str(word_list[i-i_z])
                elif (restrict_search == False):
                    query_embedding[i] = self.__query_str(word_list[i-i_z])
            return query_embedding
        else:
            for i in range(i_z, len(word_list)):
                if restrict_search and (word_list[i] in self.ranked_word_list):
                    query_embedding[i] = self.__query_str(word_list[i-i_z])
                elif not restrict_search:
                    query_embedding[i] = self.__query_str(word_list[i-i_z])
            return query_embedding

    def str_to_int_tokens(self, word_list, max_length = None, pad_first = True):
        if not max_length:
            max_length = len(word_list)
        if not pad_first:
            return self.np.concatenate([[self.integer_token_lookup(x) for x in word_list], self.np.zeros(max_length - (len(word_list)), dtype = int)])
        else:
            return self.np.concatenate([self.np.zeros(max_length - len(word_list), dtype = int), [self.integer_token_lookup(x) for x in word_list]])

    def integer_token_lookup(self, str_token):
        try:
            return self.ranked_word_list.tolist().index(str_token)
        except:
            if not self.tokenize_unknown:
                return 0
            else:
                return self.max_words + 2

    def to_keras(self, pad_first = True):
        if not self.test_corpus:
            return self.fit_integer_embedding(pad_first = pad_first), self.fit_vector_dict()
        else:
            return self.fit_integer_embedding(pad_first = pad_first), self.test_to_integer_embedding(pad_first = pad_first), self.fit_vector_dict()

    def test_to_integer_embedding(self, pad_first = True):
        if not self.test_corpus:
            raise ValueError('Vectokenizer has no test corpus input.')
        transformed_corpus = self.np.zeros((len(self.test_corpus), self.longest_sentence), dtype = int)
        for line in range(transformed_corpus.shape[0]):
            transformed_corpus[line] = self.str_to_int_tokens(self.test_corpus[line], self.longest_sentence, pad_first = pad_first)

        return transformed_corpus

class TextProcessor:

    def __init__(self, corpus, lemmatizer = 'default', stopwords = {}, punctuation = '', contractions = {}, substitutions = {}, combined_strings = False):
        ''''

        stopwords:      {SET} a simple set of words to remove

        punctuation:    (SET) a set of punctuation to remove

        contractions:   (DICT) a dictionary of contractions to make.  Example is {"can't": "can not", "n't": not, ...}

        substitution:   {DICT}  Like contractions but for grouping words together.  Example {'i': 'me', 'my': 'me', etc.}
        '''

        from nltk.corpus import wordnet

        self.corpus = corpus

        if lemmatizer == 'default':
            from nltk import WordNetLemmatizer
            self.lemmatizer = WordNetLemmatizer()
            self.default_lem = True
        elif not lemmatizer:
            self.default_lem = False
            self.lemmatizer = self.identity
        else:
            self.default_lem = False
            self.lemmatizer = lemmatizer

        self.substitutions = substitutions

        if not stopwords:
            self.stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about',
              'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be',
              'some', 'for', 'do', 'its', 'your', 'such', 'into', 'of', 'most', 'itself',
              'other', 'off', 'is', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
              'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through',
              'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our',
              'their', 'while', 'above', 'both', 'up', 'to', 'had', 'she',
              'when', 'at', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will',
              'on', 'does', 'yourselve', 'then', 'that', 'because', 'what', 'over', 'why', 'so',
              'can', 'did', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where',
              'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 'being',
              'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was',
              'here', 'than'}
        else:
            self.stopwords = stopwords

        if not contractions:
            self.contractions = {"can't": "can not", "won't": "will not", "n't": " not",
             "'ve": " have", "'m": " am", "'s": "", "'d": " had", '-': ' '}
        else:
            self.contractions = contractions

        if not punctuation:
            self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        else:
            self.punctuation = punctuation

        self.POS = [wordnet.ADJ, wordnet.VERB, wordnet.NOUN, wordnet.ADV]

    def identity(self, word):
        return word

    def lemmatize(self, word):

        if self.default_lem:
            im_just_guessing_here = [self.lemmatizer.lemmatize(word, x) for x in self.POS]

            lemmatized_word = max(set(im_just_guessing_here),
                                  key=im_just_guessing_here.count)
        else:
            lemmatized_word = self.lemmatizer(word)

        return lemmatized_word

    def transform(self, combined_strings = False):
        if not combined_strings:
            return [self.filter(x.split()) for x in self.corpus]
        else:
            return [' '.join(self.filter(x.split())) for x in self.corpus]

    def filter(self, input_string, return_list = True):
        from re import sub
        new_list = []

        if type(input_string) == str:
            input_string = input_string.lower()
            for key in self.contractions:
                input_string = input_string.replace(key, self.contractions[key])
            for key in self.substitutions:
                input_string = input_string.replace(key, self.substitutions[key])
            input_string = sub('[' + self.punctuation + ']', '', input_string)
            pro_text = input_string.split()

        elif type(input_string) == list:
            pro_text = []
            for s in input_string:
                s = s.lower()
                for key in self.contractions:
                    s = s.replace(key, self.contractions[key])
                for key in self.substitutions:
                    s = s.replace(key, self.substitutions[key])
                s = sub('[' + self.punctuation + ']', '', s)
                for i in s.split():
                    pro_text.append(i)

        for u in pro_text:
            if not (u in self.stopwords):
                new_list.append(self.lemmatize(u))

        if return_list:
            return new_list
        else:
            return ' '.join(new_list)

class FastVectokenizer:

    def __init__(self, corpus, vector_dict, test_corpus = None, max_words = None, max_sentence_length = None, tokenize_unknown = False, name = None):
        try:
            from keras.preprocessing.text import Tokenizer
            from keras.preprocessing.sequence import pad_sequences
        except:
            raise ValueError('FastVectokenizer requires Keras.  Please utilize the normal Vectokenizer class if Keras not installed.')

        import numpy as np
        self.np = np
        self.fast_type = True
        self.vectors = vector_dict
        self.vector_dim = len(self.vectors.query('test'))
        self.name = name
        self.pad_sequences = pad_sequences
        self.tokenize_unknown = tokenize_unknown

        if test_corpus != None:
            self.test_size = len(self.test_corpus)
            try:
                self.test_corpus = self.test_corpus.transform()
            except:
                self.test_corpus = test_corpus
        else:
            self.test_corpus = None
            self.test_size = 0

        try:
            self.corpus = corpus.transform()
        except:
            self.corpus = corpus

        if max_sentence_length:
            self.max_sentence_length = max_sentence_length
        else:
            self.max_sentence_length = np.max([len(x) for x in corpus])

        self.corpus_size = len(self.corpus)

        if self.tokenize_unknown:
            self.toke = Tokenizer(num_words = max_words, oov_token='UNKNOWN')
        else:
            self.toke = Tokenizer(num_words = max_words)

        self.toke.fit_on_texts(self.corpus)

        self.ranked_word_list = np.concatenate([[' '], (sorted(self.toke.word_index.keys(), key=self.toke.word_index.__getitem__, reverse=False))])

        if max_words:
            self.max_words = max_words
            if not self.tokenize_unknown:
                self.ranked_word_list = self.ranked_word_list[:self.max_words]
            else:
                self.ranked_word_list = self.ranked_word_list[:self.max_words+1]
        else:
            if self.tokenize_unknown == True:
                self.max_words = len(self.toke.word_index)-1
            else:
                self.max_words = len(self.toke.word_index)

        if self.tokenize_unknown:
            self.lost_words = set()
            for key in self.toke.word_index:
                if key not in self.ranked_word_list:
                    self.lost_words.add(key)
            self.oov_vec = np.average([self.query(word) for word in self.lost_words], axis = 0)



    def tokenized_word_index(self):
        return self.toke.word_index

    def fit_integer_embedding(self):
        seqs = self.toke.texts_to_sequences(self.corpus)
        return self.pad_sequences(seqs, maxlen = self.max_sentence_length)

    def test_to_integer_embedding(self):
        seqs = self.toke.text_to_sequences(self.test_corpus)
        return self.pad_sequences(seqs, maxlen = self.max_sentence_length)

    def fit_vector_dict(self):
        dict_size = len(self.ranked_word_list[1:self.max_words])
        vect_dict = self.np.zeros((dict_size+1, self.vector_dim))
        for i in range(1, dict_size):
            vect_dict[i] = self.query(self.ranked_word_list[i])
        return vect_dict

    def to_keras(self):
        if self.test_corpus != None:
            return self.fit_integer_embedding(), self.test_to_integer_embedding(), self.fit_vector_dict()
        else:
            return self.fit_integer_embedding(), self.fit_vector_dict()

    def query(self, word, restrict_search = False):
        if self.tokenize_unknown and (word == 'UNKNOWN'):
            return self.oov_vec
        elif restrict_search and (word in self.ranked_word_list[1:]):
            return self.vectors.query(word)
        elif not restrict_search:
            return self.vectors.query(word)
        else:
            return self.np.zeros(self.vector_dim)

class VectorDictionary:

    def __init__(self):

        self.vector_dict = {}
        self.vector_dims = 0
        self.size = 0

    def load_dict(self, file_dir, encoding = "utf8", verbose = False):
        if verbose:
            import tqdm

        file = open(file_dir, 'r+', encoding=encoding)
        lines = file.readlines()
        file.close()

        if verbose:
            for line in tqdm.tqdm(lines):
                components = line.split(' ')
                self.vector_dict[components[0]] = [float(x) for x in components[1:]]
        else:
            for line in lines:
                components = line.split(' ')
                self.vector_dict[components[0]] = [float(x) for x in components[1:]]

        self.size = len(self.vector_dict)
        self.vector_dims = len(self.vector_dict['test'])

    def query(self, word, return_empty = None):
        try:
            return self.vector_dict[word]
        except:
            if not return_empty:
                return [0.00]*self.vector_dims
            else:
                return return_empty