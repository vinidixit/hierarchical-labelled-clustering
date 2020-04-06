import pickle
from nlp_utils import lemmatization, remove_stopwords, get_bigram_model, make_bigrams, get_trigram_model, make_trigrams
from utils import get_phrase_distribution, remove_rare_words

import numpy as np
import itertools
import nltk
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

class FeaturesExtraction():

    n_gram_map = {1 : 'Unigrams', 2: 'Bigrams', 3: 'Trigrams'}

    def __init__(self, sentences, max_phrase_len=2):
        self.sentences = sentences
        self.max_phrase_len = max_phrase_len

    
    def get_content_words(self, sentence):
        names = {} #self.extracted_df.Names.loc[email_id][0]
        tokens = nltk.wordpunct_tokenize(sentence)
        data_lemmatized = lemmatization(tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        content_words = remove_stopwords(data_lemmatized, names)
        return content_words
    
    
    def get_content_words_helper(self, split_input):
        return [self.get_content_words(expr) for expr in split_input]
    
    
    def parallelize(self, df, func):
        num_partitions = 20
        columns_data = list(zip(df.email_id, df.expression))
        splits = np.array_split(columns_data, num_partitions)
        with ProcessPoolExecutor() as executor:
            result = executor.map(func, splits)
            print(result)
        return list(itertools.chain(*result))

    def extract_features(self, unique_subjects_df):
        # Get content words
        unique_subjects_df.loc[:, 'content_words'] = [self.get_content_words(expr) for expr in unique_subjects_df.expression]#self.parallelize(unique_subjects_df, self.get_content_words_helper)
        unique_subjects_df = unique_subjects_df[unique_subjects_df.content_words.map(len) > 0]
        
        print('\n', '---'*6, ' Unigrams ', '---'*6)
        unigrams = unique_subjects_df.content_words.apply(lambda x: x[:, 0])
        unique_subjects_df.insert(len(unique_subjects_df.columns), 'unigrams', unigrams, True)
        print(unique_subjects_df.unigrams[:5])
        #unique_subjects_df = self.remove_insignificant_phrases(unique_subjects_df, 'unigrams').copy()

        print('\n', '---'*6, ' Bigrams ', '---'*6)
        bigrams_corpus, bigram_model= get_bigram_model(unique_subjects_df.unigrams)
        bigrams = make_bigrams(unique_subjects_df.unigrams, bigram_model)
        unique_subjects_df.insert(len(unique_subjects_df.columns), 'bigrams', bigrams, True)
    #    unique_subjects_df = self.remove_insignificant_phrases(unique_subjects_df, 'bigrams').copy()
        print(unique_subjects_df.bigrams[:5])

        print('\n', '---'*6, ' Trigrams ', '---'*6)
        min_cnt, thresh = 3, 10
        trigram_model = get_trigram_model(bigrams_corpus, min_cnt, thresh)
        #unique_subjects_df = self.remove_insignificant_phrases(unique_subjects_df, 'trigrams').copy()
        trigrams = make_trigrams(unique_subjects_df.unigrams, bigram_model, trigram_model)
        unique_subjects_df.insert(len(unique_subjects_df.columns), 'trigrams', trigrams, True)
        print(unique_subjects_df.trigrams[:5])

        return unique_subjects_df

if __name__=='__main__':

    unique_expr_file = '../data/unique_subjects.pkl'
    unique_subjects = pickle.load(open(unique_expr_file, 'rb'))
    unique_subjects_df = pd.DataFrame(unique_subjects, columns=['email_id', 'expression'])
    print(unique_subjects_df.head(),'\n')
    fe = FeaturesExtraction()
    extracted_df = fe.extract_features(unique_subjects_df)
    pickle.dump(extracted_df, open('../data/extracted_df.pkl', 'wb'))
    #process(extracted_df_file, unique_expr_file)
