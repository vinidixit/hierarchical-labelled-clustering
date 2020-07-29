import pickle
import pandas as pd
from _nlp_utils import lemmatization, lemmatization_ticket, remove_stopwords
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import nltk
import sys
import logging
import tempfile
import os

LOG_FORMAT = '%(asctime)s : %(name)s : %(message)s'
formatter = logging.Formatter(fmt=LOG_FORMAT)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

# configuring root logger
logger = logging.getLogger('FeaturesExtraction')
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


class FeaturesExtraction:

    _n_gram_map = {1 : 'Unigrams', 2: 'Bigrams', 3: 'Trigrams'}

    def __init__(self, max_phrase_len=2, min_count = 2, threshold = 10, lemmatization_type='default'):
        assert (type(max_phrase_len) == int and 0 < max_phrase_len <= 5)
        assert (type(min_count) == int and type(threshold) == int)

        self.max_phrase_len = max_phrase_len
        self.min_count = min_count
        self.threshold = threshold
        self.lemmatization_type = lemmatization_type


    def extract_features(self, sentences, column_name=None):
        if type(sentences) == str:
            if '.pkl' in sentences:
                sentences = pickle.load(open(sentences, 'rb'))
            else:
                print('Cannot process a single string for feature extraction.')
                return

        if column_name:
            sentences = sentences[column_name]

        if type(sentences) == list:
            sentence_df = pd.DataFrame()
            sentence_df['sentence'] = list(sentences)

        elif type(sentences) == pd.Series:
            sentence_df = pd.DataFrame(index=sentences.index)
            sentence_df['sentence'] = sentences

        else:
            print('Error: Unrecognized input type :', type(sentences))
            return

        # Get content words
        sentence_df['content_words'] = sentence_df['sentence'].apply(self._get_content_words)

        sentence_df['unigrams'] = sentence_df.content_words.apply(lambda x: x[:,0] if len(x)>0 else []) # [ words[:, 0] if len(words) > 0 else [] for words in sentence_df['content_words']]

        if self.max_phrase_len == 1:
            sentence_df['n_grams'] = sentence_df['unigrams']
            return sentence_df

        n_1_corpus = sentence_df.unigrams
        n_gram_models = []

        for n in range(2, self.max_phrase_len+1):
            # build model
            n_gram = Phrases(n_1_corpus, min_count=self.min_count, threshold=self.threshold)
            n_gram_models.append(Phraser(n_gram))

            # get corpus
            n_gram_corpus = n_gram[n_1_corpus]
            n_1_corpus = n_gram_corpus

        # get phrases from sentences
        n_phrased_sentences = []
        for sent in sentence_df.unigrams:
            n_1_phrased_sent = sent
            for p_model in n_gram_models:
                n_1_phrased_sent = p_model[n_1_phrased_sent]

            n_phrased_sentences.append(n_1_phrased_sent)

        sentence_df['n_grams'] = n_phrased_sentences
        sentence_df['isNoisy'] = sentence_df.n_grams.map(len) == 0

        return sentence_df

    def _get_content_words(self, sentence):
        if type(sentence) != str:
            print('Error in input sentence:', sentence)
            return []

        names = {}
        tokens = nltk.wordpunct_tokenize(sentence)

        if self.lemmatization_type == 'default':
            data_lemmatized = lemmatization(tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        else:
            data_lemmatized = lemmatization_ticket(tokens)

        content_words = remove_stopwords(data_lemmatized, names)
        return content_words


def fe_mlflowrun(text_file, column_name=None, lemmatization_type='default'):
    fe = FeaturesExtraction(max_phrase_len=3, lemmatization_type=lemmatization_type)
    fe_df = fe.extract_features(text_file, column_name)

    local_dir = tempfile.mkdtemp()
    fe_file = os.path.join(local_dir, 'feature_extracted_df.pkl')
    pickle.dump(fe_df, open(fe_file, 'wb'))

    # prepare metrics and artifacts maps
    params_map = {'total_documents': len(fe_df)}
    artifacts_map = {"feature-extracted-dir" : fe_file}
    metrics_map = {'noisy_docs': fe_df[fe_df.isNoisy].shape[0]}

    return params_map, metrics_map, artifacts_map, fe_df


