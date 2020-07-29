import pickle
import pandas as pd
import _utils
from _utils import *

from concurrent.futures import ProcessPoolExecutor

import mlflow
import click
import tempfile
import os

# +
class FeaturesExtraction():

    def __init__(self, extracted_df):
        self.extracted_df = extracted_df
    
    
    def get_content_words(self, email_id, sent):
        names = self.extracted_df.Names.loc[email_id][0]
        tokens = nltk.wordpunct_tokenize(sent)
        data_lemmatized = lemmatization(tokens, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        content_words = remove_stopwords(data_lemmatized, names)
        return content_words
    
    
    def get_content_words_helper(self, split_input):
        return [self.get_content_words(int(email_id), expr) for email_id, expr in split_input]
    
    
    def parallelize(self, df, func):
        num_partitions = 20
        columns_data = list(zip(df.email_id, df.expression))
        splits = np.array_split(columns_data, num_partitions)
        with ProcessPoolExecutor() as executor:
            result = executor.map(func, splits)
        return list(itertools.chain(*result))
    
    def identify_insignificant_phrases(self, phrased_sents, verbose=True):
        unigram_distr,bigram_distr,trigram_distr = get_phrase_distribution(phrased_sents)
        if verbose:
            print('Phrase distr: unigrams#:%d bigrams#:%d trigrams#:%d'%
                  (len(unigram_distr), len(bigram_distr), len(trigram_distr)))

        # detect rare phrases
        rare_uni = [phrase for phrase in unigram_distr if unigram_distr[phrase]==1]
        rare_bi = [phrase for phrase in bigram_distr if bigram_distr[phrase]==1]
        rare_tri = [phrase for phrase in trigram_distr if trigram_distr[phrase]==1]

        return rare_uni, rare_bi, rare_tri    


    def remove_insignificant_phrases(self, df, column):
        rare_unigrams, rare_bigrams, rare_trigrams = self.identify_insignificant_phrases(df[column])
        rare_words = rare_unigrams + rare_bigrams + rare_trigrams
        return remove_rare_words(df, column, rare_words)

    def extract_features(self, unique_subjects_df):
        # Get content words
        unique_subjects_df.loc[:,'content_words'] = self.parallelize(unique_subjects_df, self.get_content_words_helper)
        unique_subjects_df = unique_subjects_df[unique_subjects_df.content_words.map(len)>0]
        print('Available significant expressions:', len(unique_subjects_df))

        print('\n', '---'*6, ' Unigrams ', '---'*6)
        # Get unigrams from content words
        unique_subjects_df.loc[:,'unigrams'] = unique_subjects_df.content_words.apply(lambda x:x[:,0])
        unique_subjects_df = self.remove_insignificant_phrases(unique_subjects_df, 'unigrams').copy()


        print('\n', '---'*6, ' Bigrams ', '---'*6)
        bigrams_corpus, bigram_model= get_bigram_model(unique_subjects_df.unigrams)
        unique_subjects_df.loc[:,'bigrams'] = make_bigrams(unique_subjects_df.unigrams, bigram_model)
        unique_subjects_df = self.remove_insignificant_phrases(unique_subjects_df, 'bigrams').copy()


        print('\n', '---'*6, ' Trigrams ', '---'*6)
        min_cnt, thresh = 3, 10
        trigram_model = get_trigram_model(bigrams_corpus, min_cnt, thresh)
        unique_subjects_df.loc[:,'trigrams'] = make_trigrams(unique_subjects_df.unigrams, bigram_model, trigram_model)
        unique_subjects_df = self.remove_insignificant_phrases(unique_subjects_df, 'trigrams').copy()

        return unique_subjects_df

    # Log extracted features
    def log_results(self, extracted_features):
        with mlflow.start_run() as mlrun:
            local_dir = tempfile.mkdtemp()
            extracted_features_pkl = os.path.join(local_dir, 'extracted_features_df.pkl')

            pickle.dump(extracted_features, open(extracted_features_pkl, 'wb'))
            print('Going to log at..', extracted_features_pkl)

            mlflow.log_artifact(extracted_features_pkl, "extracted_features")
            mlflow.log_param('bigram_min_count', 2)
            mlflow.log_param('bigram_thresh', 10)
            mlflow.log_param('trigram_min_count', 3)
            mlflow.log_param('trigram_thresh', 10)
            
            unigram_distr,bigram_distr,trigram_distr = get_phrase_distribution(extracted_features.trigrams)
            mlflow.log_metric('unigrams_count', len(unigram_distr))
            mlflow.log_metric('bigrams_count', len(bigram_distr))
            mlflow.log_metric('trigrams_count', len(trigram_distr))

            
#@click.command(help="Provide extracted email file and unique expressions.")
#@click.option("--extracted_df_file")
#@click.option("--unique_expr_file")
def process(extracted_df_file, unique_expr_file):
    extracted_df = pickle.load(open(extracted_df_file, 'rb'))
    
    unique_subjects = pickle.load(open(unique_expr_file, 'rb'))
    unique_subjects_df = pd.DataFrame(unique_subjects, columns=['email_id', 'expression'])
    
    unique_subjects_df['email_id'] = pd.to_numeric(unique_subjects_df['email_id'])
    unique_subjects_df = unique_subjects_df.sort_values('email_id')
    print('input loaded.')
    
    fe = FeaturesExtraction(extracted_df)
    extracted_features_df = fe.extract_features(unique_subjects_df)
    fe.log_results(extracted_features_df)
    
if __name__=='__main__':
    artifacts_file = '/workspace/Chatbot/expression-intents-analysis/mlruns/0/aa9744a4285c486eb82f9201209ff1ce/artifacts/'
    extracted_df_file = artifacts_file + 'extracted_mail_df/extracted_mail.pkl'
    unique_expr_file = artifacts_file + 'unique_expressions/unique_expressions.pkl'
    process(extracted_df_file, unique_expr_file)
