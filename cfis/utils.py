# +
from nltk.probability import FreqDist
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

def get_phrase_distribution(tok_sentences):
    unigram_distr, bigrams_distr, trigrams_distr = FreqDist(), FreqDist(), FreqDist()
    
    for sent in tok_sentences:
        for phrase in sent:
            if phrase:
                if phrase.count('_')==1:
                    bigrams_distr[phrase] += 1
                elif phrase.count('_')>1:
                    trigrams_distr[phrase] += 1
                else:
                    unigram_distr[phrase] += 1
                    
    return unigram_distr, bigrams_distr, trigrams_distr


def remove_rare_words(df, column, rare_words):
    filtered = [[word for word in sent if word not in rare_words] for sent in df[column]]
    df.loc[:,column] = filtered
    return pd.DataFrame(df[df[column].map(len)>0])


def get_tfidf_matrix(docs, min_df = 0.008, max_df = 0.8):
    def dummy_fun(doc):
        return doc

    tfidf_vectorizer = TfidfVectorizer(
        min_df = min_df,
        max_df = max_df,
        analyzer='word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None,
        ngram_range=(1,1))


    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    print('Vocab_size:', len(tfidf_vectorizer.vocabulary_))
    return tfidf_matrix, tfidf_vectorizer.get_feature_names()


def dim_reduce_svd(tfidf_matrix, n_components, normalized = False, verbose=False):
    svd = TruncatedSVD(n_components)
    if normalized:
        lsa = make_pipeline(svd, Normalizer(copy=False))
    else:
        lsa = svd
        
    X = lsa.fit_transform(tfidf_matrix)
    explained_variance = svd.explained_variance_ratio_.sum()
    
    if verbose:
        print(n_components,": Explained variance of the SVD step: {}%".format(
                int(explained_variance * 100)))

    return X, explained_variance, svd

## Some corrections in data
def remove_sw(terms):
    # remove extra stopwords
    stopwords = ['limited', 'consulting']
    new_terms = list(terms)
    for sw in stopwords:
        if sw in terms:
            new_terms.remove(sw)
    return tuple(new_terms)

def correction(terms):
    new_terms = list(terms)
    if 'funds' in terms:
        new_terms.remove('funds')
        new_terms.append('fund')
    return tuple(new_terms)

def get_tfidf_words(row, terms):
    return [terms[i] for i,val in enumerate(row) if val>0]
