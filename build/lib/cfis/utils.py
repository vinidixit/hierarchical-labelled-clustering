# +
import nltk
import spacy
from nltk.corpus import words, stopwords
import itertools
import numpy as np

from nltk.probability import FreqDist
import gensim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

# -

nltk.download('stopwords')
nltk.download('words')

nlp = spacy.load('en', disable=['parser'])


english_words = set(words.words('en'))
english_words = set([word.lower() for word in english_words])
english_words.remove('sap')

basic_stop_words = set(stopwords.words('english'))

sw_light = set(['date','january','february','march','april','may','june','july','august','september','october',
                   'november','december','am','pm','ist','pst','cst', 'let', 'us', 'know', 'me', 
           'monday','tuesday','wednesday','thursday','friday','saturday'])

sw_light.update(set([chr(ch) for ch in range(ord('a'), ord('z')+1)]))

basic_stop_words.update(sw_light)


sw_hard = set(['from', 'subject', 're', 'edu', 'use', 'co', 'email', 'would', 'could', 'reminder', 'urgent', 'gmbh'])
sw_hard.update(set(['hi', 'hello', 'good' 'afternoon', 'still', 'send', 'show', 'ticket', 'mob', 'upload', 'attachment']))
sw_hard.update(set(['fyi', 'dear', 'cid', 'image', 'png', 'jpeg', 'jpg', 'xlsx', 'internal', 'use', 'only', 'limited']))
sw_hard.update(set('http www com url mail id tel fax site web please thank thanks regards best regard kindly kind ltd'.split()))


extra_words = set()
extra_words.update(['from', 'subject', 're', 'edu', 'use', 'co', 'email', 'would', 'could', 'gmt', 'information', 'require'])
extra_words.update(['hi', 'hello', 'good' 'afternoon', 'still', 'send', 'show', 'ticket', 'mob', 'inc', 'hola', 'action'])
extra_words.update(['jan','feb','mar','apr','jun','aug', 'sep', 'oct', 'nov', 'dec', 'logos', 'campaignimage', 'header'])
extra_words.update(['fyi', 'dear', 'ticket', 'number', 'mobile', 'mailto', 'http', 'https', 'nbsp', 'jpg', 'jpeg'])
extra_words.update(['pcitc', 'facebook', 'linkedin','importance','high', 'contact', 'com', 'image', 'gif', 'png', 'http'])


advanced_stopwords = set()
advanced_stopwords.update(basic_stop_words)
advanced_stopwords.update(sw_hard)
advanced_stopwords.update(extra_words)


def remove_stopwords(sent_toks, names):
    local_sw = set(advanced_stopwords)
    local_sw.update(names)
    return np.array([(word,tag) for word,tag in sent_toks if len(word)>2 and word.lower() not in local_sw])


def get_named_entities(doc):
    entities = set(itertools.chain(*[ent.text.split() for ent in doc.ents if ent.label_ in {'ORG','LOC'}]))
    
    for token in doc:
        if token.text in entities and (token.text.lower() in english_words or token.lemma_.lower() in english_words):
            entities.remove(token.text)
    
    return entities

def lemmatization(sent, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(" ".join(sent)) 
        
    sw_entities = get_named_entities(doc)
    #print('entities:',sw_entities)
    
    lem_toks = np.array([(token.lemma_.lower(), token.pos_) 
                         for token in doc
                            if (token.pos_ in allowed_postags or
                               (token.pos_=='PROPN' and # for misidentified proper nouns
                                                     token.lemma_.lower() in english_words)) and
                               token.text not in sw_entities
                        ])
    return lem_toks


def get_bigram_model(tokenized_sents, min_cnt=2, threshold=10):
    bigram = gensim.models.Phrases(tokenized_sents, min_count=min_cnt, threshold=threshold) 
    bigram_model = gensim.models.phrases.Phraser(bigram)
    return bigram[tokenized_sents], bigram_model


def get_trigram_model(bigrams, min_cnt, threshold):
    trigram = gensim.models.Phrases(bigrams,min_count=min_cnt, threshold=threshold)  
    return gensim.models.phrases.Phraser(trigram)


# +
def make_bigrams(texts, bigram_model):
    return [bigram_model[doc] for doc in texts]

def make_trigrams(texts, bigram_model, trigram_model):
    return [trigram_model[bigram_model[doc]] for doc in texts]



# -

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
