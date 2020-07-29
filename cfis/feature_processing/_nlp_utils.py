import itertools
import nltk
import spacy
from nltk.corpus import words
from nltk.corpus import stopwords
import numpy as np

nltk.download('words')
nlp = spacy.load('en', disable=['parser'])

english_words = set(words.words('en'))
english_words = set([word.lower() for word in english_words])


nltk.download('stopwords')

# -
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
    entities = set(itertools.chain(*[ent.text.split() for ent in doc.ents if ent.label_ in {'ORG', 'LOC'}]))

    for token in doc:
        if token.text in entities and (token.text.lower() in english_words or token.lemma_.lower() in english_words):
            entities.remove(token.text)

    return entities


def lemmatization(sent, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(" ".join(sent))

    sw_entities = get_named_entities(doc)

    lem_toks = np.array([(token.lemma_.lower(), token.pos_)
                         for token in doc
                         if (token.pos_ in allowed_postags or
                             (token.pos_ == 'PROPN' and  # for misidentified proper nouns
                              token.lemma_.lower() in english_words)) and
                         token.text not in sw_entities
                         ])
    return lem_toks

def lemmatization_ticket(sent, allowed_postags=['PROPN','NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(" ".join(sent))

    lem_toks = np.array([(token.lemma_.lower(), token.pos_)
                         for token in doc
                         if token.pos_ in allowed_postags
                         ])
    return lem_toks
