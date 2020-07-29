from nltk.probability import FreqDist
import networkx as nx
import itertools
import logging
import sys


LOG_FORMAT = '%(asctime)s : %(name)s : %(message)s'
formatter = logging.Formatter(fmt=LOG_FORMAT)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger = logging.getLogger('CFIClustering')

#logger.addHandler(stream_handler)
#logger.setLevel(logging.INFO)

#logger = logging.getLogger(__name__)

## --------------------------------- Cluster Helper functions ------------------------------ ##

# %%
# ## Graph methods for Frequent Itemsets
def initialize_fi_graph(G, first_level_items):
    if not G:
        G = nx.DiGraph()
    G.add_node(('root',), freq=1)

    for item,freq in first_level_items:
        G.add_node(item, freq=freq)
        G.add_edge(('root',), item, freq=freq)

    return G

# %%
# Identify closed parent nodes for a node
def get_closed_parent_nodes(parent_cands, second_parents, G):
    closed_parents = set()

    for parent in parent_cands:
        # 'fund' -> (fund, update)
        intersection = set(G.successors(parent)).intersection(second_parents)
        if len(intersection) > 1:
            print('second_parents :', second_parents,'\n')
            logger.info('More than one intersect found..{} from: {} with:{}'.format( \
                intersection, parent, '  ', second_parents))

        if not intersection:
            closed_parents.add(parent)

    return list(closed_parents)

# %%
# Identify closed parents
def get_closed_parents(comb_freq, term1, term2, G):
    parent_term1, parent_term2 = sorted([term1, term2])
    parent_freq1 =  G.nodes[parent_term1]['freq'] if parent_term1 in G.nodes else 0
    parent_freq2 =  G.nodes[parent_term2]['freq'] if parent_term2 in G.nodes else 0

    disclosed_terms = set()
    term_parents1, term_parents2 = [parent_term1], [parent_term2]
    parents = [[parent_term1], [parent_term2]]

    if parent_freq1 == comb_freq:
        # go one level up (proven to be already closed)
        term_parents1 = list(G.predecessors(parent_term1))
        disclosed_terms.add(parent_term1)

    if parent_freq2 == comb_freq:
        # go one level up (proven to be already closed)
        term_parents2 = list(G.predecessors(parent_term2))
        disclosed_terms.add(parent_term2)

    if parent_freq1 == comb_freq:
        parents[0] = get_closed_parent_nodes(term_parents1, term_parents2, G)

    if parent_freq2 == comb_freq:
        parents[1] = get_closed_parent_nodes(term_parents2, term_parents1, G)

    parents = sorted(list(itertools.chain(*parents)))

    if ('root',) in parents:
        parents.remove(('root',))

    return sorted(parents), disclosed_terms

# %%
# ## Frequent Itemset generation
def gen_first_level_itemsets(term_docs):
    one_items = FreqDist()
    for doc in term_docs:
        for term in doc:
            one_items[(term,)] +=1

    return one_items

# %%
# Generate a valid comb
def get_comb(term1, term2, k):
    if term1 == term2:
        return
    if term1[:k]==term2[:k]:
        comb = tuple(sorted(set(term1+term2)))
        return comb


# %%
# Get support counts
def get_occurrence_counts(comb, term_docs):
    freq = 0
    for doc in term_docs:
        # check if comb exists in doc
        if set(comb).issubset(doc):
            freq += 1
    return freq

# %%
# Check if a label is valid candidate
def is_label_cand(doc, label):
    return set(label).issubset(doc) #not set(doc).isdisjoint(label)

# %%
# Identify singleton labels
def get_singletons(label_docs_map):
    return dict(filter(lambda elem: len(elem[1])==1, label_docs_map.items()))
