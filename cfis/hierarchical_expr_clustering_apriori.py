# %%
from utils import get_tfidf_matrix, dim_reduce_svd
import collections
from nltk.probability import FreqDist
from utils import remove_sw, correction, get_tfidf_words
from cluster_utils import print_graph, print_clusters
# graph library
import networkx as nx
# Visualization libraries
#from pyvis.network import Network
from pylab import *

import logging
LOG_FORMAT = '%(asctime)s : %(name)s : %(message)s'
formatter = logging.Formatter(fmt=LOG_FORMAT)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
# configuring root logger
logger = logging.getLogger('CFIClustering')
logger.addHandler(stream_handler)
#     logger.handlers = [stream_handler]
logger.setLevel(logging.INFO)

class CFIClustering():
    NUMBER_MAPPINGS = {1: '1st', 2: '2nd', 3: '3rd'}

    def __init__(self, support_threshes={1: 30, 2: 10, 3: 5, 4: 3, 5: 3}, max_n=5, debug=False):
        self.support_threshes = support_threshes
        self.max_n = max_n
        self.debug = debug

        ## CFI calculations DS
        self.all_itemsets = collections.defaultdict(FreqDist)
        self.all_freq_itemsets = collections.defaultdict(dict)

        ## Resulting Mappings
        self.doc_labels_map = {}
        self.label_docs_map = {}
        self.singleton_label_doc_map = {}

        # Final clusters
        self.clusters = {}
    # %%

    def fit(self, unique_expr_file=None):
        # select and load features
        extracted_df = pickle.load(open('../data/extracted_df.pkl', 'rb'))
        self.selected_df = load_selected_features(extracted_df)
        term_docs = self.selected_df.tfidf_terms

        # Closed FI graph construction
        self.initialize_first_pass(term_docs)
        self.gen_closed_frequent_itemsets(term_docs)

        if self.debug:
            logger.info('View Closed Frequent Item graph...')
            print_graph(self.closed_fi_graph)

        # Get top ranked matches of labels for all the expressions
        logger.info('Get label for each doc...')
        self.doc_labels_map, self.label_docs_map = self.get_matched_doc_labels(term_docs)

        # Postprocessing (if needed)
        logger.info('Postprocessing...' )
        self.singleton_label_doc_map = get_singletons(self.label_docs_map)
        #for label in self.label_docs_map.keys():
        #    print(label, len(self.label_docs_map[label]), self.label_docs_map[label], '\n')

        logger.info('Singletons found: {}'.format(len(self.singleton_label_doc_map)))
        logger.info('Removing {} singletons to get final clusters..\n'.format(len(self.singleton_label_doc_map)))
        self.clusters = {k: self.label_docs_map[k] for k in
                         set(self.label_docs_map) - set(self.singleton_label_doc_map)}

        logger.info('Final clusters Count : %d, Total docs: %d' % (
            len(self.clusters), len(self.selected_df) - len(self.singleton_label_doc_map)))

        if self.debug:
            logger.info('Final clusters '+ '=' * 10)
            print_clusters(self.clusters)

        return self.clusters

    ## --------------------------------- Closed Frequent itemsets Generation ------------------------------ ##
    def initialize_first_pass(self, term_docs):
        # initialize with 1st pass
        one_itemsets = gen_first_level_itemsets(term_docs)

        logger.info('Initializing 1st level itemsets..')
        self.all_itemsets[1] = one_itemsets

        self.all_freq_itemsets[1] = dict(
            [(term, freq) for term, freq in one_itemsets.items() if freq >= self.support_threshes[1]])

        self.closed_fi_graph = initialize_fi_graph(self.all_freq_itemsets[1].items())

        logger.info('Done. Total: {}, Frequent:{}, fi_nodes:{} \n'.format( \
              len(one_itemsets), len(self.all_freq_itemsets[1]), len(self.closed_fi_graph.nodes)))


    # %%
    # Update closed frequent nodes in graph
    def update_closed_fi(self, new_node, new_node_freq, parent_terms, disclosed_terms):
        #print('update_closed_fi: adding:', new_node, '->', parent_terms)
        parent_terms = [term for term in parent_terms if term in self.closed_fi_graph.nodes]
        self.closed_fi_graph.add_node(new_node, freq=new_node_freq)

        if not parent_terms:
            if self.debug:
                logger.info('no parent found for: {} attaching it to root.'.format(new_node))
            self.closed_fi_graph.add_edge(('root',), new_node, freq=new_node_freq)

        else:
            # update parents for new comb term
            for parent in parent_terms:
                self.closed_fi_graph.add_edge(parent, new_node, freq=new_node_freq)

        # Remove duplicate un-closed parents identified
        for disclosed_term in disclosed_terms:
            self.closed_fi_graph.remove_node(disclosed_term)


    # %%
    # Generate nth level item by combining, update frequent and closed items to graph
    def gen_nth_level_itemsets(self, term_docs, n):
        all_itemsets = self.all_itemsets
        n_1_itemset = all_itemsets[n-1].most_common()
        support_thresh = self.support_threshes[n]
        nth_itemsets = FreqDist()
        nth_freq_itemsets = FreqDist()

        k = n-2

        for i, (term1,freq1) in enumerate(n_1_itemset):
            for term2,freq2 in n_1_itemset[i+1:]:

                # check closed property
                if all_itemsets[len(term1)][term1] < support_thresh or all_itemsets[len(term2)][term2] < support_thresh:
                    continue

                # generate a combination by combining
                comb = get_comb(term1, term2, k)

                if not comb:
                    continue

                # scan all the docs
                comb_freq = get_occurrence_counts(comb, term_docs)

                if comb_freq > 0:
                    nth_itemsets[comb] = comb_freq
                    #self.all_itemsets[n][comb] = comb_freq

                # Update closed frequent itemset
                if comb_freq>=support_thresh:
                    # update comb to frequent itemsets
                    nth_freq_itemsets[comb] = comb_freq

                    # identify closed parents
                    parent_terms, disclosed_terms = get_closed_parents(comb_freq, term1, term2, self.closed_fi_graph)

                    # update closed connections to graph
                    self.update_closed_fi(comb, comb_freq, parent_terms, disclosed_terms)

        return nth_itemsets, nth_freq_itemsets


    # %%
    # Generate closed frequent itemsets for n levels
    def gen_closed_frequent_itemsets(self, term_docs):
        for n in range(2, self.max_n+1):
            # fetch n-1th level itemsets
            number_str = self.NUMBER_MAPPINGS[n] if n in self.NUMBER_MAPPINGS else str(n)+'th'
            logger.info('Generating {} level itemsets..'.format(number_str))
            items_sets, freq_itemsets = self.gen_nth_level_itemsets(term_docs, n)
            logger.info('Done. new_freq_itemsets:{}, total_cfi_nodes:{}\n'.format( \
                  len(freq_itemsets), len(self.closed_fi_graph.nodes)))

            self.all_itemsets[n] = items_sets
            self.all_freq_itemsets[n] = freq_itemsets

    ## --------------------------------- Assigning best matching labels ------------------------------ ##
    # %%
    # Penalty based scoring scheme
    # doc: ('claim', 'fund'), label_cand: ('fund',)--- valid
    # doc: ('fund',) label_cand: ('claim', 'fund') --- invalid
    def get_lbl_assign_score(self, doc, label, label_weighted=False, debug=False):
        term_scores = self.all_itemsets[1]
        match = set(doc).intersection(label)
        left = set(doc).difference(label)
        extra = set(label).difference(doc) # extra/not required terms

        match_weight = sum([np.log(term_scores[(m,)]) for m in match])
        left_weight = sum([np.log(np.sqrt(term_scores[(l,)])) for l in left])
        extra_weight = sum([np.log(term_scores[(e,)]) for e in extra])

        if debug:
            logger.info('match:{} score:{} left:{} score:{}'.format(str(match), match_weight, str(left), left_weight))

        weighted_score = (len(match)*match_weight) - (len(left)*left_weight) - (len(extra)*extra_weight)

        if label_weighted:
            label_weight = np.log(self.closed_fi_graph.node[label]['freq'])
            weighted_score = weighted_score * label_weight

        weighted_score_norm = weighted_score/(len(label)*len(doc))

        return weighted_score_norm

    # %%
    # Identify label and scores for a doc/expression
    def get_label_scores(self, doc, label_weighted=False):
        fi_graph = self.closed_fi_graph
        label_nodes = set(fi_graph.successors(('root',)))
        label_scores = FreqDist()
        level = 0
        while label_nodes:
            level += 1

            successors = set()
            for label in label_nodes:
                #print(doc, ':', label)
                if not is_label_cand(doc, label):
                    continue
                #print(doc, ':', label)
                label_score = self.get_lbl_assign_score(doc, label, label_weighted)

                if math.isinf(label_score):
                    print('Inf error:', label, label_score)

                label_scores[label] = label_score
                successors.update(list(fi_graph.successors(label)))

            label_nodes = successors

        return label_scores


    # %%
    # Get top matched labels in ranked order for the documents/expressions
    def get_matched_doc_labels(self, term_docs):
        doc_labels_map = {}
        label_docs_map = collections.defaultdict(list)
        top_unique_labels = collections.defaultdict(list)
        no_matches = 0

        for doc in term_docs:
            doc = tuple(doc)

            # already explored: Frequency of occurence can by retrieved from fi graph
            if not doc or doc in doc_labels_map:
                continue

            label_scores = self.get_label_scores(doc).most_common()
            #print('get_matched_doc_labels:',doc, ':', label_scores)
            if not label_scores:
                no_matches += 1
                label_scores = [(doc, 1.0)]

                if self.debug:
                    logger.info('No match for :{}'.format(doc))

            doc_labels_map[doc] = label_scores

            # maintain inverted map for labels
            for label,score in label_scores:
                label_docs_map[label].append((doc,score))

        logger.info('Total unmatched docs:{}'.format(no_matches))
        return doc_labels_map,label_docs_map

## --------------------------------- Helper functions ------------------------------ ##

def load_selected_features(extracted_df):
    logger.info('Selecting Features..')
    selected_df = extracted_df
    selected_df.loc[:, 'bigrams'] = selected_df.bigrams.apply(remove_sw)
    selected_df.loc[:, 'bigrams'] = selected_df.bigrams.apply(correction)

    tfidf_matrix, terms = get_tfidf_matrix(selected_df.bigrams)
    logger.info('tfidf_matrix shape: {}\n'.format(tfidf_matrix.shape))
    X = tfidf_matrix.toarray()
    selected_df.loc[:, 'tfidf_terms'] = [get_tfidf_words(X[i], terms) for i in range(len(X))]
    return selected_df

# %%
# ## Graph methods for Frequent Itemsets
def initialize_fi_graph(first_level_items):
    G = nx.DiGraph()
    G.add_node(('root',), freq=1)

    for item,freq in first_level_items:
        G.add_node(item, freq=freq)
        G.add_edge(('root',), item, freq=freq)

    return G

# %%
# Identify closed parent nodes for a node
def _get_closed_parent_nodes(parent_cands, second_parents, G):
    closed_parents = set()

    for parent in parent_cands:
        # 'fund' -> (fund,update)
        intersection = set(G.successors(parent)).intersection(second_parents)
        if len(intersection) > 1:
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
        parents[0] = _get_closed_parent_nodes(term_parents1, term_parents2, G)

    if parent_freq2 == comb_freq:
        parents[1] = _get_closed_parent_nodes(term_parents2, term_parents1, G)

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





# %%
if __name__=='__main__':
    support_threshes = {1: 30, 2: 10, 3: 5, 4: 3, 5: 3}
    max_n = 5
    cfi = CFIClustering(support_threshes, max_n, debug=False)
    cfi.fit()
    """
    term_docs = cfi.selected_df.tfidf_terms

    cfi.gen_closed_frequent_itemsets()

    print('\n', '='*20, ' View Closed Frequent Item graph ', '='*20)
    cfi.print_graph()

    print('\n', '='*20, ' Get label for each doc ', '='*20)
    # Get top ranked matches of labels for all the expressions
    doc_labels_map,label_docs_map,top_unique_labels = cfi.get_matched_doc_labels(term_docs)

    print('\n', '=' * 20, ' Postprocessing ', '=' * 20)
    # Postprocessing (if needed)
    # Identify weak cluster labels
    cfi.singleton_label_doc_map = cfi.get_singletons(label_docs_map)
    #weak_singleton_label_docs = cfi.get_weak_singletons(singleton_label_doc_map, doc_labels_map, 2)
    print('Singletons: ', len(cfi.singleton_label_doc_map))
    #print('Weak Singletons:', len(weak_singleton_label_docs))
    print('Removing singletons from final clusters..')
    cfi.clusters = label_docs_map.copy()
    for label in cfi.singleton_label_doc_map:
        del cfi.clusters[label]


    print('\n', '=' * 20, ' View clusters ', '=' * 20)
    cfi.print_clusters()
    """
    """
    print('\nViewing Closed Frequent labels distribution')
    node_distr = cfi.check_cluster_distr()
    for key in node_distr.keys():
        print(key, ':')
        for item in node_distr[key].most_common():
            print(item)
        print('-'*40)
    # Remove them if found
    #filtered_lbl_docs_map, filtered_doc_lbls_map, cluster_graph = remove_weak_singletons(weak_singleton_label_docs, label_docs_map, doc_labels_map, cluster_graph,True)
    """
