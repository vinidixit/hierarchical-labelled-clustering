# %%
import pickle
import pandas as pd
import math
import matplotlib.pyplot as plt
from utils import get_tfidf_matrix, dim_reduce_svd
import collections
import itertools
from nltk.probability import FreqDist
from frozendict import frozendict

import numpy as np
import scipy

# graph library
import networkx as nx
from copy import deepcopy

# Visualization libraries
from pyvis.network import Network
from pylab import *

## Some corrections in data
def remove_sw(terms):
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



# %%
tfidf_matrix, terms = get_tfidf_matrix(selected_df.trigrams)
tfidf_matrix.shape


# %%
def get_tfidf_words(row, terms):
    return [terms[i] for i,val in enumerate(row) if val>0]


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
# Identify closed parent nodes for a node
def get_closed_parent_nodes(closed_fi_graph, parent_cands, second_parents):
    closed_parents = set()
    
    for parent in parent_cands:
        # 'fund' -> (fund,update)
        intersection = set(closed_fi_graph.successors(parent)).intersection(second_parents)
        if len(intersection) > 1:
            print('More than one intersect found..', intersection, ' from: ', parent, ' with ', second_parents)
            
        if not intersection:
            closed_parents.add(parent)
    
    return list(closed_parents)


# %%
# Identify closed parents
def get_closed_parents(closed_fi_graph, comb, comb_freq, term1, term2):
    parent_term1, parent_term2 = sorted([term1, term2])
    parent_freq1 =  closed_fi_graph.node[parent_term1]['freq'] if parent_term1 in closed_fi_graph.nodes else 0
    parent_freq2 =  closed_fi_graph.node[parent_term2]['freq'] if parent_term2 in closed_fi_graph.nodes else 0
    
    disclosed_terms = set()
    term_parents1, term_parents2 = [parent_term1], [parent_term2]
    parents = [[parent_term1], [parent_term2]]
    
    if parent_freq1 == comb_freq:
        # go one level up (proven to be already closed)
        term_parents1 = list(closed_fi_graph.predecessors(parent_term1))
        disclosed_terms.add(parent_term1)
    
    if parent_freq2 == comb_freq:
        # go one level up (proven to be already closed)
        term_parents2 = list(closed_fi_graph.predecessors(parent_term2))
        disclosed_terms.add(parent_term2)
    
    if parent_freq1 == comb_freq:
        parents[0] = get_closed_parent_nodes(closed_fi_graph, term_parents1, term_parents2)
        
    if parent_freq2 == comb_freq:
        parents[1] = get_closed_parent_nodes(closed_fi_graph, term_parents2, term_parents1)
    
    parents = sorted(list(itertools.chain(*parents)))
    
    if ('root',) in parents:
        parents.remove(('root',))
        
    return sorted(parents), disclosed_terms


# %%
# Update closed frequent nodes in graph
def update_closed_fi(closed_fi_graph, new_node, new_node_freq, parent_terms, disclosed_terms):
    closed_fi_graph.add_node(new_node, freq=new_node_freq)
    
    # update parents for new comb term
    for parent in parent_terms:
        if parent in closed_fi_graph.nodes:
            closed_fi_graph.add_edge(parent, new_node, freq=new_node_freq)
        #else:
            #print(parent, ' does not exist in graph..')

    # Remove duplicate un-closed parents identified
    for disclosed_term in disclosed_terms:
        closed_fi_graph.remove_node(disclosed_term)


# %%
# Generate nth level item by combining, update frequent and closed items to graph
def gen_n_level_itemset(term_docs, n, support_threshes, closed_fi_graph, all_itemsets):
    n_1_itemset = all_itemsets[n-1].most_common()
    support_thresh = support_threshes[n]
    
    k = n-2
    comb_items = FreqDist()
    
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
                all_itemsets[n][comb] = comb_freq
            
            # Update closed frequent itemset
            if closed_fi_graph and comb_freq>=support_thresh:
                # update comb to frequent itemsets
                comb_items[comb] = comb_freq
                
                # identify closed parents
                parent_terms, disclosed_terms = get_closed_parents(closed_fi_graph, comb, comb_freq, term1, term2)
                
                # update closed connections to graph 
                update_closed_fi(closed_fi_graph, comb, comb_freq, parent_terms, disclosed_terms)    
                                      
    return comb_items


# %%
# Generate closed frequent itemsets for n levels
def get_closed_frequent_itemsets(term_docs, support_threshes, max_n=5):
    print('Generating %d level itemsets..'%(1))
    one_itemsets = gen_first_level_itemsets(term_docs)
    all_itemsets = {1:one_itemsets}
    all_freq_itemsets = {1:dict([(term,freq) for term,freq in one_itemsets.items() if freq>=support_threshes[1]])}
    closed_fi_graph = initialize_fi_graph(all_freq_itemsets[1].items())
    print('Done. Count:', len(all_freq_itemsets[1]), len(closed_fi_graph.nodes), '\n')
    
    for n in range(2, max_n+1):
        # fetch n-1th level itemsets
        n_1_itemset = list(all_itemsets[n-1].items())
        print('Generating %d level itemsets..'%(n))
        all_itemsets[n] = FreqDist()
        freq_itemsets = gen_n_level_itemset(term_docs, n, support_threshes, closed_fi_graph, all_itemsets)
        print('Done. Count:', len(freq_itemsets), len(closed_fi_graph.nodes), '\n')
        all_freq_itemsets[n] = freq_itemsets
    
    return frozendict(all_itemsets), frozendict(all_freq_itemsets), closed_fi_graph


# %%
# Penalty based scoring scheme
# doc: ('claim', 'fund'), label_cand: ('fund',)--- valid
# doc: ('fund',) label_cand: ('claim', 'fund') --- invalid
def get_lbl_assign_score(doc, label, fi_graph, label_weighted=False, term_scores=all_itemsets[1], debug=False):
    match = set(doc).intersection(label)
    left = set(doc).difference(label)
    extra = set(label).difference(doc) # extra/not required terms
    
    match_weight = sum([np.log(term_scores[(m,)]) for m in match])
    left_weight = sum([np.log(np.sqrt(term_scores[(l,)])) for l in left])
    extra_weight = sum([np.log(term_scores[(e,)]) for e in extra])
    
    if debug:
        print('match:%s score:%f left:%s score:%f'%(str(match), match_weight, str(left), left_weight))
        
    weighted_score = (len(match)*match_weight) - (len(left)*left_weight) - (len(extra)*extra_weight)
    
    if label_weighted:
        label_weight = np.log(fi_graph.node[label]['freq'])
        weighted_score = weighted_score * label_weight
        
    weighted_score_norm = weighted_score/(len(label)*len(doc))
    
    return weighted_score_norm


# %%
# Check if a label is valid candidate
def is_label_cand(doc, label):
    return set(label).issubset(doc) #not set(doc).isdisjoint(label)


# %%
# Identify label and scores for a doc/expression
def get_label_scores(doc, fi_graph, label_weighted=False):
    label_nodes = set(fi_graph.successors(('root',)))
    label_scores = FreqDist()
    level = 0
    while label_nodes:
        level += 1
        
        successors = set()
        for label in label_nodes:
            if not is_label_cand(doc, label):
                continue
            
            label_score = get_lbl_assign_score(doc, label, fi_graph, label_weighted)
            
            if math.isinf(label_score):
                print('Inf error:', label, label_score)
            
            label_scores[label] = label_score
            successors.update(list(fi_graph.successors(label)))
            
        label_nodes = successors
        
    return label_scores   


# %%
# Filter threshold criteria
def filter_thresh(scores, choice):
    if choice == 1:
        return (max(scores)-min(scores))/2
    elif choice == 2:
        return np.mean(scores[scores>0])
    elif choice == 3:
        return np.mean([max(0, score) for score in scores])
    elif choice == 4:
        return np.mean(scores)
    elif choice == 5:
        return np.percentile(scores, 95)
    elif choice == 6:
        return np.percentile(scores, 75) + 1.5*scipy.stats.iqr(scores)
    else:
        print('Wrong choice.')       


# %%
# Get top matched labels in ranked order for a doc
def get_top_ranked_labels(doc, fi_graph, debug=False):
    label_scores = get_label_scores(doc, fi_graph)
    ordered_label_scores = label_scores.most_common()
    
    if debug:
        print('available labels: ', ordered_label_scores[:10],'\n')
    
    scores = np.asarray(list(label_scores.values()))
    if debug:
        boxplot(scores)
    
    max_percentile = filter_thresh(scores, 5)
    outlier_thresh = filter_thresh(scores, 6)
    strict_filter_thresh = max(max_percentile, outlier_thresh)
    loose_filter_thresh = min(max_percentile, outlier_thresh)
    
    if debug:
        print('95th percentile:', filter_thresh(scores, 5))
        print('+1.5*iqr:', filter_thresh(scores, 6))
        
    best_labels, good_labels = [], []
    for label,score in ordered_label_scores:
        if score >= strict_filter_thresh:
            best_labels.append((label, score))
        elif score >= loose_filter_thresh:
            good_labels.append((label, score))
        else:
            break
        
    return (np.asarray(best_labels), np.asarray(good_labels)),(strict_filter_thresh,loose_filter_thresh)


# %%
# Get top matched labels in ranked order for the documents/expressions
def get_matched_doc_labels(term_docs, fi_graph, debug = False):
    doc_labels_map = {}
    label_docs_map = collections.defaultdict(list)
    top_unique_labels = collections.defaultdict(list)
    
    for doc in term_docs:
        doc = tuple(doc)
        
        # already explored: Frequency of occurence can by retrieved from fi graph
        if doc in doc_labels_map:
            continue
        
        label_scores = get_label_scores(doc, fi_graph).most_common()
        top_unique_labels[label_scores[0][0]].append(doc)
        doc_labels_map[doc] = label_scores
        
        # maintain inverted map for labels
        for label,score in label_scores:
            label_docs_map[label].append((doc,score))
            
    return doc_labels_map,label_docs_map,top_unique_labels


# %%
def is_perfect_match(doc, label):
    if (len(label) <= len(doc) and set(label).issubset(doc)):# or set(doc).issubset(label):
        return True
    return False


# %%
# check for any casualties
def check_unassigned_doc(doc_lbls_map):
    for doc in doc_lbls_map:
        if len(doc_lbls_map[doc])==0:
            print(doc)


# %%
# Identify singleton labels
def get_singletons(label_docs_map):
    return dict(filter(lambda elem: len(elem[1])==1, label_docs_map.items()))

singleton_label_doc_map = get_singletons(label_docs_map)
len(singleton_label_doc_map), singleton_label_doc_map


# %%
# identify weak singleton labels
def is_weak_singleton(label, doc, match_score, doc_labels_map, thresh):
    top_n = 1
    
    # high score
    if match_score >= thresh: #label==doc:
        return False
    
    available_labels = doc_labels_map[doc] #list(itertools.chain(*list(doc_labels_map[doc].values())))
    available_labels.sort(key=lambda x:x[1], reverse=True)
    top_labels = np.asarray(available_labels[:top_n])[:,0] # extract label names from top labels
    
    if label in top_labels: # highest score
        return False
    
    return True


# %%
def get_weak_singletons(singletons, doc_lbls_map, thresh=1):
    weak_singlenton = {}
    for label,[(doc,score)] in singletons.items():
        if is_weak_singleton(label, doc, score, doc_lbls_map, thresh):
            weak_singlenton[label] = (doc,score)
        
    return weak_singlenton


# %%
def remove_weak_singletons(weak_singleton_label_docs, label_docs_map, doc_labels_map, cluster_graph, debug=False):
    print('original labels:', len(label_docs_map))
    filtered_lbl_docs_map = deepcopy(label_docs_map)
    filtered_doc_lbls_map = deepcopy(doc_labels_map)
    
    for label,(doc,score) in weak_singleton_label_docs.items():
        # remove weak singleton label occurrences
        del filtered_lbl_docs_map[label]
        filtered_doc_lbls_map[doc].remove((label,score))
        cluster_graph.remove_node(label)
        
    if debug:
        print('Removed %d singletons, remaining labels: %d'% (len(weak_singleton_label_docs), len(filtered_lbl_docs_map)))
    
    return frozendict(filtered_lbl_docs_map), frozendict(filtered_doc_lbls_map),cluster_graph


# %%
def check_cluster_distr():
    closed_fi_distr = {1:FreqDist(),2:FreqDist(),3:FreqDist(),4:FreqDist(),5:FreqDist()}

    for node in closed_fi_graph.nodes:
        closed_fi_distr[len(node)][node] = closed_fi_graph.node[node]['freq']
    
    return closed_fi_distr


# %%
if __name__=='__main__':
    # ## Data Preparation
    artifacts_path = '/workspace/Chatbot/expression-intents-analysis/mlruns/0/61c3e9cfad17452383a421d6117cfdb0/artifacts'
    selected_df_file = artifacts_path + '/selected_df/selected_df.pkl'
    selected_df = pickle.load(open(selected_df_file, 'rb'))

    # remove extra stopwords
    stopwords = ['limited', 'consulting']

    selected_df.loc[:,'trigrams'] = selected_df.trigrams.apply(remove_sw)
    selected_df.loc[:,'trigrams'] = selected_df.trigrams.apply(correction)
    
    X = tfidf_matrix.toarray()
    selected_df.loc[:,'tfidf_terms'] = [get_tfidf_words(X[i], terms) for i in range(len(X))]
    
    support_threshes = {1:30, 2:10, 3:5, 4:3, 5:3}
    all_itemsets, all_freq_itemsets,closed_fi_graph = get_closed_frequent_itemsets(selected_df.tfidf_terms, support_threshes)   
    
    # Get top ranked matches of labels for all the expressions
    doc_labels_map,label_docs_map,top_unique_labels = get_matched_doc_labels(selected_df.tfidf_terms, closed_fi_graph)
    
    # Postprocessing (if needed)
    # Identify weak cluster labels
    singleton_label_doc_map = get_singletons(label_docs_map)
    weak_singleton_label_docs = get_weak_singletons(singleton_label_doc_map, doc_labels_map, 2)
    # Remove them if found
    filtered_lbl_docs_map, filtered_doc_lbls_map,cluster_graph = remove_weak_singletons(weak_singleton_label_docs, label_docs_map, doc_labels_map, cluster_graph,True)

    

