from _cluster_utils import *
from nltk.probability import FreqDist
import pandas as pd
import collections
import logging
import sys
import pickle
import math
import numpy as np
from feature_embedding_clustering import get_embedding_cluster_obj, FeatureEmbeddingClustering

LOG_FORMAT = '%(asctime)s : %(name)s : %(message)s'
formatter = logging.Formatter(fmt=LOG_FORMAT)
logger = logging.getLogger('CFIClustering')

if (logger.hasHandlers()):
    logger.handlers.clear()

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

class CFIEClustering():
    _NUMBER_MAPPINGS = {1: '1st', 2: '2nd', 3: '3rd'}
    default_thresh = {1: 30, 2: 10, 3: 5, 4: 3, 5: 3}

    def __init__(self, support_threshes={1: 30, 2: 10, 3: 5, 4: 3, 5: 3}, debug=False):
        self.support_threshes = support_threshes
        self.max_n = max(support_threshes.keys())
        self.debug = debug

        ## CFI calculations DS
        self._all_itemsets = collections.defaultdict(FreqDist)
        self._all_freq_itemsets = collections.defaultdict(dict)
        self._closed_fi_graph = nx.DiGraph()

        ## Resulting Mappings
        self.cluster_processed_df = None
        self.singletons = None

        # Final clusters
        self._clusters = {}
    # %%

    def fit(self, sentence_df, labels_cluster_obj= None, max_features_frac=.20):
        # select features from extracted candidates
        max_features = int(max_features_frac*len(sentence_df)) if max_features_frac else None
        print('Max_features: ', max_features)
        self.tfidf_matrix, self.tfidf_terms = get_tfidf_matrix(sentence_df.n_grams, max_features=max_features)

        if labels_cluster_obj:
            sentence_df['n_grams_emb'] = labels_cluster_obj.transform(sentence_df.n_grams.values) #self.get_embedded_ngrams(sentence_df, labels_cluster_obj)
            print('Initial tfidf terms :', len(self.tfidf_terms))
            self.tfidf_matrix, self.tfidf_terms = get_tfidf_matrix(sentence_df.n_grams_emb, max_features=max_features)
            print('After embedding tfidf terms :', len(self.tfidf_terms), '\n')

        sentence_df['terms_doc'] = load_selected_features(self.tfidf_matrix, self.tfidf_terms)

        # Closed FI graph construction
        self._initialize_first_pass(sentence_df['terms_doc'])
        self._gen_closed_frequent_itemsets(sentence_df['terms_doc'])

        # Get top ranked matches of labels for all the expressions
        logger.info('Get label for each doc...')
        doc_labels, label_trees, label_docs_map, levelled_labels = self._get_matched_doc_labels(sentence_df)
        #print(len(levelled_labels), len(sentence_df))
        assert(len(doc_labels) == len(sentence_df) and len(label_trees)==len(sentence_df) and len(levelled_labels)==len(sentence_df))
        sentence_df['labels'] = doc_labels
        sentence_df['labels_tree'] = label_trees
        sentence_df['levelled_labels'] = levelled_labels

        self._clusters = label_docs_map

        # Postprocessing (if needed)
        logger.info('Postprocessing...' )
        self.singletons = sentence_df[(~sentence_df.isNoisy) & (sentence_df.labels.map(len)==0)]
        print('Singletons :', len(self.singletons))

        logger.info('Final clusters Count : %d, Total docs assigned: %d, Outliers/unassigned: %d.' % (
            len(self._clusters), len(sentence_df) - len(self.singletons), len(self.singletons)))

        self.cluster_processed_df = sentence_df

        return self._clusters


    def get_embedded_ngrams(self, sentence_df, labels_cluster_obj=None):

        if labels_cluster_obj is None:
            leader_threshes = [0.8, 0.7, 0.6, 0.5]
            member_threshes = [0.7, 0.6, 0.5, 0.4]
            thresh_pairs = list(zip(leader_threshes, member_threshes))

            labels_cluster_obj = get_embedding_cluster_obj(sentence_df.n_grams.values, thresh_pairs)

        return labels_cluster_obj.transform(sentence_df.n_grams.values)


    ## --------------------------------- Closed Frequent itemsets Generation ------------------------------ ##
    def _initialize_first_pass(self, term_docs):
        # initialize with 1st pass
        one_itemsets = gen_first_level_itemsets(term_docs)
        corpus_size = sum(one_itemsets.values())
        max_freq = max(one_itemsets.values())
        max_freq_tem = max(list(one_itemsets.items()), key=lambda x:x[1])
        logger.info('***corpus_size: ' + str(corpus_size))
        logger.info('***max_freq: '+ str(max_freq))
        logger.info('***max_freq item: ' + str(max_freq_tem))

        #print('max_freq: ', max(one_itemsets.values()))
        if type(self.support_threshes[1]) == float:
            #self.support_threshes[1] = max(int(self.support_threshes[1] * max_freq / 100), 2)

            for level in self.support_threshes.keys():
                self.support_threshes[level] = max(int(self.support_threshes[level] * max_freq / 100), 2)

            logger.info(self.support_threshes)

        logger.info('Initializing 1st level itemsets..')
        self._all_itemsets[1] = one_itemsets

        self._all_freq_itemsets[1] = dict(
            [(term, freq) for term, freq in one_itemsets.items() if freq >= self.support_threshes[1]])

        self._closed_fi_graph = initialize_fi_graph(self._closed_fi_graph, self._all_freq_itemsets[1].items())

        logger.info('Done. Total: {}, Frequent:{}, fi_nodes:{} \n'.format( \
              len(one_itemsets), len(self._all_freq_itemsets[1]), len(self._closed_fi_graph.nodes)))

    # %%
    # Generate closed frequent itemsets for n levels
    def _gen_closed_frequent_itemsets(self, term_docs):
        for n in range(2, self.max_n + 1):
            # fetch n-1th level itemsets
            number_str = self._NUMBER_MAPPINGS[n] if n in self._NUMBER_MAPPINGS else str(n) + 'th'
            logger.info('Generating {} level itemsets..'.format(number_str))
            items_sets, freq_itemsets = self._gen_nth_level_itemsets(term_docs, n)
            logger.info('Done. new_freq_itemsets:{} out of {} itemsets, total_cfi_nodes:{}\n'.format( \
                len(freq_itemsets), len(items_sets), len(self._closed_fi_graph.nodes)))
            self._all_itemsets[n] = items_sets
            self._all_freq_itemsets[n] = freq_itemsets

    # %%
    # Generate nth level item by combining, update frequent and closed items to graph
    def _gen_nth_level_itemsets(self, term_docs, n):
        all_itemsets = self._all_itemsets
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

                # Update closed frequent itemset
                if comb_freq>=support_thresh:
                    #print(comb, comb_freq)
                    # update comb to frequent itemsets
                    nth_freq_itemsets[comb] = comb_freq

                    # identify closed parents
                    parent_terms, disclosed_terms = get_closed_parents(comb_freq, term1, term2, self._closed_fi_graph)

                    # update closed connections to graph
                    self._update_closed_fi(comb, comb_freq, parent_terms, disclosed_terms)

        return nth_itemsets, nth_freq_itemsets

    # %%
    # Update closed frequent nodes in graph
    def _update_closed_fi(self, new_node, new_node_freq, parent_terms, disclosed_terms):
        parent_terms = [term for term in parent_terms if term in self._closed_fi_graph.nodes]
        self._closed_fi_graph.add_node(new_node, freq=new_node_freq)

        if not parent_terms:
            if self.debug:
                logger.info('no parent found for: {} attaching it to root.'.format(new_node))
            self._closed_fi_graph.add_edge(('root',), new_node, freq=new_node_freq)

        else:
            # update parents for new comb term
            for parent in parent_terms:
                self._closed_fi_graph.add_edge(parent, new_node, freq=new_node_freq)

        # Remove duplicate un-closed parents identified
        for disclosed_term in disclosed_terms:
            self._closed_fi_graph.remove_node(disclosed_term)


    ## --------------------------------- Assigning best matching labels ------------------------------ ##
    def _get_labels(self, doc, label_weighted=False):
        fi_graph = self._closed_fi_graph
        label_nodes = set(fi_graph.successors(('root',)))
        label_scores = FreqDist()
        label_tree = nx.DiGraph()
        levelled_labels = collections.defaultdict(list)
        label_tree.add_node(('label_root',))
        level = 0

        while label_nodes:
            level += 1
            successors = set()

            for label in label_nodes:
                if not is_label_cand(doc, label):
                    continue

                label_score = self._get_lbl_assign_score(doc, label, label_weighted)

                if math.isinf(label_score):
                    logger.error('Inf error:', label, label_score)

                label_scores[label] = label_score
                levelled_labels[level].append(label)
                label_tree.add_node(label)
                connections = set(fi_graph.predecessors(label)).intersection(label_tree.nodes) if level>1 else set([('label_root',)])

                for parent in connections:
                    #print('Adding ', parent, ' to label:', label)
                    label_tree.add_edge(parent, label)

                successors.update(list(fi_graph.successors(label)))

            label_nodes = successors
        return label_scores.most_common(), label_tree, levelled_labels

    # %%
    # Penalty based scoring scheme
    # doc: ('claim', 'fund'), label_cand: ('fund',)--- valid
    # doc: ('fund',) label_cand: ('claim', 'fund') --- invalid
    def _get_lbl_assign_score(self, doc, label, label_weighted=False, debug=False):
        term_scores = self._all_itemsets[1]
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
            label_weight = np.log(self._closed_fi_graph.node[label]['freq'])
            weighted_score = weighted_score * label_weight

        weighted_score_norm = weighted_score/(len(label)*len(doc))

        return round(weighted_score_norm,3 )

    # %%
    # Identify label and scores for a doc/expression
    def _get_label_scores(self, doc, label_weighted=False):
        fi_graph = self._closed_fi_graph
        label_nodes = set(fi_graph.successors(('root',)))
        label_scores = FreqDist()
        label_tree = nx.DiGraph()
        label_tree.add_node(('label_root',))
        level = 0

        while label_nodes:
            level += 1

            successors = set()
            for label in label_nodes:
                if not is_label_cand(doc, label):
                    continue

                label_score = self._get_lbl_assign_score(doc, label, label_weighted)

                if math.isinf(label_score):
                    logger.error('Inf error:', label, label_score)

                label_scores[label] = label_score
                label_tree.add_node(label, score=label_score)

                if level == 1:
                    label_tree.add_edge(('label_root',), label)

                for s_node in list(fi_graph.successors(label)):
                    if not is_label_cand(doc, s_node):
                        label_tree.add_edge(label, s_node)
                        successors.add(s_node) #.update(list(fi_graph.successors(label)))

            label_nodes = successors

        return label_scores.most_common(), label_tree

    # %%
    # Get top matched labels in ranked order for the documents/expressions
    def _get_matched_doc_labels(self, sentence_df):
        doc_labels = []
        label_trees = []
        label_docs_map = collections.defaultdict(list)
        all_levelled_labels = []

        no_matches = 0

        for i, row in sentence_df.iterrows():
            term_doc = row.terms_doc

            if not term_doc:
                no_matches += 1
                doc_labels.append([])
                label_trees.append(nx.DiGraph())
                all_levelled_labels.append({})
                continue

            label_scores, label_tree, levelled_labels = self._get_labels(term_doc)

            if not label_scores:
                no_matches += 1
                label_scores = []

                if self.debug:
                    logger.info('No match for :{}'.format(row.sentence))

            doc_labels.append(label_scores)
            label_trees.append(label_tree)
            all_levelled_labels.append(levelled_labels)

            # maintain inverted map for labels
            for label,score in label_scores:
                label_docs_map[label].append((i,score))

        logger.info('Total unmatched docs:{}'.format(no_matches))
        return doc_labels, label_trees, label_docs_map,all_levelled_labels


    def _get_tfidf_score(self, term, doc_index):
        term_index = list(self.tfidf_terms).index(term)
        doc_row = self.tfidf_matrix.toarray()[doc_index]
        return doc_row[term_index]

    def _get_label_tfidf_score(self, labels, doc_index):
        labels = np.array(labels)[:,0]
        label_score_tuples = []

        for label in labels:
            label_score = sum([self._get_tfidf_score(term, doc_index) for term in label])/len(label)
            label_score_tuples.append((label, round(label_score, 4)))

        label_score_tuples = sorted(label_score_tuples, key=lambda x: x[1], reverse=True)

        return label_score_tuples

    def _get_label_tftdf_scores(self):
        doc_labels_df = self.cluster_processed_df
        label_tfidf_scores = []

        for doc_index in range(len(doc_labels_df)):
            doc_row = doc_labels_df.iloc[doc_index]
            labels_scores = self._get_label_tfidf_score(doc_row.labels, doc_index) if len(doc_row.labels)>0 else []
            label_tfidf_scores.append(labels_scores)

        return label_tfidf_scores

    def has_positive_label(self, labels):
        if len(labels) == 0:
            return False

        return any([l[1]>0 for l in labels])

    def evaluate_label_quality(self, max_dup=2):
        print('Total documents: ', len(self.cluster_processed_df))
        print('Total clusters :', len(self._clusters))
        print('Total singleton clusters :', len(self.singletons))

        max_dup_labels = math.ceil(2*len(self._clusters)/100)
        print('Maximum allowed labels per doc: ', max_dup_labels)
        positive_labels = [label for label,docs in self._clusters.items() if any([d[1]>0 for d in docs])]
        print('Total positive labels/clusters: ', len(positive_labels))


        labels_tfidf_scores = self._get_label_tftdf_scores()
        self.cluster_processed_df['labels_tfidf'] = labels_tfidf_scores

        analysis_df = self.cluster_processed_df[self.cluster_processed_df.labels_tfidf.map(len)>0]
        min_scores = analysis_df.labels_tfidf\
                                            .apply(lambda l: min(l, key=lambda x: x[1])).values

        max_scores = analysis_df.labels_tfidf\
                                            .apply(lambda l: max(l, key=lambda x: x[1])).values

        tfidf_scores = list(itertools.chain(*[np.array(labels)[:,1] for labels in analysis_df.labels_tfidf]))

        print('min score: ', np.min(tfidf_scores))
        print('max score: ',np.max(tfidf_scores))
        print('mean score: ', np.mean(tfidf_scores))
        print('median score: ', np.median(tfidf_scores))

        q1, q3 = np.percentile(tfidf_scores, [25, 75])
        iqr = q3 - q1
        print("IQR analysis..")
        print(q1, q3, iqr)
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        print(lower_bound, upper_bound)

        print('tfidf score range :', min(min_scores, key=lambda x:x[1]), max(max_scores, key=lambda x:x[1]))


        overlabelled_count = 0
        overlabelled_pos_count = 0

        for doc_index,doc in self.cluster_processed_df.iterrows():

            pos_labels = [l for l in doc.labels if l[1] >0]

            if len(pos_labels) >= max_dup_labels:
                overlabelled_pos_count += 1

            if len(doc.labels) >= max_dup_labels:
                overlabelled_count += 1

        print('\nDocuments with more than %d (max allowed) cluster membership..' % (max_dup_labels))
        print('Out of all assigned: %d, out of assigned positive labels : %d. ' % (overlabelled_count, overlabelled_pos_count))




        """
        for overlabelled_doc in overlabelled_pos_docs:
            print(overlabelled_doc[0])
            print(overlabelled_doc[1])
            print(overlabelled_doc[2], '\n\n')
        """
        """
        print('\n\nDocuments with no positive label..')
        for _, row in analysis_df.iterrows():
            labels = row.labels
            tfidf_labels = dict(row.labels_tfidf)

            positive_labels = []
            negative_labels = []
            for l, s in labels:
                if s < 0:
                    negative_labels.append(((l, s, tfidf_labels[l])))
                else:
                    positive_labels.append((l, s, tfidf_labels[l]))

            if len(positive_labels)==0:
                print('\n', row.n_grams)
                for label in negative_labels:
                    print(label)
        """

    def remove_label_doc(self, label, doc_id):
        if label in self._clusters:
            doc_entries = list(np.array(self._clusters[label])[:, 0])
            doc_entry = self._clusters[label][doc_entries.index(doc_id)]
            #print('removing :', doc_entry, 'from cluster :', label)
            self._clusters[label].remove(doc_entry)

            if len(self._clusters[label]) == 0:
                del self._clusters[label]


    def postprocessing_labels(self, max_dup=2, keep_negative=False):

        max_dup_labels = math.ceil(max_dup * len(self._clusters) / 100) if max_dup else None
        print('Maximum clusters allowed per doc: ', max_dup_labels)
        unassigned_doc_count = 0
        total_initial_assigned_count = 0

        print('Initial CLusters count :', len(self._clusters))
        print('Total documents count :', self.cluster_processed_df.shape[0])
        selected_labels_list = pd.Series(index=self.cluster_processed_df.index, dtype=object)

        for doc_index, doc in self.cluster_processed_df.iterrows():
            if len(doc.labels) > 0:
                total_initial_assigned_count += 1

            selected_labels = doc.labels[:max_dup_labels] if max_dup_labels else doc.labels

            if not keep_negative:
                selected_labels = [l for l in selected_labels if l[1] > 0]

            rejected_labels = set(doc.labels).difference(selected_labels)

           # if len(rejected_labels)>0:
           #     print('Rejected labels: ', len(rejected_labels))

            if len(self.cluster_processed_df.loc[doc_index].labels) > 0 and len(selected_labels)==0:
                unassigned_doc_count += 1

            selected_labels_list.loc[doc_index] = selected_labels

            # remove reject label doc entry from clusters
            for reject_label,_ in rejected_labels:
                self.remove_label_doc(reject_label, doc_index)

        self.cluster_processed_df['selected_labels'] = selected_labels_list

        print('Total initial assigned documents count :', total_initial_assigned_count)
        print('\nTotal clusters after removing extra labels: ', len(self._clusters))
        print('Newly unassigned doc count after removing extra labels: ', unassigned_doc_count)
        print('Newly unassigned doc %% out of earlier assigned docs: %.3f %% and total docs: %.3f %%' %
                                                    (unassigned_doc_count*100/total_initial_assigned_count,
                                                     unassigned_doc_count*100/self.cluster_processed_df.shape[0]))

    def evaluate_clusters_quality(self):
        doc_labels_df = self.cluster_processed_df

        label_name = 'selected_labels' if 'selected_labels' in doc_labels_df.columns else 'labels'

        total_documents = len(doc_labels_df)
        noisy_documents = len(doc_labels_df[doc_labels_df.isNoisy])
        selected_docs_count = total_documents-noisy_documents

        unassigned_count = len(doc_labels_df[doc_labels_df[label_name].map(len) == 0])-noisy_documents
        unassigned_percent = unassigned_count*100/selected_docs_count

        print('Total documents :', total_documents)
        print('Noisy documents :', noisy_documents)
        print('Selected documents :', selected_docs_count)
        print('Unassigned doc count: %d and %.3f %%' % (unassigned_count, unassigned_percent))

        assigned_count = selected_docs_count - unassigned_count

        docs_with_pos_labels = len(doc_labels_df[
                                            doc_labels_df[label_name].apply(lambda x: self.has_positive_label(x))
                                        ]) - noisy_documents

        assigned_count = selected_docs_count - unassigned_count
        print('Total assigned docs: %d, %.3f %%' % (assigned_count, assigned_count*100/selected_docs_count))
        print('Positive assigned documents: ', docs_with_pos_labels)

        pos_ass_percent = round(docs_with_pos_labels*100/assigned_count, 3)
        pos_ass_total_percent = round(docs_with_pos_labels * 100 / selected_docs_count, 3)

        print('Positive assigned out of assigned docs : %.3f %%, out of selected docs: %.3f %%, and out of all docs: %.3f %%.'%\
              (docs_with_pos_labels*100/assigned_count, docs_with_pos_labels*100/selected_docs_count,\
               docs_with_pos_labels*100/doc_labels_df.shape[0]))

        clusters_count = len(self._clusters)
        singleton_clusters_count =  len(self.singletons)

        cluster_metrics = {'noiseless_documents': selected_docs_count, 'clusters_count':clusters_count,\
                           'singleton_clusters_count':singleton_clusters_count, \
                           'unassigned_doc_percent': unassigned_percent, 'assigned_doc_percent':round(100-unassigned_percent,3),\
                           'positive_label_in_assigned': pos_ass_percent, 'positive_label_in_total':pos_ass_total_percent}

        return cluster_metrics


import mlflow
import tempfile

def log_mlflow_experiment(experiment_parameters, experiment_metrics, artifacts):
    mlflow.start_run()

    for k, v in experiment_parameters.items():
        mlflow.log_param(k, v)

    for k, v in experiment_metrics.items():
        if type(v) == str:
            print(k, v, '*string*')
            mlflow.log_param(k, v)
        else:
            mlflow.log_metric(k, v)

    local_dir = tempfile.mkdtemp()

    for name, obj in artifacts.items():
        artifact_pkl = os.path.join(local_dir, name+'.pkl')
        pickle.dump(obj, open(artifact_pkl, 'wb'))
        mlflow.log_artifact(artifact_pkl, name)

    mlflow.end_run()

import os

def prepare_cluster_setup(corpus_name, emb):

    cluster_dir = '../../sample_data/clustering_results/' + corpus_name + '/'

    if corpus_name == 'Reuters-headline':
        #max_dup = None
        #max_features_frac = None
        support_threshes = {1: 1.5, 2: 0.9, 3: 0.4, 4: 0.2, 5: 0.1}
        fe_filename = 'feature_extracted_19961119_headline_df.pkl'

    elif corpus_name == 'Reuters-text':
        #max_dup = 2
        #max_features_frac = 0.1
        support_threshes = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0, 5: 1.5}
        fe_filename = 'feature_extracted_19961119_text_df.pkl'

    else:
        print('Corpus name %s unknown..' % (corpus_name))
        sys.exit(0)

    input_data_dir = '../../sample_data/processed_data/'
    processed_sentence_df = pickle.load(open(input_data_dir + fe_filename, 'rb'))

    labels_cluster_obj = None

    if emb is not None:
        fe_cluster_fname = cluster_dir + 'feature_emb_obj.pkl'
        print(fe_cluster_fname)

        if os.path.isfile(fe_cluster_fname):
            print('Found feaure emb obj loading..')
            labels_cluster_obj = pickle.load(open(fe_cluster_fname, 'rb'))
        else:
            print('Did not find feaure emb obj creating..')
            labels_cluster_obj = get_embedding_cluster_obj(processed_sentence_df.n_grams)
            pickle.dump(labels_cluster_obj, open(fe_cluster_fname, 'wb'))

    return processed_sentence_df, labels_cluster_obj, support_threshes


def extract_clusters(support_threshes, processed_sentence_df, labels_cluster_obj, max_features_frac, max_dup, with_negative):
    cfi = CFIEClustering(support_threshes, False)
    clusters = cfi.fit(processed_sentence_df, labels_cluster_obj, max_features_frac=max_features_frac)

    print('Input docs count :', len(processed_sentence_df))
    print(processed_sentence_df.head())
    print(processed_sentence_df.columns)
    print(processed_sentence_df.iloc[0])
    # print(clusters)

    print('\n\nPostprocessing extra labels ..')
    cfi.postprocessing_labels(max_dup=max_dup, keep_negative=with_negative)
    cluster_metrics = cfi.evaluate_clusters_quality()

    cluster_metrics['terms_count'] = len(cfi.tfidf_terms)

    #cluster_dir = '../../sample_data/clustering_results/' + corpus_name + '/'
    #cfi_filename = 'cfi_emb' if emb else 'cfi' + '_with_neg.pkl' if with_negative else '.pkl'

    #pickle.dump(cfi, open(cluster_dir + cfi_filename, 'wb'))

    return cfi, cluster_metrics

def run_multiple_experiments(corpus_name):
    #corpus_name = 'Reuters-text'
    # corpus_name = 'Reuters-headline'

    if corpus_name == 'Reuters-headline':
        max_dup = None
        max_features_frac_ops = [None]

    elif corpus_name == 'Reuters-text':
        max_dup = 2
        max_features_frac_ops = [None, .3, .2, .1, .05]

    else:
        print('Corpus name %s unknown..' % (corpus_name))
        sys.exit(0)

    with_negative_ops = [True, False]
    emb_ops = [None, 'Fasttext']

    #### TMP #####
    emb_ops = [None]
    with_negative_ops = [False]
    ####################

    for with_negative in with_negative_ops:
        for emb in emb_ops:
            for max_features_frac in max_features_frac_ops:
                processed_sentence_df, labels_cluster_obj, support_threshes = \
                    prepare_cluster_setup(corpus_name, emb)

                cluster_parameters = {'corpus_name': corpus_name, 'embedding': emb, 'negative_labels': with_negative, \
                                      'max_features_frac': max_features_frac, 'max_overlap_percent': max_dup}

                cfi_obj, cluster_metrics = extract_clusters(support_threshes, processed_sentence_df, labels_cluster_obj, \
                                                            max_features_frac, max_dup, with_negative)

                artifacts_map = {'cfi_obj': cfi_obj}
                print('labels_cluster_obj:', labels_cluster_obj)

                if labels_cluster_obj:
                    cluster_metrics['feature_count_before_embedding'] = len(labels_cluster_obj.tfidf_feature_names)
                    cluster_metrics['feature_clusters_count'] = len(labels_cluster_obj.leaders_map)
                    cluster_metrics['features_singular_count'] = len(labels_cluster_obj.singular_terms)
                    artifacts_map['feature_emb_obj'] = labels_cluster_obj

                log_mlflow_experiment(cluster_parameters, cluster_metrics, artifacts_map)


# %%
if __name__=='__main__':
    # support_threshes = {1: 30, 2: 10, 3: 5, 4: 3, 5: 3}
    max_n = 5
    support_threshes = {1: 1.5, 2: 0.9, 3: 0.4, 4: 0.2, 5: 0.1}
    fe_filename = 'feature_extracted_19961119_headline_df.pkl'  # feature_extracted_df.pkl
    # fe_filename = 'feature_extracted_19961119_text_df.pkl'

    processed_sentence_df = pickle.load(open('../../sample_data/processed_data/' + fe_filename, 'rb'))
    cluster_fname = 'feature_sb_emb_obj.pkl'
    label_cluster_obj = pickle.load(open('../../sample_data/clustering_results/Reuters-headline/' + cluster_fname, 'rb'))


    cfi = CFIEClustering(support_threshes)
    clusters = cfi.fit(processed_sentence_df, label_cluster_obj, None)

    print('\n\n')
    print('Evaluate Clustering quality..')
    cfi.evaluate_clusters_quality()
    print('\n\nEvaluate labelling quality..')
    cfi.evaluate_label_quality(2)

    print('\n\nPostprocessing extra labels ..')
    cfi.postprocessing_labels()

    print('\n\nEvaluating clustering coverage quality again..')
    cfi.evaluate_clusters_quality()

    pickle.dump(cfi, open('../../sample_data/clustering_results/Reuters-headline/cluster_sb_emb_obj.pkl', 'wb'))


    """
    print('Stats...')
    for level in cfi._all_freq_itemsets.keys():
        if len(cfi._all_freq_itemsets[level].values()) > 0:
            min_v, max_v = min(cfi._all_freq_itemsets[level].values()), max(cfi._all_freq_itemsets[level].values())
            print(min_v, max_v, min_v*100/max_v, min_v*100/194, min_v*100/1231)

        #print(level, min(cfi._all_freq_itemsets[level].values()), max(cfi._all_freq_itemsets[level].values()))
    """


    ## Add embedding approach and compare the results