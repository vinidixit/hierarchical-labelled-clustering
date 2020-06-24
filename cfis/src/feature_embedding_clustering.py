import fasttext
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import collections
import pickle
import time


class FeatureEmbeddingClustering:
    def __init__(self, tokenized_texts, thresh_pairs, emb_name='fasttext'):

        assert emb_name in ['fasttext','sentbert'], 'Only fasttext and sentbert support is available at this moment.'
        assert all(len(p)==2 for p in thresh_pairs), 'Provide pair of thresholds for leader and member assignment.'

        print('Embedding chosen :', emb_name)
        # Cluster input preparation
        self.matrix, self.tfidf_feature_names = get_tfidf_matrix(tokenized_texts, min_df=1, max_df=1.0)

        self.feat_embeds = _get_embeddings(self.tfidf_feature_names, emb_name)
        self.feat_embed_map = dict(zip(self.tfidf_feature_names, self.feat_embeds))

        # cluster parameters
        self.n_passes = len(thresh_pairs)
        self.thresh_pairs = thresh_pairs

        # cluster variables
        self.leaders_map = collections.defaultdict(set)
        self.member_leader_map = collections.defaultdict(set)
        self.singular_terms = []

    def fit(self, debug_terms=None):
        features_sorted = sorted(list(self.tfidf_feature_names), key=lambda x: self._get_df_score(x), reverse=True)
        print('\nTotal initial singleton features:', len(self.tfidf_feature_names))

        t0 = time.time()
        for i in range(self.n_passes):
            t1 = time.time()

            print('\n\nPass....:', i + 1)
            leader_thresh, member_thresh = self.thresh_pairs[i]

            self._cluster_fit(features_sorted, leader_thresh, member_thresh, debug_terms)
            print('done in ', round((time.time()-t1)/60, 4), ' mins.')

        print('\n\nPostprocessing.....')
        self._postprocessing(features_sorted, 0.7, debug_terms)
        print('Total time taken in fitting: ', round((time.time() - t0) / 60, 4), ' mins.\n')


    def transform(self, tokenized_texts):
        all_tokens_emb = []

        for text_tokens in tokenized_texts:
            tokens_emb = []

            for term in text_tokens:
                if term in self.member_leader_map and len(self.member_leader_map[term])==1: # use only "clear" leaders
                    tokens_emb.append(list(self.member_leader_map[term])[0])
                else:
                    assert term in self.member_leader_map or term in self.singular_terms or term in self.leaders_map,\
                                                                                    'Error {} not found'.format(term)
                    tokens_emb.append(term)

            all_tokens_emb.append(tokens_emb)

        return all_tokens_emb


    def _cluster_fit(self, features_sorted, leader_thresh, member_thresh, debug_terms=None):

        for term in features_sorted:
            if term in self.leaders_map or term in self.member_leader_map:
                continue

            if debug_terms and term in debug_terms:
                print('\nFinding cluster for :', term)

            nearest_neighbors, scores = self._get_nearest_neighbor(term, leader_thresh, True)

            if len(nearest_neighbors) == 0:
                if debug_terms and term in debug_terms:
                    print(term, ': No nearest neighbor found with thresh: ', leader_thresh, '\n')
                continue

            leader, member = None, term

            leader = self._try_existing_grps(term, nearest_neighbors, member_thresh, debug_terms)

            if leader is None and len(nearest_neighbors) > 0:
                leader = self._try_new_cluster(term, nearest_neighbors, debug_terms)  # for previous unassigned cands

            if leader is not None:
                self.leaders_map[leader].add(member)
                self.member_leader_map[member].add(leader)

                if debug_terms and term in debug_terms:
                    print('\nAssignment: ', term, ': ', leader, '<-', member, '\n')

        multiple_leader_assignments = {k: v for k, v in self.member_leader_map.items() if len(v) > 1}

        if len(multiple_leader_assignments) > 0:
            print('\nMultiple leaders found:', multiple_leader_assignments)

        self.singular_terms = [term for term in features_sorted if (term not in self.leaders_map and term not in self.member_leader_map)]

        print('\nPass done: Singular terms: ', len(self.singular_terms))

       # return leaders_map, member_leader_map, singular_terms


    def _try_existing_grps(self, term, nearest_neighbors, member_thresh, debug_terms=None):

        leader = None
        if debug_terms and term in debug_terms:
            print(term, ': Trying existing clusters..')

        for nn in nearest_neighbors:
            if leader is not None:
                break

            matching_cand = None

            if nn in self.member_leader_map:  # already has a leader
                # print(nn, ' has a leader :', member_leader_map, leaders_map)
                nn_leader = list(self.member_leader_map[nn])[0]

                if nn_leader in nearest_neighbors:  # has leader match with term
                    matching_cand = nn_leader

            elif nn in self.leaders_map:
                # print(nn, 'is a leader :', leaders_map)
                matching_cand = nn

            else:  ## nn does not exist in existing clusters
                pass

            if matching_cand:
                existing_grp = self.leaders_map[matching_cand]
                if self._check_grp_membership(existing_grp, term, member_thresh):
                    leader = matching_cand

            if debug_terms and term in debug_terms:
                if leader:
                    print(term, ': Got membership in cluster :', leader, '<-', existing_grp)

                elif matching_cand:
                    print(term, ': Membership not possible :', matching_cand, '<-', existing_grp)
                else:
                    print(term, ': No matching leader candidate found with :', nn)

        return leader

    def _try_new_cluster(self, term, nearest_neighbors, debug_terms=None):
        term_score = self._get_df_score(term)
        leader = None
        if debug_terms and term in debug_terms:
            print(term, ': Trying new cluster..')

        for neighbor in nearest_neighbors:
            if leader is not None:
                break

            if neighbor in self.member_leader_map or neighbor in self.leaders_map:  # already got tested and rejected by it
                continue

            neighbor_score = self._get_df_score(neighbor)

            if neighbor_score >= term_score:  # assign a new leader
                leader = neighbor

        if debug_terms:
            if leader and term in debug_terms:
                print('New leader Assignment: ', term, ": ", leader)
            elif term in debug_terms:
                print('No new leader assigned: ', term)

        return leader

    def _postprocessing(self, features_sorted, thresh=0.7, debug=False):

        for term in features_sorted:
            if term in self.leaders_map or term in self.member_leader_map:
                continue

            nearest_neighbors = self._get_nearest_neighbor(term, thresh)

            if len(nearest_neighbors) == 0:
                continue

            if debug:
                print('Term: ', term)

            term_score = self._get_df_score(term)
            matching = None

            for nn in nearest_neighbors:
                if matching:
                    break

                if debug:
                    print('Nearest neighbor: ', nn)

                if nn in self.leaders_map:  # whose members are not aggreable with term
                    if term_score >= self._get_df_score(nn):  # term can become new leader
                        matching = nn


                elif nn in self.member_leader_map:  # either will become a leader or get one more leader
                    matching = nn

                if matching:
                    leader_str = 'leader' if matching in self.leaders_map else ''
                    member_str = 'member' if matching in self.member_leader_map else ''

                    if debug:
                        print(term, ': matching found in existing cluster..', matching, leader_str, member_str)

            if not matching:
                matching = nearest_neighbors[0]
                leader_str = 'leader' if matching in self.leaders_map else ''
                member_str = 'member' if matching in self.member_leader_map else ''

                if debug:
                    print('Match not found, assigning nearest neighbor: ', matching, leader_str, member_str)

            nn_score = self._get_df_score(matching)
            if term_score >= nn_score:
                leader, member = term, matching
            else:
                leader, member = matching, term

            self.leaders_map[leader].add(member)
            self.member_leader_map[member].add(leader)
            if debug:
                print(term, ': ', leader, '<-', member, '\n')

        multiple_leader_assignment = {k: v for k, v in self.member_leader_map.items() if len(v) > 1}
        if len(multiple_leader_assignment) > 0:
            print('\nMultiple leaders assignment found: ', multiple_leader_assignment)

        self.singular_terms = [term for term in features_sorted if (term not in self.leaders_map and term not in self.member_leader_map)]

        print('\nPostprocessing done: Singular terms: ', len(self.singular_terms))


    def _get_nearest_neighbor(self, term, thresh=0.8, with_score=False):
        feature_names = np.array(self.tfidf_feature_names)
        feat_embed_map = self.feat_embed_map
        feat_embeds = self.feat_embeds

        cosine_scores = cosine_similarity([feat_embed_map[term]], feat_embeds)[0]

        term_index = list(feature_names).index(term)
        cosine_scores[term_index] = -10

        selected_indices = np.where(cosine_scores > thresh)[0]
        selected_scores = cosine_scores[selected_indices]
        selected_features = feature_names[selected_indices]

        sorted_indices = np.argsort(selected_scores)[::-1]
        scores = selected_scores[sorted_indices]
        selected_features = selected_features[sorted_indices]

        if with_score:
            return selected_features, scores

        """
        sorted_indices = np.argsort(cosine_scores)[::-1]
        sorted_indices = [ind for ind in sorted_indices if cosine_scores[ind] > thresh]
        scores = cosine_scores[sorted_indices]
        feature_names = feature_names[sorted_indices]

        if with_score:
            return feature_names, scores

        return feature_names
        """
        return selected_features

    def _check_grp_membership(self, existing_grp, new_member, thresh=0.8, debug=False):
        feat_embed_map = self.feat_embed_map

        scores = cosine_similarity([feat_embed_map[new_member]], [feat_embed_map[m] for m in existing_grp])[0]
        if debug:
            print("checking grp membership for: ", new_member, '->', existing_grp, ': ', np.min(scores), thresh)

        if np.min(scores) < thresh:
            return False
        return True

    def _get_df_score(self, term):
        feature_names = self.tfidf_feature_names
        matrix = self.matrix

        t_index = list(feature_names).index(term)
        tfidf_scores = matrix.toarray()[:, t_index]
        df_score = len(tfidf_scores[tfidf_scores > 0])
        return df_score

def _get_embeddings(feature_names, model_name):
    if model_name == 'fasttext':
        emb_model = __get_fasttext_model()
        print('Encoding using fasttext..')
        t0 = time.time()
        feat_embeds = [emb_model.get_sentence_vector(f.replace('_', ' ')) for f in feature_names]
        print('done. ', round((time.time() - t0) / 60, 4), ' min')

    elif model_name == 'sentbert':
        sb_model = __get_sentbert_model()
        clean_features = [f.replace('_', ' ') for f in feature_names]
        t0 = time.time()
        print('Encoding using sentbert..')
        feat_embeds = sb_model.encode(clean_features)
        print('done. ', round((time.time()-t0)/60, 4) , ' min')

    return feat_embeds

def __get_fasttext_model():

    ftmodel_path = '/Users/vdixit/fasttext_models/'
    ft_model = fasttext.load_model(ftmodel_path + 'cc.en.300.bin')
    return ft_model

def __get_sentbert_model():
    sb_model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
    return sb_model


from _utils import get_tfidf_words, get_tfidf_matrix

def get_embedding_cluster_obj(tokenized_texts, emb_name, num_passes=4):

    leader_threshes = [0.8, 0.7, 0.6, 0.5]
    member_threshes = [0.7, 0.6, 0.5, 0.4]
    thresh_pairs = list(zip(leader_threshes, member_threshes))

    if 0 < num_passes < 4:
        thresh_pairs = thresh_pairs[:num_passes]

    emb_cl = FeatureEmbeddingClustering(tokenized_texts, thresh_pairs, emb_name)
    emb_cl.fit()
    return emb_cl

if __name__=='__main__':
    #fe_filename = 'feature_extracted_19961119_headline_df.pkl' # feature_extracted_df.pkl
    fe_filename = 'feature_extracted_19961119_text_df.pkl'

    processed_sentence_df = pickle.load(open('../../sample_data/processed_data/'+fe_filename, 'rb'))
    print(processed_sentence_df.columns)
    tfidf_matrix, terms = get_tfidf_matrix(processed_sentence_df.n_grams, min_df=1, max_df=1.0)

    leader_threshes = [0.8, 0.7, 0.6, 0.5]
    member_threshes = [0.7, 0.6, 0.5, 0.4]
    thresh_pairs = list(zip(leader_threshes, member_threshes))

    debug_terms = ['say', 'say_statement', 'significant', 'difference', 'major', 'serious', 'substantial']
    emb_cl = FeatureEmbeddingClustering(processed_sentence_df.n_grams, thresh_pairs, 'fasttext')

    emb_cl.fit()


    print('\nClusters..:')
    for label, members in emb_cl.leaders_map.items():
        print(label, len(members), ': ', members)

    """
    print('\nMembers-leaders mapping..:')
    for label, members in emb_cl.member_leader_map.items():
        print(label, len(members), ': ', members)
    """

    print('\n\nTotal features: ', len(terms))
    print('Clustered features count :', len(emb_cl.leaders_map) + len(emb_cl.member_leader_map))
    print('Singular features count: ', len(emb_cl.singular_terms),'\n\n')

   # print('total_cost' in emb_cl.singular_terms, 'total_cost' in emb_cl.leaders_map, 'total_cost' in emb_cl.member_leader_map)
   # print('total_cost' in terms)

  #  cluster_fname = 'feature_sb_emb_obj.pkl'
  #  pickle.dump(emb_cl, open('../../sample_data/clustering_results/Reuters-headline/' + cluster_fname, 'wb'))

    """
    print('\n\nSingular Terms...')
    for term in emb_cl.singular_terms:
        n, scores = emb_cl._get_nearest_neighbor(term, 0.6, True)
        print(term, ': ', (n, scores) if len(n)>0 else '')
    """