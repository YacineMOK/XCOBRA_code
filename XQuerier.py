from cobras_ts.querier import Querier
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

class XQuerier(Querier):

    def __init__(self, labels, xai_method="shap", strat="commun_fraction", top_n=3, threshold = 0.5, test=False):
        super(XQuerier, self).__init__()
        self.labels = labels # ground truth
        self.top_n = top_n
        self.xai_method = xai_method
        self.strat = strat
        self.threshold = threshold
        self.test = test
        self.ml = []
        self.cl = []
        # for debug - list of explanations
        self.explanations_list = []
        self.queried_points = []

    def query_points(self, idx1, idx2, exp1=None, exp2=None):
        # print(f"idx1 {idx1} - idx2 {idx2}\nexp1 {exp1} - exp2 {exp2}")
        
        if exp1 == None or exp2 == None or self.strat=="ground_truth":
            # use ground truth
            print("Salut ")
            answer = True

            return answer

        self.queried_points.append((idx1, idx2))

        if self.xai_method == "shap":
            return self.query_points_shap(idx1, idx2, exp1, exp2)
        else:
            # lime
            return self.query_points_lime(idx1, idx2, exp1, exp2)

        

    # SHAP -related
    def query_points_shap(self, idx1, idx2, exp1=None, exp2=None):
        if self.strat=="exp_sim":
            # TODO
            ...

        # ---
        answer = None
        exp1_values = exp1.values
        exp2_values = exp2.values
        
        # sort the values
        ind_exp1 =  np.abs(exp1_values).argsort()[-self.top_n:][::-1]
        ind_exp2 =  np.abs(exp2_values).argsort()[-self.top_n:][::-1]

        # sort feature_names
        feature_names_exp1 = np.array(list(exp1.feature_names))[ind_exp1]
        feature_names_exp2 = np.array(list(exp2.feature_names))[ind_exp2]
        
        # commun/shared feature_names
        fi_intersection = set(feature_names_exp1).intersection(set(feature_names_exp2))
        commun_fraction = len(fi_intersection)

        if self.strat == "commun_fraction":
            answer = commun_fraction*1./self.top_n > self.threshold
        
        if self.strat == "cosine_similarity":
            answer = cosine_similarity(np.array([exp1.values]), np.array([exp2.values])) >= self.threshold 

        if self.strat == "euclidean_distance":
            # bounded between 0 and 2
            euc_dist = euclidean_distances(
                normalize(np.array([exp1.values])),
                normalize(np.array([exp2.values]))
            )[0][0]

            answer = (euc_dist * 0.5) <= (1 - self.threshold)


        if self.strat == "ndcg":
                # sort the values
            ind_exp1 =  np.abs(exp1_values).argsort()[-len(exp1_values):][::-1]
            ind_exp2 =  np.abs(exp2_values).argsort()[-len(exp2_values):][::-1]
                # sort feature_names
            feature_names_exp1 = np.array(list(exp1.feature_names))[ind_exp1]
            feature_names_exp2 = np.array(list(exp2.feature_names))[ind_exp2]
            
            e1, e2 = self.labels_to_int_ids(feature_names_exp1, feature_names_exp2)
            answer = ndcg_score([e1], [e2])

        if answer:
            self.ml.append((idx1, idx2))
        else:
            self.cl.append((idx1, idx2))
        

        # save explanation vectors
        self.explanations_list.append(
            (list(exp1.values), list(exp2.values))
        )
        return answer
        
    # LIME - related
    def query_points_lime(self, idx1, idx2, exp1=None, exp2=None):
        ...

        answer = None

        # Pourquoi ensemble ? - label 1
        exp1_label1 = exp1
        exp2_label1 = exp2

        if self.strat == "commun_fraction":
            feature_names_exp1 = set(exp1_label1[:self.top_n, 0])
            feature_names_exp2 = set(exp2_label1[:self.top_n, 0])
            fi_intersection = set(feature_names_exp1).intersection(set(feature_names_exp2))
            commun_fraction = len(fi_intersection)
            answer = commun_fraction*1./self.top_n > self.threshold

        # ordonner et
        exp1_sorted, exp2_sorted, _ = self.sort_lime_explanation(exp1_label1, exp2_label1)
        if self.strat == "cosine_similarity":
            answer = cosine_similarity(np.array([exp1_sorted]), np.array([exp2_sorted])) >= self.threshold 
        
        if self.strat == "euclidean_distance":
            # bounded between 0 and 2
            euc_dist = euclidean_distances(
                normalize(np.array([exp1_sorted])),
                normalize(np.array([exp2_sorted]))
            )[0][0]

            answer = (euc_dist * 0.5) <= (1 - self.threshold)

        # sans ordonner, choisir un element comme une base et l'autre
        if self.strat == "ndcg":
            label_rank_data1 = exp1_label1[:, 0]
            label_rank_data2 = exp2_label1[:, 0]
            e1, e2 = self.labels_to_int_ids(label_rank_data1, label_rank_data2)
            answer = ndcg_score([e1], [e2]) >= self.threshold 

        if answer:
            self.ml.append((idx1, idx2))
        else:
            self.cl.append((idx1, idx2))
        

        # save explanation vectors
        self.explanations_list.append(
            (list(exp1_sorted), list(exp2_sorted))
        )
        return answer
        
    def sort_lime_explanation(self, exp1, exp2):
        """Order the two lime explanations according to the first explanation

        Args:
            exp1 (np.array): first  lime explanation np.asanyarray()
            exp2 (np.array): second lime explanation np.asanyarray()

        Returns:
            np.array: sorted_exp1_values 1D
            np.array: sorted_exp2_values 1D
            list:     corresponding feature order
        """
        res_exp1, res_exp2 = [], []
        exp1, exp2 = dict(exp1), dict(exp2)
        for idx in exp1.keys():

            res_exp1.append(exp1[idx])
            try: 
                res_exp2.append(exp2[idx])
            except:
                print(idx)
                print(list(exp1.keys()))
                print(list(exp2.keys()))
        return np.array(res_exp1), np.array(res_exp2), list(exp1.keys())
    

    def labels_to_int_ids(self, arr1, arr2):
        id = 0
        dico_str_to_id = {}
        ids_arr1, ids_arr2 = [], []
        # ARRAY 1
        for key in arr1:
            if not (key in list(dico_str_to_id.keys())):
                dico_str_to_id[key] = id
                ids_arr1.append(id)
                id+=1
            else:
                ids_arr1.append(dico_str_to_id[key])
        # ARRAY 2
        for key in arr2:
            # normalement on aura vu toutes les features, sinon, erreur
            ids_arr2.append(dico_str_to_id[key])
        
        return ids_arr1, ids_arr2

    # def labels_to_ids(arr):
    #     id = 0
    #     dico_str_to_id = {}
    #     dico_id_to_str = {}
    #     for key in arr:
    #         if not (key in arr):
    #             dico_str_to_id[key] = id
    #             dico_id_to_str[id] = key
    #             id+=1
    #     return dico_str_to_id, dico_id_to_str

    def get_pts_exps(self):
        return self.queried_points, self.explanations_list
    
    def query_quick_shap(self, pt1 = None, pt2 = None, exp1 = None, exp2 = None):
        answer = None

        if self.strat == "cosine_similarity":
            answer = cosine_similarity(np.array([exp1]), np.array([exp2])) >= self.threshold 

        if self.strat == "euclidean_distance":
            # bounded between 0 and 2
            euc_dist = euclidean_distances(
                normalize(np.array([exp1])),
                normalize(np.array([exp2]))
            )[0][0]

            answer = (euc_dist * 0.5) <= (1 - self.threshold)

        return answer