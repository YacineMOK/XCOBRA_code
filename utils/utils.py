from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, v_measure_score, jaccard_score, adjusted_mutual_info_score

def read_arff_dataset(dataset_path):
    """function that reads arff files.

    Args:
        dataset_path (str): path of the file/dataset

    Returns:
        pandas.DataFrame: dataset
    """
    temp_data = arff.loadarff(open(dataset_path, 'r'))
    dataset = pd.DataFrame(temp_data[0])
    try:
        dataset["class"] = dataset["class"].str.decode('utf-8') 
    except KeyError:
        dataset["Class"] = dataset["Class"].str.decode('utf-8') 
        dataset["class"] = dataset["Class"]
        dataset = dataset.drop(["Class"], axis=1)
    return dataset




def cluster_analysis(intermediate_clusterings, y, runtimes=None):
    """Function that stores some external measures, number of clusters (and optionnaly, runtimes) of a clustering algorithm.

    Args:
        intermediate_clusterings (list): intermediate clustering of a clustering algorithm
        runtimes (list, optional): list of execution times of each "iteration". Defaults to None.

    Returns:
        NumClusters: Number of clusters
        ARIs: adjusted_rand_score
        AMIs: adjusted_mutual_info_score
        VMSs: v_measure_score
        exec_time: runtimes
    """
    NumClusters = np.array([len(set(tmp_clustering)) for tmp_clustering in intermediate_clusterings], dtype=float)
    ARIs = np.array([adjusted_rand_score(tmp_clustering,y) for tmp_clustering in intermediate_clusterings], dtype=float)
    AMIs = np.array([adjusted_mutual_info_score(tmp_clustering,y) for tmp_clustering in intermediate_clusterings], dtype=float)
    VMSs = np.array([v_measure_score(tmp_clustering,y) for tmp_clustering in intermediate_clusterings], dtype=float)
    exec_time = np.array([0]*len(intermediate_clusterings), dtype=float)
    if runtimes:
        exec_time += np.array(runtimes, dtype=float)
    return NumClusters, ARIs, AMIs, VMSs, exec_time



def save_results_in_dataframe(dataname, budget, intermediate_clusterings, y, sub_path="ground_truth", additional=None):
    # print("\t...Saving the results...")
    # path = "./results/" + sub_path + "/"
    data = {
        'budget': [],
        'strat' : [],
        'number clusters' : [],
        'ARI': []
    }
    for i in range(0, budget):
        tmp_clustering = intermediate_clusterings[i-1]
        tmp_ari = adjusted_rand_score(tmp_clustering,y)
        
        data['budget'].append(i)                                 # VAR
        data['strat'].append(sub_path)                       # cst
        data['number clusters'].append(len(set(tmp_clustering))) # VAR
        data['ARI'].append(tmp_ari)                              # VAR

    return pd.DataFrame(data=data)


def ground_truth_accuracy(y, ml, cl, budget, number_trial):
    """
    Computes the "correct" response rate of an algorithm (e.g. top-n, cosine similarity, ... etc)
    
    Args:
        y (np.array): ground truth labels
        ml (list): must links
        cl (list): cannot links
        budget (int): budget of the Active Clustering algorithm
        number_trial (int): should be 

    Returns:
        float: percentage of correct answers
    """
    expected_fp_fn = 0
    for (m,c) in zip(ml, cl):
        i = 0
        for (x1, x2) in m: # for all the must links over the trials
            if y[x1] == y[x2]:
                i+=1
        for (x1, x2) in c: # for all the cannot links over the trials
            if y[x1] != y[x2]:
                i+=1
        expected_fp_fn+=i
    return (expected_fp_fn*1./number_trial)


def one_run_ground_truth_accuracy(y, ml, cl):
    """_summary_

    Args:
        y (_type_): _description_
        ml (_type_): _description_
        cl (_type_): _description_
    """
    expected_fp_fn = 0
    for (a,b) in ml:
        if y[a] == y[b]:
          expected_fp_fn+=1
    for (a,b) in cl:
        if y[a] != y[b]:
          expected_fp_fn+=1
    return (expected_fp_fn*1./(len(ml)+len(cl)))*100