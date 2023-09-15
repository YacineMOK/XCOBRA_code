from cobras_ts.querier import Querier
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np

class DQuerier(Querier):

    def __init__(self, strat="cosine_similarity", threshold = 0.5, test=False):
        super(DQuerier, self).__init__()
        self.strat = strat
        self.threshold = threshold
        self.test = test
        self.answers = [] # keep track of the answers: triplets (id1, id2, True/False) 
                          # : order = order of occurence
                          # : True=must link | False=cannot link  
        self.ml = []
        self.cl = []


    def query_points(self, idx1, idx2, data1=None, data2=None):
        
        #here data1 & data2 are the raw points and not explanations

        answer = None

        # print("data1---------", data1)
        if self.strat == "cosine_similarity":
            # print("please", cosine_similarity(np.array([data1]), np.array([data2])))
            answer = cosine_similarity(np.array([data1]), np.array([data2]))[0][0] >= self.threshold
            # print(f"Voici le answer: {answer}")
            self.answers.append((idx1, idx2, answer))
            # print(f"Voici le answer: {answer}")
            if answer:
                self.ml.append((idx1, idx2))
            else:
                self.cl.append((idx1, idx2))

        if self.strat == "euclidean_distance":
            # bounded between 0 and 2
            euc_dist = euclidean_distances(
                normalize(np.array([idx1])),
                normalize(np.array([idx2]))
            )

            answer == (euc_dist - 1) >= self.threshold        

        return answer

        
    def getAnswers(self):
        return self.answers
