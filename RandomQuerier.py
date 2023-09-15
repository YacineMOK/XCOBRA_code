import random
from cobras_ts.querier import Querier

class RandomQuerier(Querier):

    def __init__(self):
        super(RandomQuerier, self).__init__()

    def query_points(self, idx1, idx2):
        return random.choices([True, False], [50, 50])[0]
