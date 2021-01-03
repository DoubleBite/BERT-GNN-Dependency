import numpy as np
from Levenshtein import distance, ratio


def find_closest_candidate(target, candidates):
    distances = [distance(target, x) for x in candidates]
    return candidates[np.argmin(distances, axis=0)]
