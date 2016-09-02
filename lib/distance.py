from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


def compute_sigma(train):
    dist = euclidean_distances(train, train)
    upper_indices = np.triu_indices_from(dist, 1)
    return np.median(dist[upper_indices])


def main():
    train = [[1,2],[2,3],[3,4]]
    print compute_sigma(train)

if __name__ == '__main__':
    main()
