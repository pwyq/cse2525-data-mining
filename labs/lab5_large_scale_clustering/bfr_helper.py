from abc import ABC, abstractmethod
import sys

import numpy as np


class Cluster(ABC):
    """
    Contains all the data structures for hierarchical clustering.

    Properties:
    self.sum: np.ndarray: The sum of the summary..
    self.squared_sum: np.ndarray: The squared sum of the summary..
    self.n_data_points: int: The amount of data points summarized in this cluster.
    self.cluster_ids: List[uuid]: The ids of the data points in this cluster.
    self.self.cluster_id: uuid: The id of this cluster.

    self.mean: np.ndarray: The mean of the summary.
    self.variance: np.ndarray: The variance of the summary.
    self.std: np.ndarray: The std of the summary.
    """
    def __init__(self, sum_v, squared_sum, n_data_points, cluster_ids, mean, variance, std):
        assert n_data_points > 0
        assert len(cluster_ids) > 0
        self.sum = sum_v
        self.squared_sum = squared_sum
        self.n_data_points = n_data_points
        self.cluster_ids = cluster_ids
        self.cluster_id = next(iter(cluster_ids))

        self.mean = mean
        self.variance = variance
        self.std = std

    def distance(self, other):
        """
        Parameters:
        other: Cluster: The cluster we are intrested in.

        Returns:
        float: The abs distance between the centroids of 2 clusters.
        """
        return np.linalg.norm(self.mean - other.mean)

    def add(self, dp):
        """
        Creates a new cluster where by the datapoint has been added.
        The function is stateless.

        Parameters:
        dp: DataPoint: The DataPoint we are intrested in.

        Returns:
        Cluster: The new cluster.
        """
        return self.merge(dp.to_singleton_cluster())

    def merge(self, other):
        """
        Merges 2 clusters in a stateless manner.

        Parameters:
        other: Cluster: The cluster we want to merge with.

        Returns:
        Cluster: A new cluster that is merger of the input clusters.
        """
        sum_v = self.sum + other.sum
        squared_sum = self.squared_sum + other.squared_sum
        n_data_points = self.n_data_points + other.n_data_points
        cluster_ids = self.cluster_ids.union(other.cluster_ids)

        return self.__class__(sum_v=sum_v, squared_sum=squared_sum, n_data_points=n_data_points, cluster_ids=cluster_ids)

    def contains(self, dp):
        """
        Pre condition: The datapoint has been assigned to a cluster.

        Parameters:
        dp: DataPoint: The datapoint we are intrested in.

        Returns:
        bool: Returns True if the data point has been assinged to this cluster.
        """
        assert dp.cluster_id is not None
        return dp.cluster_id in self.cluster_ids

    def __repr__(self):
        return f"BFRCluster(mean: {self.mean},\n variance: {self.variance}\n)\n"

    @abstractmethod
    def mahalanobis_distance(self, dp):
        """
        Parameters:
        dp: DataPoint: The DataPoint we are intrested in.

        Returns:
        float: The mahalanobis distance between the centroids of this cluster and a datapoint
        """
    pass

    @abstractmethod
    def is_data_point_sufficiently_close(self, dp):
        """
        Parameters:
        dp: DataPoint: The DataPoint we are intrested in.

        Returns:
        bool: True iff the mahalanobis distance is less than 3 times the std on atleast one axis.
        """
    pass


def load_data(file_path, chunk_size, create_data_point_func):
    """
    Load the data from disk.
    Simulates limited memory size by putting the data into chunks.

    Parameters:
    file_path str: The path to the file.
    chunk_size: int: The chunk size of the data
    create_data_point_func: Callable[[np.ndarray], DataPoint] transforms a vector into a data point

    Returns:
    List[List[DataPoint]]: Returns the datapoints as a list of chunks.
    """
    lines = []
    with open(file_path) as f:
        lines = [line.rstrip("\n") for line in f]

    data = []
    chunk = []
    for line in lines:
        elements = line.split(" ")
        vector = np.array([[float(el) for el in elements]])
        chunk.append(create_data_point_func(vector))
        if len(chunk) >= chunk_size:
            data.append(chunk)
            chunk = []
    return data


def hierarchical_clustering(data, k):
    """
    Preforms hierarchical clustering on the data set.
    Is stateless and does not mutate the original input list.

    Parameters:
    data List[Cluster]: The K clusters  we are intrested in.
    k: int: The amount of cluster to return

    Returns:
    List[Cluster]: The K clusters. If the data contains less than K datapoints it returns the orignal list.
    """

    if len(data) <= k:
        return data

    res = data.copy()

    for _ in range(len(res) - k):
        i, j = find_two_closest(res)
        merged = res[i].merge(res[j])
        res[i] = merged
        del res[j]

    return res


def find_two_closest(data):
    """
    Finds the indexes of two closest clusters.

    Parameters:
    data List[Cluster]: The K clusters  we are intrested in.

    Returns:
    Tuple[int, int]:Returns indices of the two clusters.
    """
    min_one_i = None
    min_other_j = None
    min_dist = sys.maxsize
    for i, one, in enumerate(data):
        for j in range(i + 1, len(data)):
            other = data[j]
            dist = one.distance(other)
            if dist < min_dist:
                min_one_i = i
                min_other_j = j
                min_dist = dist

    assert min_one_i is not None, "We need to find a value for min_one_i"
    assert min_other_j is not None, "We need to find a value for min_one_j"
    assert min_one_i != min_other_j, f"min_one_i: {min_one_i} cannot be equal to min_other_j: {min_other_j}"

    return min_one_i, min_other_j