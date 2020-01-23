import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import trange


class Cluster(object):
    """
    Con
    """

    def __init__(self, data):
        self.data: np.ndarray = data
        self.centroid: np.ndarray = find_centroid(data)

    def data_points_as_cluster(self):
        """
        Transforms the data in this cluster into a list of singleton clusters.
        This function is stateless.


        Returns:
        List[Cluster]: A list of singleton clusters.
        """

        return [Cluster(np.array([dp])) for dp in self.data]

    def merge(self, other):
        """
        Creates a new cluster that is a merger of this and the other cluster.
        This function is stateless.

        Parameters:
        other: Cluster: The cluster we are intrested in.

        Returns:
        float: A new cluster.
        """

        data = np.vstack((self.data, other.data))
        return Cluster(data)

    def distance(self, other, distance_measure="closests_point"):
        """
        Calculates the distances between this cluster and an other.
        This method will be used for the traditional clusting method.

        Parameters:
        other: Cluster: The cluster we are intrested in.
        distance_measure: ["closests_point", "mean_sqaured_distance"]: The type of distance measure to use.

        Returns:
        float: Returns the distance as a float.
        """

        if distance_measure == "closests_point":
            # This method returns the distance to the closests point in the stored data.
            return np.min(cdist(self.data, other.data, metric="euclidean"))
        elif distance_measure == "mean_sqaured_distance":
            # This methods returns the msqd between the centroids.
            return np.linalg.norm(self.centroid - other.centroid)
        else:
            raise Exception(f"Unknown distance_measure='{distance_measure}'. See documentation for valid values.")

        # Student end

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Cluster(\ncentroid: {self.centroid},\n data: {self.data}\n)\n"


def find_centroid(data):
    """
    finds the centroid of the n x m matrix

    Parameters:
    data: np.ndarray: An n x m matrix.

    Returns:
    np.ndarray: Returns a 1 x m matrix.
    """

    assert len(data.shape) == 2, "We assume that the data is 2d matrix of data points."
    return np.mean(data, axis=0)


def load_data(file_path):
    """
    Load the data from disk.

    Parameters:
    file_path str: The path to the file.

    Returns:
    List[Cluster]: Returns the datapoints as a list of singleton clusters.
    """

    lines = []
    with open(file_path) as f:
        lines = [line.rstrip("\n") for line in f]

    clusters = []
    for line in lines:
        elements = line.split(" ")
        vector = np.array([[float(el) for el in elements]])
        clusters.append(Cluster(vector))
    return clusters


def plot_clusters(clusters):
    """
    Plots a Clusters

    Parameters:
    clusters: List[Cluster]: A list of clusters.
    """

    data = [c.data for c in clusters]
    plot_data(data)
    plt.show()


def plot_data(data, marker="o"):
    """
    Plots a list of 1x2 datapoints.

    Parameters:
    data: List[np.ndarray]: A list of 1x2 datapoints.
     marker: str: The marker type used to display this data.

    """
    k = len(data)
    colors = cm.rainbow(np.linspace(0, 1, k))

    for dp, color in zip(data, colors):
        x = dp[:, 0]
        y = dp[:, 1]
        plt.scatter(x, y, marker=marker, c=[color])


def plot_cure_clusters(cure_clusters):
    """
    Plots the cure clusters.

    Parameters:
    cure_clusters: List[CureCluster]: The cure clusters we want to plot.
    """
    data = [c.data.data for c in cure_clusters]
    k_most_representative_points = [c.k_most_representative_points for c in cure_clusters]

    # Plot the assigned data points
    plot_data(data, marker=".")
    # Plot the K representative points
    plot_data(k_most_representative_points, marker="D")
    plt.show()


def hierarchical_clustering(data, k, distance_measure):
    """
    Performs hierarchical clustering on the data set.
    Is stateless and does not mutate the original input list.

    Parameters:
    data List[Cluster]: The K clusters  we are intrested in.
    k: int: The amount of cluster to return
    distance_measure: ["closests_point", "mean_sqaured_distance"]: The type of distance measure to use.

    Returns:
    List[Cluster]: The K clusters. If the data contains less than K datapoints it returns the orignal list.
    """

    assert 0 <= k <= len(data), f"k={k} while len(data)={len(data)}"
    if len(data) <= k:
        return data

    res = data.copy()

    for _ in trange(len(res) - k):
        i, j = find_two_closest(res, distance_measure)
        merged = res[i].merge(res[j])
        res[i] = merged
        del res[j]

    return res


def find_two_closest(data, distance_measure):
    """
    Finds the indexes of two closest clusters.

    Parameters:
    data List[Cluster]: The K clusters  we are intrested in.
    distance_measure: ["closests_point", "mean_sqaured_distance"]: The type of distance measure to use.

    Returns:
    Tuple[int, int]:Returns indices of the two clusters.
    """
    min_one_i = None
    min_other_j = None
    min_dist = sys.maxsize
    for i, one, in enumerate(data):
        for j in range(i + 1, len(data)):
            other = data[j]
            dist = one.distance(other, distance_measure)
            if dist < min_dist:
                min_one_i = i
                min_other_j = j
                min_dist = dist

    assert min_one_i is not None, "We need to find a value for min_one_i"
    assert min_other_j is not None, "We need to find a value for min_one_j"
    assert min_one_i != min_other_j, f"min_one_i: {min_one_i} cannot be equal to min_other_j: {min_other_j}"

    return min_one_i, min_other_j