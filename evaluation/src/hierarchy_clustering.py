# Copyright (c) 2022 Panagiotis Anagnostou

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Implementation of the clustering algorithms, members of the HiPart package.

@author Panagiotis Anagnostou
@author Nicos Pavlidis
"""

import HiPart.__utility_functions as util
import numpy as np
import statsmodels.api as sm

from HiPart.__partition_class import Partition
from sklearn.cluster import KMeans
from treelib import Tree


class BisectingKmeans(Partition):
    """
    Class BisectingKmeans. It executes the bisecting k-Means algorithm.

    References
    ----------
    Savaresi, S. M., & Boley, D. L. (2001, April). On the performance of
    bisecting K-means and PDDP. In Proceedings of the 2001 SIAM International
    Conference on Data Mining (pp. 1-14). Society for Industrial and Applied
    Mathematics.

    Parameters
    ----------
    max_clusters_number : int, (optional)
        Desired maximum number of clusters for the algorithm.
    min_sample_split : int, (optional)
        The minimum number of points needed in a cluster for a split to occur.
    random_state : int, (optional)
        The random seed fed in the k-Means algorithm.

    Attributes
    ----------
    output_matrix : numpy.ndarray
        Model's step by step execution output.
    labels_ : numpy.ndarray
        Extracted clusters from the algorithm.
    tree : treelib.Tree
        The object which contains all the information about the execution of
        the bisecting k-Means algorithm.
    samples_number : int
        The number of samples contained in the data.
    fit_predict(X) :
        Returns the results of the fit method in the form of the labels of the
        predicted clustering labels.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns. If the distance_matrix is True then X should be a square
            distance matrix.

        Returns
        -------
        labels_ : numpy.ndarray
            Extracted clusters from the algorithm.

    """

    decreasing = True

    def __init__(self, max_clusters_number=100, min_sample_split=5, random_state=None):
        super().__init__(
            max_clusters_number=max_clusters_number,
            min_sample_split=min_sample_split,
        )
        self.random_state = random_state

    def fit(self, X):
        """
        Execute the BisectingKmeans algorithm and return all the execution
        data in the form of a BisectingKmeans class object.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns.

        Returns
        -------
        self
            A BisectingKmeans class type object, with complete results on the
            algorithm's analysis.

        """
        self.X = X
        self.samples_number = X.shape[0]

        # create an id vector for the samples of X
        indices = np.array([int(i) for i in range(np.size(self.X, 0))])

        # initialize tree and root node                         # step (0)
        tree = Tree()
        # nodes` unique IDs indicator
        self.node_ids = 0
        # nodes` next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        tree.create_node(
            tag="cl_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.cluster_color),
        )
        # indicator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data
        if not tree.get_node(0).data["split_permission"]:
            print("cannot split at all")
            return self

        # Initialize the ST1 stopping criterion counter that count the number
        # of clusters.
        found_clusters = 1
        while (selected_node is not None) and (
            found_clusters < self.max_clusters_number
        ):
            self.split_function(tree, selected_node)  # step (1)

            # select the next kid for split based on the local minimum density
            selected_node = self.select_kid(
                tree.leaves(), decreasing=self.decreasing
            )  # step (2)
            found_clusters = found_clusters + 1  # (ST1)

        self.tree = tree
        return self

    def split_function(self, tree, selected_node):
        """
        Split the indicated node by clustering the data with a binary k-means
        clustering algorithm.

        Because python passes by reference data this function doesn't need a
        return statement.

        Parameters
        ----------
        tree : treelib.tree.Tree
            The tree build by the BisectingKmeans algorithm, in order to
            cluster the input data.
        selected_node : int
            The numerical identifier for the tree node that i about to be split.

        Returns
        -------
            There is no returns in this function. The results of this function
            pass to execution by utilizing the python's pass-by-reference
            nature.

        """
        node = tree.get_node(selected_node)
        node.data["split_permission"] = False

        # left child indices extracted from the nodes split-point and the
        # indices included in the parent node
        left_index = node.data["left_indices"]

        # right child indices
        right_index = node.data["right_indices"]

        # Nodes and data creation for the children
        # Uses the calculate_node_data function to create the data for the node
        tree.create_node(
            tag="cl" + str(self.node_ids + 1),
            identifier=self.node_ids + 1,
            parent=node.identifier,
            data=self.calculate_node_data(
                left_index,
                node.data["color_key"],
            ),
        )
        tree.create_node(
            tag="cl" + str(self.node_ids + 2),
            identifier=self.node_ids + 2,
            parent=node.identifier,
            data=self.calculate_node_data(
                right_index,
                self.cluster_color + 1,
            ),
        )

        self.cluster_color += 1
        self.node_ids += 2

    def calculate_node_data(self, indices, key):
        """
        Execution of the binary k-Means algorithm on the samples presented by
        the data_matrix. The two resulted clusters are the two new clusters if
        the leaf is chosen to be split. And calculation of the splitting
        criterion.

        Parameters
        ----------
        indices : numpy.ndarray
            The index of the samples in the original data matrix.
        data_matrix : numpy.ndarray
            The data matrix containing all the data for the samples.
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dict
            The necessary data for each node which are splitting point.

        """
        # if the number of samples
        if indices.shape[0] > self.min_sample_split:
            model = KMeans(n_clusters=2, n_init=10, random_state=self.random_state)
            labels = model.fit_predict(self.X[indices, :])
            centers = model.cluster_centers_

            left_child = indices[np.where(labels == 0)]
            right_child = indices[np.where(labels == 1)]
            centers = centers

            centered = util.center_data(self.X[indices, :])
            # Total scatter value calculation for the selection of the next
            # cluster to split.
            scat = np.linalg.norm(centered, ord="fro")

            split_criterion = scat
            flag = True
        # =========================
        else:
            left_child = None  # (ST2)
            right_child = None  # (ST2)
            centers = None  # (ST2)
            split_criterion = None  # (ST2)
            flag = False  # (ST2)

        return {
            "indices": indices,
            "left_indices": left_child,
            "right_indices": right_child,
            "centers": centers,
            "split_criterion": split_criterion,
            "split_permission": flag,
            "color_key": key,
            "dendrogram_check": False,
        }

    @property
    def random_state(self):
        return self._random_seed

    @random_state.setter
    def random_state(self, v):
        if v is not None and (not isinstance(v, int)):
            raise ValueError(
                "BisectingKmeans: min_sample_split: Invalid value it should be int and > 1"
            )
        np.random.seed(v)
        self._random_seed = v


class SceneHierarchicalClustering(BisectingKmeans):
    def __init__(
        self,
        max_clusters_number=100,
        min_sample_split=5,
        random_state=None,
    ):
        super().__init__(
            max_clusters_number=max_clusters_number,
            # min_sample_split=min_sample_split,
            random_state=random_state,
        )

    @staticmethod
    def select_frame(obj_set, frames, current_keys):
        # iterate through frames keys, if obj_set is a subset of frames[key].full_obj_list, return the key
        candidates = []
        for key in frames.keys():
            if obj_set.issubset(frames[key].full_obj_list.keys()):
                candidates.append(key)
        if len(candidates) == 0:
            return None
        else:
            # use the size of the full_obj_list and the confidence sum of all objects in obj_set to sort
            def key_func(x):
                return -len(frames[x].full_obj_list), -sum(
                    [frames[x].full_obj_list[obj] for obj in obj_set]
                )

            sorted_candidates = sorted(candidates, key=lambda x: key_func(x))

            for key in sorted_candidates:
                if key in current_keys:
                    return key

            return sorted_candidates[0]

    def fit(self, X, obj_ids, frames):
        """
        Execute the BisectingKmeans algorithm and return all the execution
        data in the form of a BisectingKmeans class object.

        Parameters
        ----------
        X : numpy.ndarray
            Data matrix with the samples on the rows and the variables on the
            columns.

        Returns
        -------
        self
            A BisectingKmeans class type object, with complete results on the
            algorithm's analysis.

        """
        snapshots = {}

        self.X = X
        self.samples_number = X.shape[0]

        # create an id vector for the samples of X
        indices = np.array([int(i) for i in range(np.size(self.X, 0))])

        # initialize tree and root node                         # step (0)
        tree = Tree()
        # nodes` unique IDs indicator
        self.node_ids = 0
        # nodes` next color indicator (necessary for visualization purposes)
        self.cluster_color = 0
        tree.create_node(
            tag="cl_" + str(self.node_ids),
            identifier=self.node_ids,
            data=self.calculate_node_data(indices, self.cluster_color),
        )
        # indicator for the next node to split
        selected_node = 0

        # if no possibility of split exists on the data
        if not tree.get_node(0).data["split_permission"]:
            print("cannot split at all")
            return self

        # Initialize the ST1 stopping criterion counter that count the number
        # of clusters.
        found_clusters = 1
        while True:
            while True:
                selected_node = self.select_kid(
                    tree.leaves(), decreasing=self.decreasing
                )
                if selected_node is None:
                    break
                obj_indices = tree.get_node(selected_node).data["indices"]
                obj_set = set([obj_ids[idx] for idx in obj_indices])
                key = self.select_frame(obj_set, frames, snapshots.keys())
                if key is not None:
                    if key not in snapshots:
                        snapshots[key] = frames[key]
                        snapshots[key].cluster = list(obj_set)
                    else:
                        snapshots[key].cluster += list(obj_set)
                    tree.remove_node(selected_node)
                else:
                    break
            if selected_node is None:
                break
            self.split_function(tree, selected_node)  # step (1)
            found_clusters = found_clusters + 1  # (ST1)

        return snapshots

    def calculate_node_data(self, indices, key):
        """
        Execution of the binary k-Means algorithm on the samples presented by
        the data_matrix. The two resulted clusters are the two new clusters if
        the leaf is chosen to be split. And calculation of the splitting
        criterion.

        Parameters
        ----------
        indices : numpy.ndarray
            The index of the samples in the original data matrix.
        data_matrix : numpy.ndarray
            The data matrix containing all the data for the samples.
        key : int
            The value of the color for each node.

        Returns
        -------
        data : dict
            The necessary data for each node which are splitting point.

        """
        # if the number of samples
        if indices.shape[0] > 2:
            # print(self.X[indices, :])
            model = KMeans(n_clusters=2, n_init=10, random_state=self.random_state)
            labels = model.fit_predict(self.X[indices, :])
            centers = model.cluster_centers_

            left_child = indices[np.where(labels == 0)]
            right_child = indices[np.where(labels == 1)]
            centers = centers

            centered = util.center_data(self.X[indices, :])
            # Total scatter value calculation for the selection of the next
            # cluster to split.
            # max_distance = np.max(np.linalg.norm(centered, axis=-1))
            scat = np.linalg.norm(centered, ord="fro")
            # split_criterion = max_distance
            split_criterion = indices.shape[0]
            flag = True
        # =========================
        elif indices.shape[0] == 2:
            left_child = indices[0:1]
            right_child = indices[1:2]
            centers = self.X[indices, :]
            centered = util.center_data(self.X[indices, :])
            scat = np.linalg.norm(centered, ord="fro")
            split_criterion = indices.shape[0]
            flag = True
        else:
            left_child = None  # (ST2)
            right_child = None  # (ST2)
            centers = None
            centered = util.center_data(self.X[indices, :])
            scat = np.linalg.norm(centered, ord="fro")
            split_criterion = indices.shape[0]
            flag = True

        return {
            "indices": indices,
            "left_indices": left_child,
            "right_indices": right_child,
            "centers": centers,
            "split_criterion": split_criterion,
            "split_permission": flag,
            "color_key": key,
            "dendrogram_check": False,
        }
