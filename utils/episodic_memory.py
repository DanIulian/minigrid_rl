# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class that represents an episodic memory."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F


class EpisodicMemory(object):
    """Episodic memory."""
    def __init__(self,
                 observation_shape,
                 observation_compare_fn,
                 device,
                 nr_neighbours=10,
                 eps=0.001,
                 cluster_distance=0.008,
                 kernel_eps=0.0001,
                 max_similarity=8,
                 replacement='fifo',
                 capacity=200):
        """Creates an episodic memory.
        Args:
        observation_shape: Shape of an observation as a list
        observation_compare_fn: Function used to measure similarity between
            two observations. This function returns the estimated probability that
            two observations are similar.
        nr_neighbors: Number of nearest neighbours when computing pseudo-counts
        eps: Pseudo-counts constant
        cluster_distance: cluster distance when computing kernel simiarity
        kernel_eps: kernel constant
        max_similarity: maximum similarity
        replacement: String to select the behavior when a sample is added
            to the memory when this one is full.
            Can be one of: 'fifo', 'random'.
            'fifo' keeps the last "capacity" samples into the memory.
            'random' results in a geometric distribution of the age of the samples
            present in the memory.
        capacity: Capacity of the episodic memory.
        Raises:
        ValueError: when the replacement scheme is invalid.
        """
        self._capacity = capacity
        self._replacement = replacement
        self._device = device
        self._nr_neighbours = nr_neighbours
        self._eps = eps
        self._cluster_distance = cluster_distance
        self._kernel_eps = kernel_eps
        self._max_similarity = max_similarity

        if self._replacement not in ['fifo', 'random']:
            raise ValueError('Invalid replacement scheme')

        self._observation_shape = observation_shape
        self._observation_compare_fn = observation_compare_fn
        self.reset(False)

    def reset(self, show_stats=False):
        """Resets the memory."""
        if show_stats:
            size = len(self)
            age_histogram, _ = np.histogram(self._memory_age[:size],
                                          10, [0, self._count])
            age_histogram = age_histogram.astype(np.float32)
            age_histogram = age_histogram / np.sum(age_histogram)
            print('Number of samples added in the previous trajectory: {}'.format(
                self._count))
            print('Histogram of sample freshness (old to fresh): {}'.format(
                age_histogram))

        self._count = 0
        # Stores environment observations.
        self._obs_memory = np.zeros([self._capacity] + self._observation_shape)
        # Stores the infos returned by the environment. For debugging and
        # visualization purposes.
        self._info_memory = [None] * self._capacity
        self._memory_age = np.zeros([self._capacity], dtype=np.int32)

    @property
    def capacity(self):
        return self._capacity

    def __len__(self):
        return min(self._count, self._capacity)

    @property
    def info_memory(self):
        return self._info_memory

    def add(self, observation, info):
        """Adds an observation to the memory.
        Args:
          observation: Observation to add to the episodic memory.
          info: Info returned by the environment together with the observation,
                for debugging and visualization purposes.
        Raises:
          ValueError: when the capacity of the memory is exceeded.
        """
        if self._count >= self._capacity:
            if self._replacement == 'random':
                # By using random replacement, the age of elements inside the memory
                # follows a geometric distribution (more fresh samples compared to
                # old samples).
                index = np.random.randint(low=0, high=self._capacity)
            elif self._replacement == 'fifo':
                # In this scheme, only the last self._capacity elements are kept.
                # Samples are replaced using a FIFO scheme (implemented as a circular
                # buffer).
                index = self._count % self._capacity
            else:
                raise ValueError('Invalid replacement scheme')
        else:
            index = self._count

        self._obs_memory[index] = observation
        self._info_memory[index] = info
        self._memory_age[index] = self._count
        self._count += 1

    def compute_intrinsic_reward(self, observation):
        """Compute intrinsic reward as similarity between the observation and the
           K-nearest neighbours in memory
        Args:
          observation: The input observation embedding after passing it through ICM
        Returns:
          intrinsic_reward
        """

        size = len(self)

        # -- Compute kNN by computing Euclidian distances between the observation and
        # the memory content
        observation = observation.unsqueeze(0).repeat(size, 1)
        mem = torch.tensor(self._obs_memory[:size], dtype=torch.float, device=self._device)
        dist = torch.norm(observation - mem, dim=1)
        kNN_values, kNN_idx = torch.topk(dist, self._nr_neighbours, dim=0)

        knn_dist = kNN_values.pow(2)

        # TODO update moving average
        dm = compute_moving_average(knn_dist)
        # -- Compute distances and apply the kernel computing the pseudo counts

        dist = knn_dist / dm
        dist = torch.max(dist - self._cluster_distance, torch.zeros_like(dist))

        tensor_eps = torch.full(dist.shape, self._kernel_eps, dtype=torch.float, device=self._device)
        kernel_values = torch.exp(torch.log(tensor_eps) - torch.log(tensor_eps + dist))

        sim = (kernel_values.sum().sqrt()).item() + self._eps
        if sim > self._max_similarity:
            return 0
        return 1 / sim



def similarity_to_memory(observation,
                         episodic_memory,
                         similarity_aggregation='percentile'):
    """Returns the similarity of the observation to the episodic memory.
    Args:
        observation: The observation the agent transitions to.
        episodic_memory: Episodic memory.
        similarity_aggregation: Aggregation method to turn the multiple
            similarities to each observation in the memory into a scalar.
    Returns:
        A scalar corresponding to the similarity to episodic memory. This is
        computed by aggregating the similarities between the new observation
        and every observation in the memory, according to 'similarity_aggregation'.
    """
    # Computes the similarities between the current observation and the past
    # observations in the memory.
    memory_length = len(episodic_memory)
    if memory_length == 0:
        return 0.0

    # similarities is a 2d tensor, where the second value correspond to how reachable the obs are
    similarities = (episodic_memory.similarity(observation))[:, 1].cpu()
    # Implements different surrogate aggregated similarities.
    # TODO(damienv): Implement other types of surrogate aggregated similarities.
    if similarity_aggregation == 'max':
        aggregated = np.max(similarities)
    elif similarity_aggregation == 'nth_largest':
        n = min(10, memory_length)
        aggregated = np.partition(similarities, -n)[-n]
    elif similarity_aggregation == 'percentile':
        percentile = 90
        aggregated = np.percentile(similarities, percentile)
    elif similarity_aggregation == 'relative_count':
        # Number of samples in the memory similar to the input observation.
        count = sum(similarities > 0.5)
        aggregated = float(count) / len(similarities)

    return aggregated