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

"""Set of functions used to train a R-network."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Const(object):
  """Constants"""
  MAX_ACTION_DISTANCE = 5
  NEGATIVE_SAMPLE_MULTIPLIER = 5
  # env
  OBSERVATION_HEIGHT = 120
  OBSERVATION_WIDTH = 160
  OBSERVATION_CHANNELS = 3
  OBSERVATION_SHAPE = (OBSERVATION_HEIGHT, OBSERVATION_WIDTH,
                       OBSERVATION_CHANNELS)
  # model and training
  BATCH_SIZE = 64
  EDGE_CLASSES = 2
  DUMP_AFTER_BATCHES = 100
  EDGE_MAX_EPOCHS = 2000
  ADAM_PARAMS = {
      'lr': 1e-04,
      'beta_1': 0.9,
      'beta_2': 0.999,
      'epsilon': 1e-08,
      'decay': 0.0
  }
  ACTION_REPEAT = 4
  STORE_CHECKPOINT_EVERY_N_EPOCHS = 30


def generate_positive_example(buffer_position,
                              next_buffer_position):
    """Generates a close enough pair of states."""
    first = buffer_position
    second = next_buffer_position

    # Make R-network symmetric.
    if random.random() < 0.5:
        first, second = second, first

    return first, second


def generate_negative_example(buffer_position,
                              len_episode_buffer,
                              max_action_distance,
                              negative_sample_multiplier=5):
    """Generates a far enough pair of states."""
    assert buffer_position < len_episode_buffer
    # Defines the interval that must be excluded from the sampling.
    time_interval = (negative_sample_multiplier * max_action_distance)
    min_index = max(buffer_position - time_interval, 0)
    max_index = min(buffer_position + time_interval + 1, len_episode_buffer)

    # Randomly select an index outside the interval.
    effective_length = len_episode_buffer - (max_index - min_index)
    range_max = effective_length - 1
    if range_max <= 0:
        return buffer_position, None
    index = random.randint(0, range_max)
    if index >= min_index:
        index = max_index + (index - min_index)
    return buffer_position, index


def compute_next_buffer_position(buffer_position,
                                 positive_example_candidate,
                                 max_action_distance,
                                 mode):
    """Computes the buffer position for the next training example."""
    if mode == 'v3_affect_num_training_examples_overlap':
        # This version was initially not intended (changing max_action_distance
        # affects the number of training examples, and we can also get overlap
        # across generated examples), but we have it because it produces good
        # results (reward at ~40 according to raveman@ on 2018-10-03).
        # R-nets /cns/vz-d/home/dune/episodic_curiosity/raphaelm_train_r_mad2_4 were
        # generated with this version (the flag was set
        # v1_affect_num_training_examples, but it referred to a "buggy" version of
        # v1 that is reproduced here with that v3).
        return buffer_position + random.randint(1, max_action_distance) + 1
    if mode == 'v1_affect_num_training_examples':
        return positive_example_candidate + 1
    if mode == 'v2_fixed_num_training_examples':
        # Produces the ablation study in the paper submitted to ICLR'19
        # (https://openreview.net/forum?id=SkeK3s0qKQ), section S4.1.
        return buffer_position + random.randint(1, 5) + 1


def create_training_data_from_episode_buffer_v4(episode_buffer,
                                                max_action_distance,
                                                avg_num_examples_per_env_step,
                                                negative_sample_multiplier=5):
    """Sampling of positive/negative examples without using stride logic."""

    num_examples = int(avg_num_examples_per_env_step * len(episode_buffer))
    num_examples_per_class = num_examples // 2

    # We first generate positive pairs, and then sample from them (ensuring that
    # we don't select twice exactly the same pair (i, i + j)).
    positive_pair_candidates = []
    for first in range(len(episode_buffer)):
        for j in range(1, max_action_distance + 1):
            second = first + j
            if second >= len(episode_buffer):
                continue
            positive_pair_candidates.append(
                (first, second) if random.random() > 0.5 else (second, first))
    assert len(positive_pair_candidates) >= num_examples_per_class
    positive_pairs = random.sample(positive_pair_candidates,
                                   num_examples_per_class)

    # Generate negative pairs.
    num_negative_candidates = len(episode_buffer) * \
                              (len(episode_buffer) - 2 * negative_sample_multiplier * max_action_distance) / 2
    # Make sure we have enough negative examples to sample from (with some
    # headroom). If that does not happen (meaning very short episode buffer given
    # current values of negative_sample_multiplier, max_action_distance), don't
    # generate any training example.
    if num_negative_candidates < 2 * num_examples_per_class:
        return [], [], []
    negative_pairs = set()

    while len(negative_pairs) < num_examples_per_class:
        i = random.randint(0, len(episode_buffer) - 1)
        j = generate_negative_example(
            i, len(episode_buffer), max_action_distance, negative_sample_multiplier)[1]
        # Checking this is not strictly required, because it should happen
        # infrequently with current parameter values.
        # We still check for it for the symmetry with the positive example case.
        if (i, j) not in negative_pairs and (j, i) not in negative_pairs:
            negative_pairs.add((i, j))

    x1 = []
    x2 = []
    labels = []
    for i, j in positive_pairs:
        x1.append(episode_buffer[i])
        x2.append(episode_buffer[j])
        labels.append(1) # they are reacheable one from another

    for i, j in negative_pairs:
        x1.append(episode_buffer[i])
        x2.append(episode_buffer[j])
        labels.append(0) # unreachable states

    return x1, x2, labels


def create_training_data_from_episode_buffer_v123(episode_buffer,
                                                  max_action_distance,
                                                  mode,
                                                  negative_sample_multiplier=5):
    """Samples intervals and forms pairs."""
    first_second_label = []
    buffer_position = 0
    while True:
        positive_example_candidate = (
            buffer_position + random.randint(1, max_action_distance))
        next_buffer_position = compute_next_buffer_position(
            buffer_position, positive_example_candidate,
            max_action_distance, mode)

        if (next_buffer_position >= len(episode_buffer) or
            positive_example_candidate >= len(episode_buffer)):
            break

        label = random.randint(0, 1)
        if label:
            first, second = generate_positive_example(buffer_position,
                                                      positive_example_candidate)
        else:
            first, second = generate_negative_example(buffer_position,
                                                      len(episode_buffer),
                                                      max_action_distance,
                                                      negative_sample_multiplier)
        if first is None or second is None:
            break

        first_second_label.append((first, second, label))
        buffer_position = next_buffer_position

    x1 = []
    x2 = []
    labels = []
    for first, second, label in first_second_label:
        x1.append(episode_buffer[first])
        x2.append(episode_buffer[second])
        labels.append(label)

    return x1, x2, labels


class RNetworkTrainer(object):
    """Train a R network in an online way."""

    def __init__(self,
                 r_model,
                 optimizer,
                 device,
                 preprocess_obs_fn,
                 batch_size=64,
                 num_epochs=6,
                 observation_history_size=20000,
                 training_interval=20000,
                 training_data_type="v4"):
        # The training interval is assumed to be the same as the history size
        # for invalid negative values.
        if training_interval < 0:
            training_interval = observation_history_size

        self._r_model = r_model
        self._optimizer = optimizer
        self._device = device
        self._preprocess_obs_fn = preprocess_obs_fn

        self._training_interval = training_interval
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._training_data_type = training_data_type

        # Keeps track of the last N observations.
        # Those are used to train the R network in an online way.
        self._fifo_observations = [None] * observation_history_size
        self._fifo_dones = [None] * observation_history_size
        self._fifo_index = 0
        self._fifo_count = 0

        # Used to save checkpoints.
        self._current_epoch = 0

    def on_new_observation(self, observations, unused_rewards, dones, infos):
        """Event triggered when the environments generate a new observation."""

        self._fifo_observations[self._fifo_index] = self._preprocess_obs_fn(observations)
        self._fifo_dones[self._fifo_index] = dones
        self._fifo_index = (
                (self._fifo_index + 1) % len(self._fifo_observations))
        self._fifo_count += 1

        if self._fifo_count > 0 and (self._fifo_count % self._training_interval == 0):
            print('Training the R-network after: {}'.format(self._fifo_count))
            history_observations, history_dones = self._get_flatten_history()
            self.train(history_observations, history_dones)

    def _get_flatten_history(self):
        """Convert the history given as a circular fifo to a linear array."""

        if self._fifo_count < len(self._fifo_observations): # the buffer is not full yet
            return (self._fifo_observations[:self._fifo_count],
                    self._fifo_dones[:self._fifo_count])

        # Reorder the indices.
        history_observations = self._fifo_observations[self._fifo_index:]
        history_observations.extend(self._fifo_observations[:self._fifo_index])
        history_dones = self._fifo_dones[self._fifo_index:]
        history_dones.extend(self._fifo_dones[:self._fifo_index])

        return history_observations, history_dones

    def _split_history(self, observations, dones):
        """Returns some individual trajectories."""

        if len(observations) == 0:  # pylint: disable=g-explicit-length-test
            return []

        # Number of environments that generated "observations",
        # and total number of steps.
        nenvs = len(dones[0])
        nsteps = len(dones)

        # Starting index of the current trajectory.
        start_index = [0] * nenvs

        trajectories = []
        for k in range(nsteps):
            for n in range(nenvs):
                if dones[k][n] or k == nsteps - 1:
                    next_start_index = k + 1
                    time_slice = observations[start_index[n] : next_start_index]
                    trajectories.append([obs[n] for obs in time_slice])
                    start_index[n] = next_start_index

        return trajectories

    def _prepare_data(self, observations, dones):
        """Generate the positive and negative pairs used to train the R network."""

        max_action_distance = 5
        avg_num_examples_per_env_step = 1
        negative_sample_multiplier=5
        mode = 'v2_fixed_num_training_examples'

        all_x1 = []
        all_x2 = []
        all_labels = []
        trajectories = self._split_history(observations, dones)
        for trajectory in trajectories:
            if self._training_data_type == "v4":
                x1, x2, labels = create_training_data_from_episode_buffer_v4(trajectory,
                                                                             max_action_distance,
                                                                             avg_num_examples_per_env_step,
                                                                             negative_sample_multiplier)
            else:
                x1, x2, labels = create_training_data_from_episode_buffer_v123(trajectory,
                                                                               max_action_distance,
                                                                               mode,
                                                                               negative_sample_multiplier)

            all_x1.extend(x1)
            all_x2.extend(x2)
            all_labels.extend(labels)

        return all_x1, all_x2, all_labels

    def _shuffle(self, x1, x2, labels):
        sample_count = len(x1)
        assert len(x2) == sample_count
        assert len(labels) == sample_count
        permutation = np.random.permutation(sample_count)
        x1 = [x1[p] for p in permutation]
        x2 = [x2[p] for p in permutation]
        labels = [labels[p] for p in permutation]

        return x1, x2, labels

    def _generate_batch(self, x1, x2, labels):
        """Generate batches of data used to train the R network."""

        while True:
            # Train for one epoch.
            sample_count = len(x1)
            number_of_batches = sample_count // self._batch_size

            for batch_index in range(number_of_batches):
                from_index = batch_index * self._batch_size
                to_index = (batch_index + 1) * self._batch_size
                yield ([np.array(x1[from_index : to_index]),
                        np.array(x2[from_index : to_index])],
                        np.eye(2, dtype=np.uint8)[labels[from_index: to_index]])

            # After each epoch, shuffle the data.
            x1, x2, labels = self._shuffle(x1, x2, labels)

    def train(self, history_observations, history_dones):
        """Do one pass of training of the R-network."""

        x1, x2, labels = self._prepare_data(history_observations, history_dones)
        x1, x2, labels = self._shuffle(x1, x2, labels)

        # Split between train and validation data.
        n = len(x1)
        train_count = (95 * n) // 100

        x1_train, x2_train, labels_train = (
            x1[:train_count], x2[:train_count], labels[:train_count])
        x1_valid, x2_valid, labels_valid = (
            x1[train_count:], x2[train_count:], labels[train_count:])

        validation_data = ([np.array(x1_valid),
                            np.array(x2_valid)],
                            np.eye(2, dtype=np.uint8)[labels_valid])

        loss_fn = nn.CrossEntropyLoss()

        nr_steps = train_count // self._batch_size
        for epoch in range(self._num_epochs):
            epoch_loss = 0.0
            for step in range(nr_steps):
                batch_x1, batch_x2, batch_labels = self._generate_batch(x1_train, x2_train, labels_train)

                self._optimizer.zero_grad()

                batch_x1 = torch.tensor(batch_x1, dtype=torch.FloatTensor, device=self._device)
                batch_x2 = torch.tensor(batch_x2, dtype=torch.FloatTensor, device=self._device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.LongTensor, device=self._device)

                emb_x1 = self._r_model.forward(batch_x1)
                emb_x2 = self._r_model.forward(batch_x2)

                predicted_similarities = self._r_model.forward_similarity(emb_x1, emb_x2)

                loss = loss(predicted_similarities, batch_labels)

                epoch_loss += loss.item()

                loss.backward()
                self._optimizer.step()

            epoch_loss /= nr_steps
            print("Training N Network: Epoch-{}  Loss-{}".format(epoch, epoch_loss))
