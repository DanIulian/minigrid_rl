"""
    Copyright (c) https://github.com/lcswillems/torch-rl
"""

import os
import json
import numpy
import re
import torch
import torch_rl
import gym

import utils


def get_obss_preprocessor(*args, type=None, **kwargs):
    if type is None:
        return get_obss_preprocessor_simple(*args, **kwargs)
    elif type == "conditional":
        return get_obss_preprocessor_cond(*args, **kwargs)
    elif type == "default":
        return get_obss_preprocessor_default(*args, **kwargs)
    elif type == "tensor":
        return get_obss_preprocessor_tensor(*args, **kwargs)
    elif type == "aux_in":
        return get_obss_preprocessor_aux_in(*args, **kwargs)
    elif type == "carrying":
        return get_obss_preprocessor_carrying(*args, **kwargs)
    else:
        raise NotImplemented


def get_obss_preprocessor_carrying(env_id, obs_space, model_dir, max_image_value=15., normalize=True,
                                   permute=False, obs_type="compact"):

    if obs_type == "compact":
        preprocess_images_fn = preprocess_images
    elif obs_type == "rgb":
        preprocess_images_fn = preprocess_rgb_images
    else:
        raise ValueError("Wrong observation type")

    # Check if it is a MiniGrid environment
    if re.match("MiniGrid-.*", env_id):
        obs_space = {"image": obs_space.spaces['image'].shape, "text": 100, "carrying": 18}

        vocab = Vocabulary(model_dir, obs_space["text"])

        def preprocess_obss(obss, device=None, permute=permute):
            return torch_rl.DictList({
                "image": preprocess_images_fn([obs["image"] for obs in obss], device=device,
                                              max_image_value=max_image_value, normalize=normalize),
                "carrying": preprocess_images([obs["carrying"] for obs in obss], device=device,
                                              max_image_value=1., normalize=False)
            })
        preprocess_obss.vocab = vocab

    return obs_space, preprocess_obss


def get_obss_preprocessor_cond(env_id, obs_space, model_dir, max_image_value=15., normalize=True,
                               permute=False, obs_type="compact"):
    # Check if it is a MiniGrid environment
    if re.match("MiniGrid-.*", env_id):
        obs_space = {"image": obs_space.spaces['image'].shape, "text": 100}

        vocab = Vocabulary(model_dir, obs_space["text"])

        def preprocess_obss(obss, device=None, permute=permute):
            return torch_rl.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device,
                                           max_image_value=max_image_value, normalize=normalize),
                "text": torch.tensor([obs["mission"] for obs in obss], device=device)
            })
        preprocess_obss.vocab = vocab

    # Check if the obs_space is of type Box([X, Y, 3])
    elif isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3\
            and obs_space.shape[2] == 3:
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_rl.DictList({
                "image": preprocess_images(obss, device=device)
            })

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def get_obss_preprocessor_simple(env_id, obs_space, model_dir, max_image_value=15., normalize=True,
                                 permute=False, obs_type="compact"):

    if obs_type == "compact":
        preprocess_images_fn = preprocess_images
    elif obs_type == "rgb":
        preprocess_images_fn = preprocess_rgb_images
    else:
        raise ValueError("Wrong observation type")

    # Check if it is a MiniGrid environment
    if re.match("MiniGrid-.*", env_id):
        obs_space = {"image": obs_space.spaces['image'].shape, "text": 100}

        vocab = Vocabulary(model_dir, obs_space["text"])

        def preprocess_obss(obss, device=None, permute=permute):
            return torch_rl.DictList({
                "image": preprocess_images_fn([obs["image"] for obs in obss], device=device,
                                              max_image_value=max_image_value, normalize=normalize),
            })
        preprocess_obss.vocab = vocab

    # Check if the obs_space is of type Box([X, Y, 3])
    elif isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3\
            and obs_space.shape[2] == 3:
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_rl.DictList({
                "image": preprocess_images_fn(obss, device=device)
            })

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def get_obss_preprocessor_default(env_id, obs_space, model_dir, max_image_value=15., normalize=True,
                                  permute=False, obs_type="compact"):
    if obs_type == "compact":
        preprocess_images_fn = preprocess_images
    elif obs_type == "rgb":
        preprocess_images_fn = preprocess_rgb_images
    else:
        raise ValueError("Wrong observation type")

    # Check if it is a MiniGrid environment
    if re.match("MiniGrid-.*", env_id):
        obs_space = {"image": obs_space.spaces['image'].shape, "text": 100}

        vocab = Vocabulary(model_dir, obs_space["text"])

        def preprocess_obss(obss, device=None, permute=permute):
            return torch_rl.DictList({
                "image": preprocess_images_fn([obs["image"] for obs in obss], device=device,
                                              max_image_value=max_image_value, normalize=normalize),
                "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            })
        preprocess_obss.vocab = vocab

    # Check if the obs_space is of type Box([X, Y, 3])
    elif isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3 \
            and obs_space.shape[2] == 3:
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_rl.DictList({
                "image": preprocess_images_fn(obss, device=device)
            })

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def get_obss_preprocessor_aux_in(env_id, obs_space, model_dir, max_image_value=15., normalize=True,
                                 permute=False, obs_type="compact", aux_in_size: int = 2):
    # Check if it is a MiniGrid environment
    if re.match("MiniGrid-.*", env_id):
        obs_space = {"image": obs_space.spaces['image'].shape, "text": aux_in_size}

        def preprocess_obss(obss, device=None, permute=permute):
            return torch_rl.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device,
                                           max_image_value=max_image_value, normalize=normalize),
                "text": preprocess_aux_in([obs["aux_in"] for obs in obss], device=device)
            })
    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def get_obss_preprocessor_tensor(env_id, obs_space, model_dir, max_image_value=15., normalize=True,
                                 permute=False, obs_type="compact"):
    # Check if it is a MiniGrid environment
    if re.match("MiniGrid-.*", env_id):
        obs_space = {"image": obs_space.spaces['image'].shape, "text": 100}

        vocab = Vocabulary(model_dir, obs_space["text"])

        def preprocess_obss(obss, device=None, permute=permute):
            return torch_rl.DictList({
                "image": preprocess_images_tensor([obs["image"] for obs in obss], device=device,
                                           max_image_value=max_image_value, normalize=normalize),
                "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            })
        preprocess_obss.vocab = vocab

    # Check if the obs_space is of type Box([X, Y, 3])
    elif isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3 \
            and obs_space.shape[2] == 3:
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_rl.DictList({
                "image": preprocess_images_tensor(obss, device=device)
            })

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_aux_in(aux_data, device=None, norm_value=16., normalize=True):
    # Bug of Pytorch: very slow if not first converted to numpy array
    aux_data = numpy.array(aux_data)
    aux_data = torch.tensor(aux_data, device=device, dtype=torch.float)
    if normalize:
        aux_data.div_(norm_value)

    return aux_data


def preprocess_images(images, device=None, max_image_value=15., normalize=True, permute=False):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    images = torch.tensor(images, device=device, dtype=torch.float)
    if normalize:
        images.div_(max_image_value)
    if permute:
        images = images.permute(0, 3, 1, 2)
    return images


def preprocess_rgb_images(images, device=None, max_image_value=127.5, normalize=True, permute=False):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    images = torch.tensor(images, device=device, dtype=torch.float)
    if normalize:
        images.div_(max_image_value)
        images.sub_(1.0)
    if permute:
        images = images.permute(0, 3, 1, 2)
    return images


def preprocess_images_tensor(images, device=None, max_image_value=15., normalize=True,
                            permute=False):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = torch.stack(images)
    images = images.to(device).float()
    if normalize:
        images.div_(max_image_value)
    if permute:
        images = images.permute(0, 3, 1, 2)
    return images


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)

class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, model_dir, max_size):
        self.path = utils.get_vocab_path(model_dir)
        self.max_size = max_size
        self.vocab = {}
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self):
        utils.create_folders_if_necessary(self.path)
        json.dump(self.vocab, open(self.path, "w"))