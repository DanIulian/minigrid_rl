# AndreiN, 2019

from agents.base import BaseAlgo
from abc import abstractmethod


class BaseAlgov2(BaseAlgo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def update_parameters(self):
        raise NotImplemented

    @abstractmethod
    def get_save_data(self):
        raise NotImplemented
