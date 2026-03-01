from abc import ABC, abstractmethod
from typing import Dict

class BaseSaver(ABC):

    @abstractmethod
    def save(self, slide_file: str, result: Dict):
        raise NotImplementedError