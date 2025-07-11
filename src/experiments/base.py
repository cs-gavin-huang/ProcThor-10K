from abc import ABC, abstractmethod

class ExperimentBase(ABC):
    def __init__(self, house_id, output_dir):
        self.house_id = house_id
        self.output_dir = output_dir

    @abstractmethod
    def run(self):
        pass 