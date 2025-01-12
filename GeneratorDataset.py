from torch.utils.data import IterableDataset

class GeneratorDataset(IterableDataset):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def __iter__(self):
        return iter(self.generator)

    def __len__(self):
        return 2048  # fixed steps
