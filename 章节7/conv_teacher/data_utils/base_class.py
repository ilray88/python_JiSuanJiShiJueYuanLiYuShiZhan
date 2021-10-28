import abc


class Dataset(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, 
                 dataset_path, 
                 batch_size, 
                 shuffle=True, 
                 normalize=True, 
                 augmentation=True):
        pass

    @abc.abstractmethod
    def next_batch(self, which_set):
        pass

    @abc.abstractmethod
    def num_examples(self, which_set):
        pass
