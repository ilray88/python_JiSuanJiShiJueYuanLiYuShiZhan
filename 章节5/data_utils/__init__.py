import importlib
import tensorflow as tf
from data_utils.base_class import Dataset


def get_dataset_with_name(d_name):
    module_name = 'data_utils.{}'.format(d_name)
    
    dataset_module = importlib.import_module(module_name)

    for name, cls in dataset_module.__dict__.items():
        if name.lower() == d_name.lower() and issubclass(cls, Dataset):
            dataset_cls= cls
    
    if dataset_cls is None:
        raise ValueError('Unsupported dataset: {}'.format(d_name))

    return dataset_cls

def get_dataloader(args):
    dataset_name = args.dataset
    dataset_cls = get_dataset_with_name(dataset_name)

    inp_params = {
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'normalize': not args.not_normalize,
        'shuffle': not args.not_shuffle,
        'augmentation': args.augmentation
    }

    if 'cifar' in dataset_name:
        inp_params['c10'] = args.c10
        inp_params['coarse_label'] = args.coarse_label
    elif 'iris' in dataset_name:
        inp_params['split_train_and_test'] = args.split_train_and_test
    elif 'oxford_flower' in dataset_name:
        inp_params['resize'] = args.resize

    dataset_instance = dataset_cls(**inp_params)
    
    return dataset_instance

def get_placeholders(args):
    dataset_name = args.dataset

    def init_placeholder(shape, name):
        return tf.placeholder(dtype=tf.float32, shape=shape, name=name)

    if 'mnist' in dataset_name:
        return init_placeholder(shape=[None, 784], name='X'), init_placeholder(shape=[None, 10], name='Y')
    
    if 'cifar' in dataset_name:
        if args.coarse_label:
            return init_placeholder(shape=[None, 32, 32, 3], name='X'), init_placeholder(shape=[None, 20], name='Y')
        return init_placeholder(shape=[None, 32, 32, 3], name='X'), init_placeholder(shape=[None, 10 if args.c10 else 100], name='Y')
    
    if 'iris' in dataset_name:
        return init_placeholder(shape=[None, 5], name='X'), init_placeholder(shape=[None, 3], name='Y')
    
    if 'oxford_flower' in dataset_name:
        return init_placeholder(shape=[None, *args.resize, 3], name='X'), init_placeholder(shape=[None, 102], name='Y')
        