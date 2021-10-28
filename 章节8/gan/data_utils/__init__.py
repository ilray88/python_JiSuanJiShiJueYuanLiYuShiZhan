import tensorflow.compat.v1 as tf


# 根据用户传入的数据集名称返回数据集对象
def get_dataset_with_name(d_name):
    import importlib
    from data_utils.base_class import Dataset

    # 使用importlib根据用户传入的名称导入相应模块（py文件）
    module_name = 'data_utils.{}'.format(d_name)
    dataset_module = importlib.import_module(module_name)

    # 确保导入正确的类
    for name, cls in dataset_module.__dict__.items():
        if name.lower() == d_name.lower() and issubclass(cls, Dataset):
            dataset_cls= cls
            break
    
    if dataset_cls is None:
        raise ValueError('Unsupported dataset: {}'.format(d_name))

    return dataset_cls

def get_dataloader(args):
    dataset_name = args.dataset
    
    # 得到指定名称的数据集类
    dataset_cls = get_dataset_with_name(dataset_name)

    # 首先从传入的参数取得所有数据集都有的参数
    inp_params = {
        'data_path': args.data_path,
        'batch_size': args.batch_size,
        'normalize': not args.not_normalize,
        'shuffle': not args.not_shuffle,
        'augmentation': args.augmentation
    }

    # 再根据数据集的不同取得它们对应的特有参数
    # 使用CIFAR数据集时，需要指定使用CIFAR-10或CIFAR-100
    # 使用CIFAR-100时,需要指定是否使用大类进行分类
    if 'cifar' in dataset_name:
        inp_params['c10'] = args.c10
        inp_params['coarse_label'] = args.coarse_label
    # elif 'iris' in dataset_name:
    #     inp_params['split_train_and_test'] = args.split_train_and_test
    # 使用Oxford flower数据集时,需要指定统一的缩放大小
    elif 'oxford_flower' in dataset_name:
        inp_params['resize'] = args.resize

    # 使用所有的参数实例化该数据集对象并返回
    dataset_instance = dataset_cls(**inp_params)
    
    return dataset_instance

def get_placeholders(args):
    dataset_name = args.dataset

    def init_placeholder(shape, name):
        return tf.placeholder(dtype=tf.float32, shape=shape, name=name)
    
    # MNIST数据集图像为28*28的单通道图像，总共10类
    if 'mnist' in dataset_name:
        return init_placeholder(shape=[None, 28, 28, 1], name='X'), \
               init_placeholder(shape=[None, 10], name='Y')

    # CIFAR数据集图像为32*32的多通道图像
    # 可以分为CIFAR-10或CIFAR-100，CIFAR-100时有可能指定为20大类
    if 'cifar' in dataset_name:
        if args.coarse_label:
            return init_placeholder(shape=[None, 32, 32, 3], name='X'), \
                   init_placeholder(shape=[None, 20], name='Y')
        return init_placeholder(shape=[None, 32, 32, 3], name='X'), \
               init_placeholder(shape=[None, 10 if args.c10 else 100], name='Y')
    
    # if 'iris' in dataset_name:
    #     return init_placeholder(shape=[None, 5], name='X'), \
    #            init_placeholder(shape=[None, 3], name='Y')
    
    # Oxford flower数据集的图像大小不确定，可以由用户指定，一共102类
    if 'oxford_flower' in dataset_name:
        return init_placeholder(shape=[None, *args.resize, 3], name='X'), \
               init_placeholder(shape=[None, 102], name='Y')
        