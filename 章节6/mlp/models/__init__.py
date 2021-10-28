import importlib
from models.base_class import Model
from layers.fc import *

def get_model_with_name(m_name):
    module_name = 'models.{}'.format(m_name)
    
    model_module = importlib.import_module(module_name)

    for name, cls in model_module.__dict__.items():
        if name.lower() == m_name.lower() and issubclass(cls, Model):
            model_cls = cls
    
    if model_cls is None:
        raise ValueError('Unsupported model: {}'.format(m_name))

    return model_cls
