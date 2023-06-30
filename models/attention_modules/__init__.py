from .CBAM import *
from .CSAM import *
import torch
import sys
import inspect


def find_model_using_name(model_name):
    for cls_name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if model_name.lower() == cls_name.lower() and issubclass(cls, torch.nn.Module):
            return cls
    raise Exception('不存在该attention类型')
