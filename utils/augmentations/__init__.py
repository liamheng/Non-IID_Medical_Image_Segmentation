from utils.augmentations.augment_prototype import AugmentBase
from utils.util import get_class_from_subclasses
from .augment_lib.randomaugment import RandAugment
from .fmaug import FMAug


def get_augment_by_name(name):
    return get_class_from_subclasses(AugmentBase, name)
