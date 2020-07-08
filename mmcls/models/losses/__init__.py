from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import CrossEntropyLoss, cross_entropy
from .label_smooth_loss import LabelSmoothLoss, label_smooth
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'label_smooth', 'LabelSmoothLoss', 'weighted_loss'
]
