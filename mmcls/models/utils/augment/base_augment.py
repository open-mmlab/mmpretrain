class BaseAugment:
    """Base class for MixupLayer.

    Args:
        num_classes (int): The number of classes.
        prob (float): MixUp probability. It should be in range [0, 1].
            Default to 0.5
    """

    def __init__(self, num_classes, prob=0.5):
        super(BaseAugment, self).__init__()

        assert isinstance(num_classes, int)
        assert isinstance(prob, float) and 0.0 <= prob <= 1.0

        self.num_classes = num_classes
        self.prob = prob
