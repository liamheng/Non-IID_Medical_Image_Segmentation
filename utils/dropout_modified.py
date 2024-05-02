import random

from torch import nn

DROPOUT_NONE = 0
DROPOUT_VANILLA = 1
DROPOUT_UD = 2

def get_dropout(dropout_type, dropout_rate=None):
    if dropout_type == DROPOUT_NONE:
        return nn.Identity()
    elif dropout_type == DROPOUT_VANILLA:
        return nn.Dropout(0.5 if dropout_rate is None else dropout_rate)
    elif dropout_type == DROPOUT_UD:
        return UDBasedDropout(0.1 if dropout_rate is None else dropout_rate)
    else:
        raise NotImplementedError("Unknown dropout type: {}".format(dropout_type))

class UDBasedDropout(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(UDBasedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if self.training:
            return (1 + random.uniform(-self.dropout_rate, self.dropout_rate)) * x
        return x

    def dropout(self, x):
        return x * (1 - self.dropout_rate)