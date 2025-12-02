from . import backprop, forward, serialization, training
from .core import Network

# Attach method implementations to the Network class
Network._get_activation = staticmethod(forward.get_activation)
Network._get_activation_derivative = staticmethod(
    forward.get_activation_derivative
)
Network.forward = forward.forward
Network.predict = forward.predict
Network._backprop_sample = backprop.backprop_sample

Network.train_epoch = training.train_epoch
Network.train = training.train

Network.to_dict = serialization.to_dict
Network.save = serialization.save
Network.from_dict = staticmethod(
    lambda data: serialization.from_dict(data, Network)
)
Network.load = staticmethod(
    lambda filepath: serialization.load(filepath, Network)
)


__all__ = ["Network"]
