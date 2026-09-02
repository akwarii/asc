from .box_strain import BoxStrain
from .dropout_edge import DropoutEdge
from .dropout_node import DropoutNode
from .filter_atomic_types import FilterAtomicTypes
from .line_graph import LineGraph
from .random_perturbation import RandomPerturbation

# Runtime augmentations that the datamodule applies only to training data.
TRAIN_ONLY_TRANSFORMS = (BoxStrain, DropoutEdge, DropoutNode, RandomPerturbation)
