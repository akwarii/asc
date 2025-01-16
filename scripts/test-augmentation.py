from src.datasets import CustomDataset
from src.transforms import DropoutNode

dataset = CustomDataset(k=12, rcut=6.0)
data = dataset.get(0)
print(data)

augmenter = DropoutNode(p=1.0)
augmented_graphs = augmenter.forward(x=data)  # type: ignore
print(augmented_graphs)
