from pathlib import Path

import pandas as pd
import torch
from line_profiler import profile
from src.graph import KNNGraph
from src.transforms.line_graph import LineGraph
from tqdm.auto import tqdm


class TestGraph:
    def __init__(self):
        self.knn = KNNGraph(k=20, rcut=7.5)
        self.lg_transform = LineGraph()
        self.structures = self._load_random_structure()

    @staticmethod
    def _load_random_structure() -> list[str]:
        df = pd.read_csv(
            Path(__file__).parent.parent / "data" / "csg" / "raw" / "CSG_micro.csv",
            usecols=["Structure"],
        )
        structures: list[str] = df.sample(frac=0.05, random_state=42)["Structure"].to_list()

        return structures

    @profile
    def test_knn_performances(self):
        for struct in tqdm(self.structures):
            g = self.knn.convert(struct, fmt="vasp")
            _ = self.lg_transform(g)

    def test_lg_adj(self):
        """Check that the optimized _get_new_adj returns the same rows/cols as the original."""
        if not hasattr(self.lg_transform, "optimized"):
            return

        for struct in tqdm(self.structures[:1_000]):
            g = self.knn.convert(struct, fmt="vasp")

            # Reference (slow) result
            old = self.lg_transform.forward(g)

            # Optimized result from the LineGraph transform
            new = self.lg_transform.optimized(g)

            # Check list lengths
            assert len(old.edge_index[0]) == len(new.edge_index[0])
            assert len(old.edge_index[1]) == len(new.edge_index[1])

            # Element-wise tensor equality
            for r_ref, r_opt in zip(old.edge_index[0], new.edge_index[0]):
                assert torch.equal(r_ref, r_opt)

            for c_ref, c_opt in zip(old.edge_index[1], new.edge_index[1]):
                assert torch.equal(c_ref, c_opt)


if __name__ == "__main__":
    test = TestGraph()
    # test.test_lg_adj()
    test.test_knn_performances()
