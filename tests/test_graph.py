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
        for struct in tqdm(self.structures[:1_000]):
            g = self.knn.convert(struct, fmt="vasp")

            old_row, old_col = g.edge_index  # type: ignore
            num_atoms = g.num_nodes
            num_bonds = g.edge_index.size(1)  # type: ignore

            # Reference (slow) result
            ref_rows, ref_cols = self.lg_transform._get_new_adj(
                old_row, old_col, num_atoms, num_bonds
            )

            # Optimized result from the LineGraph transform
            opt_rows, opt_cols = self.lg_transform._get_new_adj_test(
                old_row, old_col, num_atoms, num_bonds
            )

            # Check list lengths
            assert len(ref_rows) == len(opt_rows)
            assert len(ref_cols) == len(opt_cols)

            # Element-wise tensor equality
            for r_ref, r_opt in zip(ref_rows, opt_rows):
                assert torch.equal(r_ref, r_opt)

            for c_ref, c_opt in zip(ref_cols, opt_cols):
                assert torch.equal(c_ref, c_opt)


if __name__ == "__main__":
    test = TestGraph()
    test.test_lg_adj()
    test.test_knn_performances()
