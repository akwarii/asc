from line_profiler import profile
from src.graph import KNNGraph


class TestGraph:
    knn = KNNGraph(k=20, rcut=7.5)

    def _load_random_structure(self):
        from pathlib import Path

        import pandas as pd

        df = pd.read_csv(
            Path(__file__).parent.parent / "data" / "csg" / "raw" / "CSG_tiny.csv",
            usecols=["Structure"],
        )
        structures = df.sample(frac=0.05, random_state=42)["Structure"].to_list()

        return structures

    @profile
    def test_knn_performances(self):
        from tqdm import tqdm

        structs = self._load_random_structure()
        for struct in tqdm(structs):
            _ = self.knn.convert(struct, fmt="vasp")

if __name__ == "__main__":
    test = TestGraph()
    test.test_knn_performances()
