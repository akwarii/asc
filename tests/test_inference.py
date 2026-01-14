import time
from copy import copy
from pathlib import Path

import torch
from line_profiler import profile
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit
from torch_geometric.loader import CachedLoader, NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import AddSelfLoops
from tqdm import tqdm


class SAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.output_layer = torch.nn.Linear(out_channels, out_channels)

    @property
    def num_layers(self) -> int:
        return len(self.convs)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x, edge_index) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.output_layer(x)
        return x

    @torch.no_grad()
    def inference_per_layer(
        self,
        layer: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        x = self.convs[layer](x, edge_index)[:batch_size]

        if layer == self.num_layers - 1:
            return x

        x = self.act(x)

        return x

    @torch.no_grad()
    @profile
    def inference(
        self,
        x_all,
        subloader: NeighborLoader,
        embedding_device: str = "cpu",
        *,
        cache: bool = True,
    ) -> torch.Tensor:
        assert isinstance(subloader, NeighborLoader)
        assert len(subloader.dataset) == subloader.data.num_nodes
        assert len(subloader.node_sampler.num_neighbors) == 1
        assert not self.training

        pbar = tqdm(total=len(subloader.dataset) * len(self.convs))
        pbar.set_description("Evaluating")

        def transform(data: Data) -> Data:
            kwargs = dict(n_id=data.n_id, batch_size=data.batch_size)
            if hasattr(data, 'adj_t'):
                kwargs['adj_t'] = data.adj_t
            else:
                kwargs['edge_index'] = data.edge_index
            return Data.from_dict(kwargs).to(self.device)

        if cache:
            loader = CachedLoader(subloader, device=self.device, transform=transform)
        else:
            loader = subloader

        for i in range(self.num_layers):
            xs = torch.empty(
                x_all.size(0),
                self.convs[i].out_channels,
                device=embedding_device,
                pin_memory=(embedding_device == "cpu"),
            )

            for batch in loader:
                batch_size = batch.batch_size

                n_id = batch.n_id.to(self.device)
                edge_index = batch.edge_index.to(self.device)

                x = x_all[n_id]
                x = self.inference_per_layer(i, x, edge_index, batch_size)

                global_id = batch.n_id.narrow(0, 0, batch_size)
                xs[global_id] = x.to(embedding_device, non_blocking=True)

                pbar.update(batch_size)

            if embedding_device == "cpu":
                torch.cuda.synchronize()
            x_all = xs.to(self.device)

        x_all = self.output_layer(x_all)

        pbar.close()
        return x_all


def prepare_datasets(device: str | torch.device):
    if isinstance(device, str):
        device = torch.device(device)

    dataset = Reddit(root="data/Reddit", pre_transform=AddSelfLoops())
    data: Data = dataset[0]  # type: ignore
    data = data.to(device, "x", "y")
    data = data.sort(sort_by_row=False)

    kwargs = {"batch_size": 1024, "pin_memory": True, "num_workers": 8}
    train_loader = NeighborLoader(
        data,
        is_sorted=True,
        num_neighbors=[25, 10],
        input_nodes=data.train_mask,
        shuffle=True,
        **kwargs,
    )

    subgraph_loader = NeighborLoader(
        copy(data),
        is_sorted=True,
        input_nodes=None,
        num_neighbors=[-1],
        shuffle=False,
        **kwargs,
    )

    # Add global node index information.
    subgraph_loader.data.num_nodes = data.num_nodes
    subgraph_loader.data.n_id = torch.arange(data.num_nodes)

    return dataset, train_loader, subgraph_loader


def train_step(
    model: torch.nn.Module,
    epoch: int,
    train_loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> tuple[float, float]:
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))  # type: ignore
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[: batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[: batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, total_correct / total_examples


def model_inference(
    model: torch.nn.Module,
    loader: NeighborLoader,
    device: torch.device,
    embedding_device: torch.device | str = "cpu",
    *,
    use_layerwise_inference: bool,
) -> torch.Tensor:
    has_layerwise_inference = hasattr(model, "inference") and callable(model.inference)

    with torch.inference_mode():
        if has_layerwise_inference and use_layerwise_inference:
            out = model.inference(loader.data.x.to(device), loader, embedding_device)  # type: ignore
        else:
            out = model(loader.data.x.to(device), loader.data.edge_index.to(device))

    return out  # type: ignore


@torch.no_grad()
def test_step(
    model: torch.nn.Module,
    subgraph_loader: NeighborLoader,
    embedding_device: str | torch.device = "cpu",
    *,
    use_layerwise_inference: bool = True,
) -> list[float]:
    model.eval()

    y_hat = model_inference(
        model,
        subgraph_loader,
        device=model.device,  # type: ignore
        embedding_device=embedding_device,
        use_layerwise_inference=use_layerwise_inference,
    ).argmax(dim=-1)
    data = subgraph_loader.data
    y = data.y.to(y_hat.device)  # type: ignore

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))

    return accs


def training_loop(model, train_loader, subgraph_loader, optimizer, device, num_epochs):
    times = []
    for epoch in range(1, num_epochs + 1):
        start = time.perf_counter()

        loss, acc = train_step(model, epoch, train_loader, optimizer, device)  # type: ignore
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}")

        train_acc, val_acc, test_acc = test_step(model, subgraph_loader)  # type: ignore
        print(
            f"Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, "
            f"Test: {test_acc:.4f}"
        )
        times.append(time.perf_counter() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")


@torch.no_grad()
def benchmark_model_inference(
    model: torch.nn.Module,
    loader: NeighborLoader,
    *,
    use_layerwise_inference: bool = True,
    runs: int = 10,
) -> None:
    """Benchmark inference time and VRAM usage."""
    model.eval()

    times: list[float] = []
    vram_vals: list[float] = []

    check_cuda_vram = torch.cuda.is_available() and model.device.type == "cuda"

    for _ in range(runs):
        if check_cuda_vram:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = model_inference(
            model,
            loader=loader,
            device=model.device,
            use_layerwise_inference=use_layerwise_inference,
        )
        end = time.perf_counter()

        if check_cuda_vram:
            torch.cuda.synchronize()
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_vram_mb = peak_bytes / (1024**2)
        else:
            peak_vram_mb = 0.0

        times.append(end - start)
        vram_vals.append(peak_vram_mb)

    if runs > 1:
        t_mean = torch.tensor(times).mean().item()
        t_std = torch.tensor(times).std().item()
        v_mean = torch.tensor(vram_vals).mean().item()
        v_std = torch.tensor(vram_vals).std().item()
    else:
        t_mean = times[0]
        t_std = 0.0
        v_mean = vram_vals[0]
        v_std = 0.0

    print(
        f"Benchmark Results - Layer-wise inference {use_layerwise_inference}:\n"
        f"Time: {t_mean:.4f} ± {t_std:.4f} s\n"
        f"VRAM: {v_mean:.2f} ± {v_std:.2f} MB\n"
    )
    print("Detailed results:\n" f"Times: {times}\n" f"VRAM usages: {vram_vals}\n")


def main() -> None:
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    num_epochs = 3
    retrain = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset, train_loader, subgraph_loader = prepare_datasets(device)

    model = SAGE(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=dataset.num_classes,
        dropout=0.3,
    ).to(device)
    model = torch.compile(model, fullgraph=True)

    model_weights_path = Path("sage_reddit.pth")
    if retrain:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)  # type: ignore
        training_loop(
            model,
            train_loader,
            subgraph_loader,
            optimizer,
            device,
            num_epochs,
        )
        torch.save(model.state_dict(), model_weights_path)  # type: ignore
    elif not retrain and model_weights_path.exists():
        del train_loader
        model.load_state_dict(torch.load(model_weights_path, weights_only=True))  # type: ignore
    else:
        raise FileNotFoundError("Model weights not found. Set retrain=True to train the model.")

    # accs = test_step(model, subgraph_loader, embedding_device="cpu")
    # print(f"Final Test Accuracies: Train: {accs[0]:.4f}, Val: {accs[1]:.4f}, Test: {accs[2]:.4f}")
    benchmark_model_inference(
        model,  # type: ignore
        subgraph_loader,
        use_layerwise_inference=True,
        runs=1,
    )


if __name__ == "__main__":
    main()
