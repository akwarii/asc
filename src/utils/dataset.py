import math
import warnings
from collections import Counter
from collections.abc import Sequence
from itertools import accumulate
from numbers import Real

import torch
from torch_geometric.data import Dataset


def _split_lengths(n: int, lengths: Sequence[int | float]) -> list[int]:
    """Resolve a sequence of integer or fractional split lengths into exact integer sizes.

    When the lengths are fractions that sum to 1, each split size is ``floor(frac * n)`` with any
    remainder distributed round-robin, matching the historical behaviour of ``random_split``.

    Args:
        n: Total number of samples in the dataset.
        lengths: Integer lengths or fractions of the dataset, summing to ``n`` (or 1).

    Returns:
        list[int]: Exact per-split sample counts summing to ``n``.

    Raises:
        ValueError: If the lengths do not sum to ``n`` (or to 1 when given as fractions), or a
            fraction is outside ``[0, 1]``.
    """
    total = sum(lengths)

    # Case 1: lengths are fractions that sum to 1 -> compute exact lengths + distribute remainder
    if math.isclose(total, 1):
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if not isinstance(frac, Real) or frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = math.floor(n * frac)
            subset_lengths.append(n_items_in_split)

        remainder = n - sum(subset_lengths)

        # add 1 to all the lengths (each after the other) until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1

        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. This might result in an empty dataset.",
                    stacklevel=2,
                )
    else:
        if any(
            not isinstance(length, Real) or not float(length).is_integer() for length in lengths
        ):
            raise ValueError("Integer split lengths must be whole numbers.")
        if any(length < 0 for length in lengths):
            raise ValueError("Split lengths must be non-negative.")

    # Case 2: lengths are integers that sum to n -> return as-is
    if sum(lengths) != n:
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    return [int(length) for length in lengths]


def random_split(
    dataset: Dataset,
    lengths: Sequence[int | float],
) -> list[Dataset]:
    r"""Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given, the lengths will be computed automatically
    as floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be distributed in
    round-robin fashion to the lengths until there are no remainders left.

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced

    Returns:
        list[Dataset]: List of datasets of provided lengths
    """
    dataset = dataset.shuffle()  # type: ignore
    split_lengths = _split_lengths(len(dataset), lengths)  # type: ignore[arg-type]

    return [
        dataset[int(offset - length) : int(offset)]
        for offset, length in zip(accumulate(split_lengths), split_lengths)
    ]  # type: ignore


def graph_labels(dataset: Dataset) -> torch.Tensor:
    """Return the per-graph class label of every sample as a ``[num_graphs]`` long tensor.

    Each graph (crystal) carries a single class label broadcast to all of its atoms, which is
    stored on the first atom (``data.y[0]``). For in-memory datasets the labels are collected from
    the shared concatenated buffer in one vectorised pass; a per-graph Python loop is used as a
    fallback otherwise.

    Args:
        dataset: The dataset to read the labels from.

    Returns:
        torch.Tensor: A ``torch.long`` tensor of shape ``[len(dataset)]`` with each graph's class.
    """
    buffer = getattr(dataset, "_data", None)

    # InMemoryDataset : gather the labels from the concatenated buffer all at once
    if buffer is not None and hasattr(buffer, "_num_nodes"):
        # Per-graph node counts give the first-node offset in the concatenated buffer.
        nodes_per_graph = torch.as_tensor(buffer._num_nodes, dtype=torch.long)
        first_node_offsets = torch.cumsum(nodes_per_graph, 0) - nodes_per_graph

        # The class of a whole crystal equals the label of its first atom (data.y[0]).
        all_labels = torch.as_tensor(buffer.y)[first_node_offsets].reshape(-1).to(torch.long)
        selected_indices = torch.as_tensor(dataset.indices(), dtype=torch.long)
        return all_labels[selected_indices]

    # Fallback : gather the labels graph by graph with a loop (slower, should always work)
    return torch.tensor(
        [int(dataset[i].y[0]) for i in range(len(dataset))],  # type: ignore[arg-type]
        dtype=torch.long,
    )


def stratified_split(
    dataset: Dataset,
    lengths: Sequence[int | float],
    *,
    seed: int | None = None,
) -> list[Dataset]:
    """Split a dataset into subsets with approximately matching class distributions.

    Stratification is performed per graph (each crystal may contain many atoms sharing one class),
    so the samples of every class are distributed across the splits following ``lengths`` as
    closely as possible. The algorithm is deliberately simple:

    - Each class is shuffled deterministically.
    - Classes with no more graphs than the number of splits are split trivially: one sample is
      given to each split in priority order (train, then validation, then test), so a
      single-sample class goes entirely to training and a two-sample class fills training and
      validation.
    - Larger classes are split proportionally using the same size computation as ``random_split``.

    Per-split total sizes are then reconciled to exactly match those produced by ``random_split``
    for the same ``lengths``, which also smooths out skew from small classes (e.g. a class that
    landed only in training). Reconciliation drains the split's most populous class first to
    preserve scarce classes, but can still empty a class from a split when forced to meet the
    exact target sizes (e.g. a lone tiny dataset collapses back towards training).

    The sampling is deterministic for a given ``seed``. When ``seed`` is ``None``, the global Torch
    generator is used, matching the default random-number behavior of ``random_split``.

    Args:
        dataset: Dataset to split.
        lengths: Integer lengths or fractions of the dataset to be produced.
        seed: Optional seed for reproducible sampling. Defaults to None (uses the global
            generator).

    Returns:
        list[Dataset]: List of stratified datasets of the requested lengths.
    """
    n = len(dataset)
    n_splits = len(lengths)
    targets = _split_lengths(n, lengths)  # type: ignore[arg-type]

    labels = graph_labels(dataset).tolist()

    # Group local dataset indices by class. Sorted class keys make seeded output reproducible.
    indices_by_class: dict[int, list[int]] = {}
    for index, cls in enumerate(labels):
        indices_by_class.setdefault(cls, []).append(index)

    # Handles the case where a seed is provided, otherwise uses the default generator
    generator = (
        torch.Generator().manual_seed(seed) if seed is not None else torch.default_generator
    )

    # Create empty buckets for each split to hold the indices of the samples assigned to that split
    buckets: list[list[int]] = [[] for _ in range(n_splits)]

    # Distribute the samples of each class one by one into the buckets.
    for cls in sorted(indices_by_class):
        indices = indices_by_class[cls]  # Indices of samples from the current class
        class_size = len(indices)  # Number of samples in the current class

        # Shuffle the group deterministically so the assignment order is reproducible.
        perm = torch.randperm(class_size, generator=generator).tolist()
        indices = [indices[p] for p in perm]

        # ? Trivial case: less (or as many) samples than splits
        # ?     Fill train first, then validation, then test (same order as `lengths`)
        if class_size <= n_splits:
            for split in range(class_size):
                buckets[split].append(indices[split])
            continue

        # ? Most common case: more samples than splits
        # ?  -> Proportional split, reusing the size computation of `random_split`.
        # The indices of cls spreads across the splits following `lengths`.
        # Small non-trivial classes may still be skewed (e.g. empty val/test)
        # but `_reconcile` will handle it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            split_sizes = _split_lengths(class_size, lengths)  # type: ignore[arg-type]
        cursor = 0
        for split, size in enumerate(split_sizes):
            buckets[split].extend(indices[cursor : cursor + size])
            cursor += size

    # Exact split sizes are mandatory; class presence is preserved when possible.
    _reconcile(buckets, targets, labels)

    return [dataset[sorted(ids)] for ids in buckets]  # type: ignore[return-value]


def _reconcile(buckets: list[list[int]], targets: list[int], labels: list[int]) -> None:
    """Adjust bucket sizes in place until they equal ``targets``, deterministically.

    Global per-class allocation using integer rounding can leave each split one or two samples
    off its exact target. This pass moves surplus samples into deficit splits, preferring to take
    them from the split's most populous class so scarce classes are drained last. Exact target
    sizes take priority: if every class in the surplus split is a singleton, the final sample is
    removed regardless of class.

    Args:
        buckets: Per-split index lists; mutated in place.
        targets: Exact per-split target sizes.
        labels: Per-graph class label (indexed by the global graph index) used to select the most
            populous class of a split and to maintain per-split class counts.
    """
    n_splits = len(buckets)
    sizes = [len(bucket) for bucket in buckets]
    class_counts = [Counter(labels[index] for index in bucket) for bucket in buckets]

    while True:
        # Which splits are under- or over-sized?
        deficits = [split for split in range(n_splits) if sizes[split] < targets[split]]
        surplus = [split for split in range(n_splits) if sizes[split] > targets[split]]

        # If no splits are under- or over-sized, we're done.
        if not deficits and not surplus:
            break

        # Find the largest deficit and largest surplus splits
        deficit_split = max(deficits, key=lambda split: targets[split] - sizes[split])
        surplus_split = max(surplus, key=lambda split: sizes[split] - targets[split])

        bucket = buckets[surplus_split]  # The list of indices in the surplus split

        # Choose which sample to move:
        #   - From the most populous class in the surplus split (preserve scarce classes)
        #   - If there are multiple classes candidates -> last one in the bucket (deterministic)
        #   - In the selected class, choose the last sample in the bucket (deterministic)
        #  When every class is a singleton (all counts 1) this degrades to the fallback
        #       len(bucket) - 1
        #  so exact sizes still win over preserving class presence.
        counts = class_counts[surplus_split]
        max_count = max(counts.values())
        position = next(
            (
                pos
                for pos in range(len(bucket) - 1, -1, -1)  # last sample from
                if counts[labels[bucket[pos]]] == max_count  # the largest class (last in order)
            ),
            len(bucket) - 1,  # Default to the last sample if all classes are singletons (count 1)
        )
        index = bucket.pop(position)  # Remove from surplus split and keep the index
        buckets[deficit_split].append(index)  # Add to deficit split

        # Update the class counts
        cls = labels[index]
        class_counts[surplus_split][cls] -= 1
        class_counts[deficit_split][cls] += 1

        # Update the sizes
        sizes[surplus_split] -= 1
        sizes[deficit_split] += 1
