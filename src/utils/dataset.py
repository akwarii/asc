import math
import warnings
from collections.abc import Sequence
from itertools import accumulate

from torch_geometric.data import Dataset


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

    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = math.floor(len(dataset) * frac)
            subset_lengths.append(n_items_in_split)

        remainder = len(dataset) - sum(subset_lengths)

        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1

        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    return [
        dataset[int(offset - length) : int(offset)]
        for offset, length in zip(accumulate(lengths), lengths)
    ]  # type: ignore
