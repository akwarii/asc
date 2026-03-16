# CLI arguments review

Remarks collected while reading the code and running the CLI on commit `7c1d989c773fa3fa7ff01b29f302d2cece467b1f` (2026-03-13).

## `fit` subprogram

Arguments checked by using:

```bash
python main.py fit --help
```

- I don't understand what `--trainer.overfit_batches` does. The description repeats the argument name but doesn't explain what it does. Also, I wonder whether the practice of voluntarily overfitting a model is common enough to deserve a dedicated argument.

Answer: this parameter is only useful for quickly debugging or trying to overfit on purpose. It appears because it's part of the trainer parameters and I didn't remove any of them yet. We have to list the ones we want to exclude.

```text
--trainer.overfit_batches OVERFIT_BATCHES
                        Overfit a fraction of training/validation data (float) or a set number of batches (int). Default: ``0.0``. (type: Union[int, float], default: 0.0)
```

- Would it make sense to allow a much finer control over the model summary? We allow passing `RichModelSummary` as a callback, but we only have a boolean argument to enable/disable the model summary. Maybe the depth of the model summary could be controlled by an integer argument, to see where exactly the model is getting too big ?

Answer: directly from the CLI I don't know, probably it's possible but it's definitely not a priority since we can pass the argument from a config file (see configs/painn.yaml).

```text
  --trainer.enable_model_summary {true,false,null}
                        Whether to enable model summarization by default. Default: ``True``. (type: Optional[bool], default: null)
```

- I think `--trainer.accumulate_grad_batches` could be a bit more explained. It is a key feature, allowing to mimic larger batch sizes, and we saw during our tests that it of course has a noticeable impact on the training quality. For now, the description is a bit too straightforward and doesn't really explain the concept of gradient accumulation: i.e., it's easy to miss for someone who is not familiar with it.

Answer: the descriptions provided by the cli help message are extracted from the docstrings so I didn't modify anything. Probably we can tune the displayed text but, as above, it's not a priority right now.

```text
  --trainer.accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Accumulates gradients over k batches before stepping the optimizer. Default: 1. (type: int, default: 1)
```

- The list of accepted model names should be documented.

Answer: I will do it

```text
    --model.model_name MODEL_NAME
                        The name of the model. (required, type: str)
```

- I find the description of `--model.model_kwargs` a bit too vague. I understand this behaviour is inherited from the default `LightningCLI`, but in our case a lot of features are only accessible through this argument, and it would be nice to have a more detailed description of how to use it.

Answer: while I agree I can't think of a good solution since the kwargs will depend on the model and we can't know in advance which model the user will select.

```text
    --model.model_kwargs MODEL_KWARGS
                        Additional keyword arguments for the model. Defaults to None. (type: Optional[dict[str, Any]], default: null)
```

- The `--data.dataset` argument description should be pruned, as many of the given dataset examples are not actually what the code is meant for. Furthermore, the `--data.dataset_name` argument already does what we want to do here.

Answer: I totaly agree. This behavior is due to the type hints so the easiest solution might be for us to create a base class that inherit from InMemoryDataset and use it for type hints.

```text
      --data.dataset DATASET
                        The dataset to use for training. If lengths are provided, the dataset is split into training, validation, and test datasets (default: `None`). (type: Dataset |
                        None, default: null, known subclasses: torch_geometric.data.Dataset, torch_geometric.data.InMemoryDataset, torch_geometric.datasets.KarateClub,
                        torch_geometric.datasets.TUDataset, torch_geometric.datasets.GNNBenchmarkDataset, torch_geometric.datasets.Planetoid, torch_geometric.datasets.NELL,
                        torch_geometric.datasets.CitationFull, torch_geometric.datasets.CoraFull, torch_geometric.datasets.Coauthor, torch_geometric.datasets.Amazon,
                        torch_geometric.datasets.PPI, torch_geometric.datasets.Reddit, torch_geometric.datasets.Reddit2, torch_geometric.datasets.Flickr, torch_geometric.datasets.Yelp,
                        torch_geometric.datasets.AmazonProducts, torch_geometric.datasets.QM7b, torch_geometric.datasets.QM9, torch_geometric.datasets.MD17, torch_geometric.datasets.ZINC,
                        torch_geometric.datasets.AQSOL, torch_geometric.datasets.MoleculeNet, torch_geometric.datasets.Entities, torch_geometric.datasets.RelLinkPredDataset,
                        torch_geometric.datasets.GEDDataset, torch_geometric.datasets.AttributedGraphDataset, torch_geometric.datasets.MNISTSuperpixels, torch_geometric.datasets.FAUST,
                        torch_geometric.datasets.DynamicFAUST, torch_geometric.datasets.ShapeNet, torch_geometric.datasets.ModelNet, torch_geometric.datasets.MedShapeNet,
                        torch_geometric.datasets.CoMA, torch_geometric.datasets.SHREC2016, torch_geometric.datasets.TOSCA, torch_geometric.datasets.PCPNetDataset,
                        torch_geometric.datasets.S3DIS, torch_geometric.datasets.GeometricShapes, torch_geometric.datasets.BitcoinOTC, torch_geometric.datasets.GDELTLite,
                        torch_geometric.datasets.icews.EventDataset, torch_geometric.datasets.ICEWS18, torch_geometric.datasets.GDELT, torch_geometric.datasets.WILLOWObjectClass,
                        torch_geometric.datasets.PascalVOCKeypoints, torch_geometric.datasets.PascalPF, torch_geometric.datasets.SNAPDataset,
                        torch_geometric.datasets.SuiteSparseMatrixCollection, torch_geometric.datasets.WordNet18, torch_geometric.datasets.WordNet18RR, torch_geometric.datasets.FB15k_237,
                        torch_geometric.datasets.WikiCS, torch_geometric.datasets.WebKB, torch_geometric.datasets.WikipediaNetwork, torch_geometric.datasets.HeterophilousGraphDataset,
                        torch_geometric.datasets.Actor, torch_geometric.datasets.UPFD, torch_geometric.datasets.GitHub, torch_geometric.datasets.FacebookPagePage,
                        torch_geometric.datasets.LastFMAsia, torch_geometric.datasets.DeezerEurope, torch_geometric.datasets.GemsecDeezer, torch_geometric.datasets.Twitch,
                        torch_geometric.datasets.Airports, torch_geometric.datasets.LRGBDataset, torch_geometric.datasets.MalNetTiny, torch_geometric.datasets.OMDB,
                        torch_geometric.datasets.PolBlogs, torch_geometric.datasets.EmailEUCore, torch_geometric.datasets.LINKXDataset, torch_geometric.datasets.EllipticBitcoinDataset,
                        torch_geometric.datasets.EllipticBitcoinTemporalDataset, torch_geometric.datasets.DGraphFin, torch_geometric.datasets.HydroNet,
                        torch_geometric.datasets.hydro_net.Partition, torch_geometric.datasets.AirfRANS, torch_geometric.datasets.JODIEDataset, torch_geometric.datasets.Wikidata5M,
                        torch_geometric.datasets.MyketDataset, torch_geometric.datasets.BrcaTcga, torch_geometric.datasets.NeuroGraphDataset,
                        torch_geometric.datasets.web_qsp_dataset.KGQABaseDataset, torch_geometric.datasets.WebQSPDataset, torch_geometric.datasets.CWQDataset,
                        torch_geometric.datasets.GitMolDataset, torch_geometric.datasets.MoleculeGPTDataset, torch_geometric.datasets.InstructMolDataset,
                        torch_geometric.datasets.ProteinMPNNDataset, torch_geometric.datasets.TAGDataset, torch_geometric.datasets.CityNetwork, torch_geometric.datasets.Teeth3DS,
                        torch_geometric.datasets.DBP15K, torch_geometric.datasets.AMiner, torch_geometric.datasets.OGB_MAG, torch_geometric.datasets.DBLP,
                        torch_geometric.datasets.MovieLens, torch_geometric.datasets.MovieLens100K, torch_geometric.datasets.MovieLens1M, torch_geometric.datasets.IMDB,
                        torch_geometric.datasets.LastFM, torch_geometric.datasets.HGBDataset, torch_geometric.datasets.Taobao, torch_geometric.datasets.IGMCDataset,
                        torch_geometric.datasets.AmazonBook, torch_geometric.datasets.HM, torch_geometric.datasets.OSE_GVCS, torch_geometric.datasets.RCDD,
                        torch_geometric.datasets.OPFDataset, torch_geometric.datasets.CornellTemporalHyperGraphDataset, torch_geometric.datasets.FakeDataset,
                        torch_geometric.datasets.FakeHeteroDataset, torch_geometric.datasets.StochasticBlockModelDataset, torch_geometric.datasets.RandomPartitionGraphDataset,
                        torch_geometric.datasets.MixHopSyntheticDataset, torch_geometric.datasets.ExplainerDataset, torch_geometric.datasets.InfectionDataset,
                        torch_geometric.datasets.BA2MotifDataset, torch_geometric.datasets.BAMultiShapesDataset, torch_geometric.datasets.ba_shapes.BAShapes,
                        torch_geometric.deprecation.BAShapes, src.datasets.Aflow, src.datasets.CSG, src.datasets.CustomDataset, src.datasets.Gnome, src.datasets.MaterialProject,
                        torch_geometric.data.OnDiskDataset, torch_geometric.datasets.PCQM4Mv2)

    --data.dataset_name DATASET_NAME
                         The name of the dataset to use. It can be either `aflow`, `csg`, `custom`, `gnome`, or `mp`. If `dataset` is provided, this argument is ignored (default: `None`).
                        (type: str | None, default: null)
```

- The same goes for `--data.pred_dataset`, where only a few of the datasets quoted in the description can be used in practice by the code.

Answer: same answer as above

- The description for `--data.pre_transforms` and `--data.transforms` could be more detailed, and in particular present the transforms implemented in the code with examples of how to use them. For now, the description is a bit too vague and doesn't present these arguments, nor does it explain how to use them. It is also not ideal that `--data.pre_transforms.help` and `--data.transforms.help` already require knowing the transforms beforehand, as many users may just not know and skip these (important) features altogether.

Answer: I agree. The easy solution is to update the docstring but we can't detail every transform in the docstring nor in the help message as it will be too much.

```text
  --data.pre_transforms.help
                        Show the help for object and exit.
  --data.pre_transforms PRE_TRANSFORMS
                        A function or a list of functions that takes in a `~torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before
                        being saved to disk (default: `None`). (type: Optional[object], default: null)
  --data.transforms.help
                        Show the help for object and exit.
  --data.transforms TRANSFORMS
                        A function or a list of functions that takes in a `torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before
                        every access (default: `None`). (type: Optional[object], default: null)
```

> **General remark**
> I think the documentation of the CLI arguments is a bit too heavy overall, with many extremely specific (yet sometimes useful) arguments overshadowing the most important ones. Maybe we could have a "basic" CLI with only the most important arguments, and an "advanced" CLI with all the possible arguments for users who want to have more control over the training process? This would make the documentation easier to read and understand for new users, while still allowing advanced users to have access to all the features of the code.

Answer: It will be a pain to keep 2 different CLIs. I think the best is to trim the arguments we don't want / feel are not important.

## `validate`, `test` and `predict` subprograms

Arguments checked by using:

```bash
python main.py validate --help
python main.py test --help
python main.py predict --help
```

My remarks for these three subprograms are mostly the same as for the `fit` subprogram, as they share a lot of arguments. In particular, the arguments related to the dataset could be better documented, and the list of accepted model and dataset names should be documented as well. Also, maybe a section related to the use (or lack thereof) `pre_transforms` and `transforms` in these subprograms should be added.

## Main `help` command

Obtained by using:

```bash
python main.py --help
```

Returns:

```text
usage: main.py [-c CONFIG] {fit,validate,test,predict} ...

Lightning Trainer command line tool

options:
  -h, --help            Show this help message and exit.
  -c, --config CONFIG   Path to a configuration file in json or yaml format.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords separated by comma. The
                        supported flags are: skip_default, skip_null.

subcommands:
  For more details of each subcommand, add it as an argument followed by --help.

  Available subcommands:
    fit                 Runs the full optimization routine.
    validate            Perform one evaluation epoch over the validation set.
    test                Perform one evaluation epoch over the test set. It's separated from fit to make sure you never run on your
    predict             Run inference on your data. This will call the model forward function to compute predictions. Useful to
```

We see that the main `help` command is quite clear and concise. However, the descriptions of the `test` and `predict` subcommands are cut off.

I also notice that those short descriptions (eg. "Runs the full optimization routine") are printed in an incorrect place (just before checkpoint arguments description) when displaying the help for their respective subcommands. For example, when running `python main.py fit --help`, we get:

```text
usage: main.py [options] fit [-c CONFIG] [...]

  Runs the full optimization routine.

  options:
    -h, --help            Show this help message and exit.
    -c, --config CONFIG   Path to a configuration file in json or yaml format.
    --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output and are one or more keywords separated by comma. The
                        supported flags are: skip_default, skip_null.

    [... LOTS OF OTHER ARGUMENTS ...]

Runs the full optimization routine:
  --ckpt_path CKPT_PATH
                        [...]
```

This looks like a bug, probably in the `LightningCLI` class we inherit from.

***IF*** we decide to have a "basic" and an "advanced" CLI, it would be nice for it to be documented in the main `help` command.

Answer: That's definitely a bug with the LightningCLI so I don't know if we can do much.
