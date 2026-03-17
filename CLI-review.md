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

Update: The only solution I found to answer all the comments above is the use a custom Trainer subclass with less `__init__` arguments and custom docstring.

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

Answer: That's definitely a bug with the LightningCLI so I don't know if we can do much.

Update: Absolutely no idea why we have (side note this is from `python main fit --help` and not `python main --help`)

```text
Runs the full optimization routine:
  --ckpt_path CKPT_PATH
```

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

Answer: That's definitely a bug with the LightningCLI so I don't know if we can do much.

Update: It looks like `jsonargparse[signatures]` is only capturing the very first line of the docstring.
