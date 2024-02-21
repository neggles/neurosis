# neurosis

a neural network trainer for ~~weebs~~ diffusion models.

## OwO what's this?

I got sick and tired of trying to trace execution through the arcane codepaths of existing Stable Diffusion trainers, so I wrote my own.
This is based off the modeling/training code from [@Stability-AI/generative-models](https://github.com/Stability-AI/generative-models),
with *significant* modification for usability and readability.

## Features

Major changes from `generative-models` include:

- Architectural changes:
  - Migration to full PyTorch Lightning
  - LightningCLI trainer interface and configuration management (config file format is largely the same as `generative-models`, but with some minor differences in keywords)
  - Refactoring of some of the configuration code and model subclassing
  - "oops, all wandb" approach to logging (with some TensorBoard support as well)
- Trainer changes:
  - Use of PyTorch Lightning's `Trainer` class for training
  - Support for multiple GPUs (and multiple nodes, if you're into that)
  - Support for individual learning rates for the UNet and for each TE module
  - VAE training support! (kinda! discriminators are iffy but it works mostly)
- Module changes:
  - Rework of the `ImageLogger` to... sorta kinda work?
  - Adding support for `Adafactor` scheduler as well as the usual BitsAndBytes etc. ones
  - hey look tag frequency based loss scaling wonder where we got that one from :eyes:
  - Cleanup and refactoring of most modules to make them more readable and easier to trace execution
  - Probably more duplicated code than there really should be but here we are in this hell timeline
  - A huge pile of small changes too numerous to mention
- Dataset handling:
  - Support for Huggingface datasets (kinda! you're on your own but it should work if the keys match)
  - Shiny new ImageFolder datasets in "square", "square with captions", and "aspect-bucketed with captions" flavors
  - Support for custom datasets (see `neurosis/dataset/` for examples)
  - Funny hybrid mongo+s3 dataset we're using for large-scale training (stop judging me, it works)
  - Support for custom data transform functions injected into the dataset pipeline[^1]

[^1]: completely untested and not entirely pushed to public yet sry

## Installation

OK so first of all:

1. **Do Not.**

if you REALLY must, here:

```bash
git clone https://github.com/neggles/neurosis.git neurosis
cd neurosis
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel setuptools-scm
python -m pip install -e '.[all]'
```

There is a docker container in GitHub packages, if you're feeling masochistic today.

Will also be throwing up Kubeflow manifest examples at some point (needs more testing)

## Usage

I am once again asking,

1. Please, *please just **Do Not***

but if the idea of arguing with my code and trying to figure shit out from incomplete outdated configuration templates
sounds attractive to you, by all means:

```bash
python -m neurosis.trainer.cli --help
```

If you need more assistance than that, this code is probably not ready for you to use yet.

### If you open an issue without providing liberal amounts of detail, logs, exactly what you tried, and exactly how it broke, you're going to get WONTFIX'd for the time being.

This will change as/if/when the code gets a bit more stable and usable.

## License

My own code is GPLv3, see [LICENSE.md](LICENSE.md)

A significant amount of code is copied from from [@Stability-AI/generative-models](https://github.com/Stability-AI/generative-models),which is MIT licensed and has been relicensed here under GPLv3 due to the extensive modifications.

Some code carries its own licenses (most notably LPIPS and Adafactor),
see the appropriate SPDX identifiers and LICENSE files in the relevant files' directories.
