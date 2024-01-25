# Modified from PyTorch Lightning code by Andi Powers-Holmes <aholmes@omnom.net>
# for the purposes of "make this work with any nn.Module instead of just LightningModules"
#
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities related to model weights summary."""

import logging
from collections import OrderedDict
from itertools import chain
from typing import Optional

import torch
from humanfriendly import format_size
from lightning.pytorch.utilities.model_summary.model_summary import (
    LayerSummary,
    _is_lazy_weight_tensor,
    get_human_readable_count,
)
from lightning.pytorch.utilities.rank_zero import WarningCache
from torch import Tensor, nn

log = logging.getLogger(__name__)
warning_cache = WarningCache()

PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]
UNKNOWN_SIZE = "?"
LEFTOVER_PARAMS_NAME = "other params"
NOT_APPLICABLE = "n/a"

MBYTE = 2**20


class ModelSummary:
    def __init__(
        self,
        model: nn.Module,
        max_depth: int = 1,
        example_input: Optional[Tensor | dict[str, Tensor]] = None,
    ) -> None:
        if not isinstance(max_depth, int) or max_depth < -1:
            raise ValueError(f"`max_depth` can be -1, 0 or > 0, got {max_depth}.")

        self._model: nn.Module = model
        self._max_depth = max_depth
        self._example_input = example_input
        self._layer_summary = self.summarize()

    @property
    def named_modules(self) -> list[tuple[str, nn.Module]]:
        mods: list[tuple[str, nn.Module]]
        if self._max_depth == 0:
            mods = []
        else:
            mods = self._model.named_modules()
        return mods

    @property
    def layer_names(self) -> list[str]:
        return list(self._layer_summary.keys())

    @property
    def layer_types(self) -> list[str]:
        return [layer.layer_type for layer in self._layer_summary.values()]

    @property
    def in_sizes(self) -> list:
        return [layer.in_size for layer in self._layer_summary.values()]

    @property
    def out_sizes(self) -> list:
        return [layer.out_size for layer in self._layer_summary.values()]

    @property
    def param_nums(self) -> list[int]:
        return [layer.num_parameters for layer in self._layer_summary.values()]

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters() if not _is_lazy_weight_tensor(p))

    @property
    def trainable_parameters(self) -> int:
        return sum(
            p.numel() for p in self._model.parameters() if p.requires_grad and not _is_lazy_weight_tensor(p)
        )

    @property
    def total_layer_params(self) -> int:
        return sum(self.param_nums)

    @property
    def model_size_bytes(self) -> float:
        param_bytes = sum(
            p.numel() * p.element_size()
            for p in chain(self._model.parameters(), self._model.buffers())
            if not _is_lazy_weight_tensor(p)
        )
        return param_bytes

    def summarize(self) -> dict[str, LayerSummary]:
        summary = OrderedDict((name, LayerSummary(module)) for name, module in self.named_modules)
        if self._example_input is not None:
            self._forward_example_input()
        for layer in summary.values():
            layer.detach_hook()

        if self._max_depth >= 0:
            # remove summary entries with depth > max_depth
            for k in [k for k in summary if k.count(".") >= self._max_depth]:
                del summary[k]
        return summary

    def _forward_example_input(self) -> None:
        """Run the example input through each layer to get input- and output sizes."""
        if self._example_input is None:
            raise ValueError("No example input was provided.")

        with torch.inference_mode():
            # let the model hooks collect the input- and output shapes
            if isinstance(self._example_input, (list, tuple)):
                self._model(*self._example_input)
            elif isinstance(self._example_input, dict):
                self._model(**self._example_input)
            else:
                self._model(self._example_input)

    def _get_summary_data(self) -> list[tuple[str, list[str]]]:
        """Makes a summary listing with:

        Layer Name, Layer Type, Number of Parameters, Input Sizes, Output Sizes, Model Size

        """
        arrays = [
            (" ", list(map(str, range(len(self._layer_summary))))),
            ("Name", self.layer_names),
            ("Type", self.layer_types),
            ("Params", list(map(get_human_readable_count, self.param_nums))),
        ]
        if self._example_input is not None:
            arrays.append(("In sizes", [str(x) for x in self.in_sizes]))
            arrays.append(("Out sizes", [str(x) for x in self.out_sizes]))

        total_leftover_params = self.total_parameters - self.total_layer_params
        if total_leftover_params > 0:
            self._add_leftover_params_to_summary(arrays, total_leftover_params)

        return arrays

    def _add_leftover_params_to_summary(
        self, arrays: list[tuple[str, list[str]]], total_leftover_params: int
    ) -> None:
        """Add summary of params not associated with module or layer to model summary."""
        layer_summaries = dict(arrays)
        layer_summaries[" "].append(" ")
        layer_summaries["Name"].append(LEFTOVER_PARAMS_NAME)
        layer_summaries["Type"].append(NOT_APPLICABLE)
        layer_summaries["Params"].append(get_human_readable_count(total_leftover_params))
        if "In sizes" in layer_summaries:
            layer_summaries["In sizes"].append(NOT_APPLICABLE)
        if "Out sizes" in layer_summaries:
            layer_summaries["Out sizes"].append(NOT_APPLICABLE)

    def __str__(self) -> str:
        arrays = self._get_summary_data()

        total_parameters = self.total_parameters
        trainable_parameters = self.trainable_parameters
        model_size_bytes = self.model_size_bytes

        return _format_summary_table(total_parameters, trainable_parameters, model_size_bytes, *arrays)

    def __repr__(self) -> str:
        return str(self)


def _format_summary_table(
    total_parameters: int,
    trainable_parameters: int,
    model_size_bytes: int,
    *cols: tuple[str, list[str]],
) -> str:
    """Takes in a number of arrays, each specifying a column in the summary table, and combines them all into one big
    string defining the summary table that are nicely formatted."""
    n_rows = len(cols[0][1])
    n_cols = 1 + len(cols)

    # Get formatting width of each column
    col_widths = []
    for c in cols:
        col_width = max(len(str(a)) for a in c[1]) if n_rows else 0
        col_width = max(col_width, len(c[0]))  # minimum length is header length
        col_widths.append(col_width)

    # Formatting
    s = "{:<{}}"
    total_width = sum(col_widths) + 3 * n_cols
    header = [s.format(c[0], w) for c, w in zip(cols, col_widths)]

    # Summary = header + divider + Rest of table
    summary = " | ".join(header) + "\n" + "-" * total_width
    for i in range(n_rows):
        line = []
        for c, w in zip(cols, col_widths):
            line.append(s.format(str(c[1][i]), w))
        summary += "\n" + " | ".join(line)
    summary += "\n" + "-" * total_width

    params_size = format_size(model_size_bytes, binary=True)
    f_width = max(len(str(params_size)) + 1, 10)

    summary += "\n" + s.format(get_human_readable_count(trainable_parameters), f_width)
    summary += "Trainable params"
    summary += "\n" + s.format(get_human_readable_count(total_parameters - trainable_parameters), f_width)
    summary += "Non-trainable params"
    summary += "\n" + s.format(get_human_readable_count(total_parameters), f_width)
    summary += "Total params"
    summary += "\n" + s.format(params_size, f_width)
    summary += "Total estimated model params size (MB)"

    return summary


def summarize(module: nn.Module, max_depth: int = 1) -> ModelSummary:
    """Summarize the nn.Module specified by `lightning_module`.

    Args:
        module: `nn.Module` to summarize.

        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
            layer summary off. Default: 1.

    Return:
        The model summary object

    """
    return ModelSummary(module, max_depth=max_depth)
