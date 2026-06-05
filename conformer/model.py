# Copyright (c) 2021, Soohwan Kim. All rights reserved.
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

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from .encoder import ConformerEncoder
from .modules import Linear


class Conformer(nn.Module):
    """
    Conformer: Convolution-augmented Transformer for Speech Recognition
    The paper used a one-lstm Transducer decoder, currently still only implemented
    the conformer encoder shown in the paper.

    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_encoder_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
        input_layout (str, optional): Default input layout. Supports ``"btc"`` for
            ``(batch, time, features)``, ``"bft"`` for ``(batch, features, time)``,
            and ``"bcft"`` for image-like tensors ``(batch, channels, features, time)``.

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim), (batch, dim, time), or (batch, channels, dim, time):
          Tensor containing a sequence or image-like feature map whose x-axis is time and y-axis is features
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer.
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(
            self,
            num_classes: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_encoder_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            input_layout: str = "btc",
    ) -> None:
        super(Conformer, self).__init__()
        self.input_dim = input_dim
        self.input_layout = input_layout
        self.encoder = ConformerEncoder(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        )
        self.fc = Linear(encoder_dim, num_classes, bias=False)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)

    def _prepare_inputs(self, inputs: Tensor, input_layout: str) -> Tensor:
        """Convert supported input layouts into (batch, time, features)."""
        if input_layout not in ("auto", "btc", "bft", "bcft"):
            raise ValueError("input_layout must be one of: 'auto', 'btc', 'bft', 'bcft'")

        if inputs.dim() == 3:
            if input_layout == "auto":
                if inputs.size(-1) == self.input_dim and inputs.size(1) != self.input_dim:
                    input_layout = "btc"
                elif inputs.size(1) == self.input_dim and inputs.size(-1) != self.input_dim:
                    input_layout = "bft"
                else:
                    input_layout = self.input_layout if self.input_layout in ("btc", "bft") else "btc"

            if input_layout == "btc":
                prepared_inputs = inputs
            elif input_layout == "bft":
                prepared_inputs = inputs.transpose(1, 2)
            else:
                raise ValueError("3D inputs only support 'btc', 'bft', or 'auto' layouts")
        elif inputs.dim() == 4:
            if input_layout == "auto":
                input_layout = "bcft"

            if input_layout != "bcft":
                raise ValueError("4D inputs only support 'bcft' or 'auto' layouts")

            batch_size, channels, features, time = inputs.size()
            prepared_inputs = (
                inputs.permute(0, 3, 1, 2)
                .contiguous()
                .view(batch_size, time, channels * features)
            )
        else:
            raise ValueError("inputs must be a 3D or 4D tensor")

        if prepared_inputs.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, but received {prepared_inputs.size(-1)}. "
                "For image-like inputs (B, C, F, T), set input_dim to C * F."
            )

        return prepared_inputs

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            input_layout: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` and `targets` pair for training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``. Image-like feature maps with x-axis = time
                and y-axis = features are also supported as ``(batch, channels, features, time)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            input_layout (str, optional): Overrides the default layout used to interpret ``inputs``.

        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        inputs = self._prepare_inputs(inputs, input_layout or self.input_layout)
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        outputs = self.fc(encoder_outputs)
        outputs = nn.functional.log_softmax(outputs, dim=-1)
        return outputs, encoder_output_lengths
