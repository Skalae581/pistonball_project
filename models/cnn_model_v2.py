# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 14:48:29 2025

@author: TAKO
"""

# models/cnn_model_v2.py
from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),  # -> 32x20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> 64x7x7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(64*7*7, 512), nn.ReLU())
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.backbone(x)
        x = self.fc(x)
        logits = self.policy_fn(x)
        self._value_out = self.value_fn(x)
        return logits, state

    def value_function(self):
        return self._value_out.flatten()
