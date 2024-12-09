import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusion(nn.Module):
    def __init__(self, model_raw, model_spec, num_classes):
        super(LateFusion, self).__init__()
        self.model_raw = model_raw
        self.model_spec = model_spec
        self.num_classes = num_classes

        # Learnable attention weight vectors
        self.WR = nn.Parameter(torch.randn(1, num_classes) * 0.01)
        self.WS = nn.Parameter(torch.randn(1, num_classes) * 0.01)

        self.softmax = nn.Softmax(dim=1)

        # Freeze independent models so it doesn't update
        for param in self.model_raw.parameters():
            param.requires_grad = False
        for param in self.model_spec.parameters():
            param.requires_grad = False

    def forward(self, x_raw, x_spec):
        # Obtain classification outputs from both models
        OTE, _ = self.model_raw(x_raw)
        OTFR, _ = self.model_spec(x_spec)
        
        # Apply attention weights with Hadamard product (element-wise multiplication)
        weighted_OTE = self.WR * OTE  # WR ⊙ OTE
        weighted_OTFR = self.WS * OTFR  # WS ⊙ OTFR
        
        # Compute final classification scores by summing the weighted outputs
        OLF = weighted_OTE + weighted_OTFR

        return OLF

    def get_raw_weights(self):
        # Get raw weights
        return self.WR, self.WS
    
    def get_attention_weights(self):
        # Get interpretable weights
        WR_softmax = self.softmax(self.WR)
        WS_softmax = self.softmax(self.WS)
        return WR_softmax, WS_softmax