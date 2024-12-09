import torch
import torch.nn as nn
import torch.nn.functional as F

class IntermediateFusion(nn.Module):
    def __init__(self, model_raw, model_spec, num_classes, fusion_type="concat"):
        super(IntermediateFusion, self).__init__()

        self.device=None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model_raw = model_raw.to(self.device)
        self.model_spec = model_spec.to(self.device)
        self.num_classes = num_classes

        self.fusion_type = fusion_type

        self.hidden_dim = 64

        self.fc_TE = None
        self.fc_TFR = None
        
        # Learnable weights for attention fusion
        self.alpha = nn.Parameter(torch.rand(1) * 0.01)
        self.beta = nn.Parameter(torch.rand(1) * 0.01)

        if (fusion_type == "concat"):
            self.fc_out = nn.Linear(self.hidden_dim * 2, num_classes)
        else:
            self.fc_out = nn.Linear(self.hidden_dim, num_classes)

        # Freeze independent models so it doesn't update
        for param in self.model_raw.parameters():
            param.requires_grad = False
        for param in self.model_spec.parameters():
            param.requires_grad = False

        

    def forward(self, x_raw, x_spec):
        _, H_TE = self.model_raw(x_raw)
        _, H_TFR = self.model_spec(x_spec)

        H_TE = H_TE.to(self.device)
        H_TFR = H_TFR.to(self.device)

        # print("H_TE size: ", H_TE.size())
        # print("H_TFR size: ", H_TFR.size())
        # print("H_TE shape: ", H_TE.shape[1] * H_TE.shape[2])
        # print("H_TFR shape: ", H_TFR.shape[1] * H_TFR.shape[2] * H_TFR.shape[3])
        if (self.fc_TE is None):
            self.fc_TE = nn.Linear(H_TE.shape[1] * H_TE.shape[2], self.hidden_dim).to(self.device)
        if (self.fc_TFR is None):
            self.fc_TFR = nn.Linear(H_TFR.shape[1] * H_TFR.shape[2] * H_TFR.shape[3], self.hidden_dim).to(self.device)
        
        H_TE_flat = H_TE.view(H_TE.size(0), -1).to(self.device)
        H_TFR_flat = H_TFR.view(H_TFR.size(0), -1).to(self.device)
        # print("H_TE_flat: ", H_TE_flat.size())
        # print("H_TFR_flat: ", H_TFR_flat.size())

        H_TE = self.fc_TE(H_TE_flat)
        H_TFR = self.fc_TFR(H_TFR_flat)
        # print("H_TE: ", H_TE.size())
        # print("H_TFR: ", H_TFR.size())

        
        if self.fusion_type == "concat":
            # Concatenation fusion
            H = torch.cat((H_TE, H_TFR), dim=-1)
        
        elif self.fusion_type == "sum":
            # Summation fusion
            H = H_TE + H_TFR
        
        elif self.fusion_type == "attention":
            # Weighted attention fusion
            H = self.alpha * H_TFR + self.beta * H_TE
        
        else:
            raise ValueError("fusion_type must be 'concat', 'sum', or 'attention'")
        
        #print("H: ", H.size())
        
        H = self.fc_out(H)
        
        # Classification output
        return F.log_softmax(H, dim=-1)