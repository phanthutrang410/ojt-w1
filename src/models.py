# add count_params
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int=784, hidden_dim:int=256, num_classes: int=10, dropout: float=0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout), 
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
   
    def forward(self, x:torch.Tensor) -> torch.Tensor: 
        x_flattened = torch.flatten(x, start_dim=1) 
        output = self.net(x_flattened)
        return output 

def count_params(model: nn.Module) -> dict: 
    total_params = sum(p.numel() for p in model.parameters()) 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = model.__class__.__name__ 
    print(f"{model_name}: {total_params:,} params ({trainable_params:,} trainable)")
    return {
        "total": total_params,
        "trainable": trainable_params
    }