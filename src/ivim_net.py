import torch
import torch.nn as nn

class IVIM_NET(nn.Module):
    """
    IVIM-NET baseline implementation.
    Reference: Barbieri et al., MRM 2020.
    
    A single Multi-Layer Perceptron (MLP) for IVIM parameter estimation.
    """
    def __init__(self, b_values=[0, 5, 50, 100, 200, 500, 800, 1000], device='cpu'):
        super().__init__()
        self.b_values = torch.tensor(b_values, dtype=torch.float32, device=device)
        self.device = device
        num_b = len(b_values)
        
        self.net = nn.Sequential(
            nn.Linear(num_b, num_b),
            nn.ELU(),
            nn.Linear(num_b, num_b),
            nn.ELU(),
            nn.Linear(num_b, num_b),
            nn.ELU(),
            nn.Linear(num_b, 4), # S0, f, Dt, D*
        )

    def forward(self, x):
        """
        Args:
            x: shape (N, num_b) or (num_b,) - input signal (normalized or unnormalized)
        Returns:
            S_pred: shape (N, num_b) - reconstructed signal
            f, Dt, Dstar: shape (N,) - estimated parameters
        """
        out = self.net(x)
        
        # Scaling to physiological bounds
        S0 = torch.sigmoid(out[..., 0]) * 2.0
        f = torch.sigmoid(out[..., 1]) * 0.395 + 0.005
        Dt = torch.sigmoid(out[..., 2]) * 0.0024 + 0.0001
        Dstar = torch.sigmoid(out[..., 3]) * 0.063 + 0.002
        
        # Physics decoder: bi-exponential equation
        # S(b) = S0 * [(1-f)*exp(-b*Dt) + f*exp(-b*D*)]
        b = self.b_values.view(1, -1) if x.dim() > 1 else self.b_values
        
        S_pred = S0.unsqueeze(-1) * (
            (1 - f.unsqueeze(-1)) * torch.exp(-b * Dt.unsqueeze(-1)) + 
            f.unsqueeze(-1) * torch.exp(-b * Dstar.unsqueeze(-1))
        )
        
        return S_pred, f, Dt, Dstar

    def loss_function(self, pred_signal, true_signal):
        return nn.MSELoss()(pred_signal, true_signal)


class IVIM_NEToptim(nn.Module):
    """
    IVIM-NEToptim baseline implementation.
    Reference: Kaandorp et al., MRM 2021.
    
    A parallel sub-network architecture for IVIM parameter estimation.
    """
    def __init__(self, b_values=[0, 5, 50, 100, 200, 500, 800, 1000], device='cpu'):
        super().__init__()
        self.b_values = torch.tensor(b_values, dtype=torch.float32, device=device)
        self.device = device
        num_b = len(b_values)
        
        def build_branch():
            return nn.Sequential(
                nn.Linear(num_b, num_b),
                nn.BatchNorm1d(num_b),
                nn.ELU(),
                nn.Dropout(0.10),
                nn.Linear(num_b, num_b),
                nn.BatchNorm1d(num_b),
                nn.ELU(),
                nn.Dropout(0.10),
                nn.Linear(num_b, 1)
            )
            
        self.branch_S0 = build_branch()
        self.branch_f = build_branch()
        self.branch_Dt = build_branch()
        self.branch_Dstar = build_branch()

    def forward(self, x):
        """
        Args:
            x: shape (N, num_b) - input signal (normalized or unnormalized)
        Returns:
            S_pred: shape (N, num_b) - reconstructed signal
            f, Dt, Dstar: shape (N,) - estimated parameters
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        out_S0 = self.branch_S0(x).squeeze(-1)
        out_f = self.branch_f(x).squeeze(-1)
        out_Dt = self.branch_Dt(x).squeeze(-1)
        out_Dstar = self.branch_Dstar(x).squeeze(-1)
        
        # Scaling to physiological bounds
        S0 = torch.sigmoid(out_S0) * 2.0
        f = torch.sigmoid(out_f) * 0.395 + 0.005
        Dt = torch.sigmoid(out_Dt) * 0.0024 + 0.0001
        Dstar = torch.sigmoid(out_Dstar) * 0.063 + 0.002
        
        b = self.b_values.view(1, -1)
        
        S_pred = S0.unsqueeze(-1) * (
            (1 - f.unsqueeze(-1)) * torch.exp(-b * Dt.unsqueeze(-1)) + 
            f.unsqueeze(-1) * torch.exp(-b * Dstar.unsqueeze(-1))
        )
        
        return S_pred, f, Dt, Dstar

    def loss_function(self, pred_signal, true_signal):
        return nn.MSELoss()(pred_signal, true_signal)
