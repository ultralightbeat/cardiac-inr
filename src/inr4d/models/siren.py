"""Role 2: SIREN (Sitzmann et al., arXiv:2006.09661).

Implement yourselves from the paper: sine layer, first-layer vs hidden-layer
initialisation, omega_0. Explain WHY the initialisation is what it is.
"""


class SIREN:  # TODO(Role 2): make this an nn.Module
    def __init__(self, in_features=4, hidden_features=256, hidden_layers=3,
                 out_features=1, first_omega_0=30.0, hidden_omega_0=30.0):
        raise NotImplementedError

    def forward(self, coords):
        """coords [B, in_features] -> [B, out_features]."""
        raise NotImplementedError
