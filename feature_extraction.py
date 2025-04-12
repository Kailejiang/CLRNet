import math
from typing import Callable, Dict, Union, Optional
import torch
from torch import nn
import torch.nn.functional as F

# Define shifted_softplus if not already available
def shifted_softplus(x):
    return F.softplus(x) - math.log(2.0)

def scatter_add(x, idx_i, dim_size, dim = 0):
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y

class Cfconv(nn.Module):
    def __init__(
        self,
        n_atom_feature: int,
        n_interactions: int,
        n_distance_feature: int,
        cutoff: float,
        n_filters: int = None,
        activation: Union[Callable, nn.Module] = shifted_softplus,
        nuclear_embedding: Optional[nn.Module] = None,
        start: float = 0.0,
    ):
        super().__init__()
        self.n_atom_feature = n_atom_feature
        self.n_filters = n_filters or self.n_atom_feature
        self.cutoff = cutoff
        self.n_distance_feature = n_distance_feature

        offset = torch.linspace(start, cutoff, n_distance_feature)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        self.register_buffer("widths", widths)
        self.register_buffer("offsets", offset)

        # Initialize embeddings
        if nuclear_embedding is None:
            nuclear_embedding = nn.Embedding(400, n_atom_basis)
        self.embedding = nuclear_embedding

        # Initialize interaction block layers
        self.interactions = nn.ModuleList([
            nn.ModuleDict({
                "in2f": nn.Linear(n_atom_feature, self.n_filters, bias=False),  # Replace Dense
                "filter_network": nn.Sequential(
                    nn.Linear(n_distance_feature, self.n_filters),  # Replace Dense
                    activation(),
                    nn.Linear(self.n_filters, self.n_filters),  # Replace Dense
                ),
                "f2out": nn.Sequential(
                    nn.Linear(self.n_filters, n_atom_feature),  # Replace Dense
                    activation(),
                    nn.Linear(n_atom_feature, n_atom_feature)  # Replace Dense
                )
            })
            for _ in range(n_interactions)
        ])

    def forward(self, inputs: Dict[str, torch.Tensor]):

        # Distance
        R = inputs["R"]  # Replace properties.R
        idx_i = inputs["idx_i"].long()  # Ensure indices are long
        idx_j = inputs["idx_j"].long()

        Rij = R[idx_j] - R[idx_i]
        inputs["Rij"] = Rij  # Replace properties.Rij

        # Get tensors from input dictionary
        atomic_numbers = inputs["Z"]  # Replace properties.Z
        r_ij = inputs["Rij"]
        d_ij = torch.norm(r_ij, dim=1)

        # Compute pair features
        coeff = -0.5 / torch.pow(self.widths, 2)
        diff = d_ij[..., None] - self.offsets
        f_ij = torch.exp(coeff * torch.pow(diff, 2))

        rcut_ij = 0.5 * (torch.cos(d_ij * math.pi / self.cutoff) + 1.0)
        rcut_ij *= (d_ij < self.cutoff).float()  # Remove contributions beyond cutoff

        # Compute initial embeddings
        x = self.embedding(atomic_numbers)

        # Compute interaction blocks and update atomic embeddings
        for interaction in self.interactions:
            # Interaction logic (merged from SchNetInteraction)
            in2f = interaction["in2f"]
            filter_network = interaction["filter_network"]
            f2out = interaction["f2out"]

            x_f = in2f(x)
            Wij = filter_network(f_ij)
            Wij = Wij * rcut_ij[:, None]

            # Continuous-filter convolution
            x_j = x_f[idx_j]
            x_ij = x_j * Wij
            x_f = scatter_add(x_ij, idx_i, dim_size=x.shape[0]) 

            # Update atomic embeddings
            x = x + f2out(x_f)

        # Collect results
        inputs["scalar_representation"] = x

        return inputs
