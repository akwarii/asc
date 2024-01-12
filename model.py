import torch
import torch.nn as nn


class GBFExpansion(nn.Module):
    def __init__(self, gbf):
        """
        Initializes the GBFExpansion module.

        Args:
            gbf (dict): A dictionary containing the parameters for Gaussian basis function expansion.
                - dmin (float): The minimum value for the Gaussian basis function.
                - dmax (float): The maximum value for the Gaussian basis function.
                - steps (int): The number of steps for the Gaussian basis function.

        """
        super().__init__()

        self.min = gbf["dmin"]
        self.max = gbf["dmax"]
        self.steps = gbf["steps"]
        self.gamma = (self.max - self.min) / self.steps
        self.register_buffer(
            "filters",
            torch.linspace(self.min, self.max, self.steps)
        )

    def forward(self, data: torch.Tensor, bond=True) -> torch.Tensor:
        """
        Performs the forward pass of the GBFExpansion module.

        Args:
            data (torch.Tensor): The input data tensor.
            bond (bool, optional): Whether to include bond dimension. Defaults to True.

        Returns:
            torch.Tensor: The output tensor after applying the Gaussian basis function expansion.

        """
        dim = 2
        if bond:
            dim += 1
        return torch.exp(-((data.unsqueeze(dim) - self.filters) ** 2) / self.gamma**2)


class GraphAttentionV2Layer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.6,
        leaky_relu_negative_slope: float = 0.2,
        bias: bool = True,
        share_weights: bool = False
    ) -> None:
        super.__init__()
        
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights
        
        if self.is_concat:
            assert out_features % self.n_heads == 0, "out_features must be divisible by n_heads"
            self.n_hidden = out_features // self.n_heads
        else:
            self.n_hidden = out_features
        
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=bias)
        
        if self.share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=bias)
            
        self.attention = nn.Linear(self.n_hidden, 1, bias=bias)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
        n_nodes = h.shape[0]
        
        # Initial transformations for each each
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)
        
        # Compute attention scores
        g_l_repeat = g_l.repeat(n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)
        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)
        
        e = self.attention(self.activation(g_sum))
        e = e.squeeze(-1)
        
        # Mask attention scores
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        
        e = e.masked_fill(adj_mat == 0, float('-inf'))
        
        # Normalize attention scores
        a = self.softmax(e)
        a = self.dropout(a)
        
        # Compute final output for each head
        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)
        
        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)


class ConvAngle(nn.Module):
    """
    ConvAngle module applies convolutional operations on angle features and edge features.

    Args:
        edge_fea_len (int): The length of the edge features.
        angle_fea_len (int): The length of the angle features.
    """

    def __init__(
        self,
        edge_fea_len,
        angle_fea_len,
    ):
        super().__init__()

        self.angle_fea_len = angle_fea_len
        self.edge_fea_len = edge_fea_len

        angle_input_dim = self.angle_fea_len + 2 * self.edge_fea_len

        self.linear = nn.Linear(angle_input_dim, self.angle_fea_len)
        self.attention = nn.Sequential(
            nn.Linear(angle_input_dim, 1),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.normalized_activation = nn.Sequential(
            nn.LayerNorm(self.angle_fea_len),
            nn.Softplus(),
        )

    def forward(self, angle_fea, edge_fea, nbr_idx):
        """
        Forward pass of the ConvAngle module.

        Args:
            angle_fea (torch.Tensor): The angle features.
            edge_fea (torch.Tensor): The edge features.
            nbr_idx (torch.Tensor): The neighbor indices.

        Returns:
            torch.Tensor: The output of the ConvAngle module.
        """
        n, m, o, p = angle_fea.shape

        eij = edge_fea.unsqueeze(2).expand(n, m, m, p)
        eik = edge_fea[nbr_idx, :]
        eijk = torch.cat([eij, eik], dim=3)

        cat_fea = torch.cat([eijk, angle_fea], dim=3)

        output = self.normalized_activation(
            angle_fea + self.attention(cat_fea) * self.linear(cat_fea)
        )

        return output


class ConvEdge(nn.Module):
    def __init__(self, edge_fea_len, angle_fea_len):
        """
        Initializes the ConvEdge module.

        Args:
            edge_fea_len (int): The length of the edge feature.
            angle_fea_len (int): The length of the angle feature.
        """
        super().__init__()

        self.edge_fea_len = edge_fea_len
        self.angle_fea_len = angle_fea_len

        edge_input_dim = 2 * self.edge_fea_len + self.angle_fea_len

        self.linear = nn.Linear(edge_input_dim, self.edge_fea_len)
        self.attention = nn.Sequential(
            nn.Linear(edge_input_dim, 1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Softmax(dim=2),
        )
        self.normalized_activation = nn.Sequential(
            nn.LayerNorm(self.edge_fea_len),
            nn.Softplus(),
        )

    def forward(self, edge_fea, angle_fea, nbr_idx):
        """
        Forward pass of the ConvEdge module.

        Args:
            edge_fea (torch.Tensor): The input edge features.
            angle_fea (torch.Tensor): The input angle features.
            nbr_idx (torch.Tensor): The neighbor indices.

        Returns:
            torch.Tensor: The output of the ConvEdge module.
        """
        n, m = nbr_idx.shape

        eij = edge_fea.unsqueeze(2).expand(n, m, m, self.edge_fea_len)
        eik = edge_fea[nbr_idx, :]

        cat_fea = torch.cat([eij, eik, angle_fea], dim=3)

        output = self.normalized_activation(
            edge_fea + torch.sum(
                self.normalized_activation(
                    self.attention(cat_fea) *
                    self.linear(cat_fea)
                ),
                dim=2
            )
        )

        return output


class CEGANN(nn.Module):
    """
    Crystal Edge Graph Attention Neural Network (CEGANN) model.
    Implementation based on the paper: https://doi.org/10.1038/s41524-023-00975-z

    Args:
        gbf_bond (dict): Dictionary containing information about the Gaussian basis function expansion for bond features.
        gbf_angle (dict): Dictionary containing information about the Gaussian basis function expansion for angle features.
        n_conv_edge (int): Number of convolutional layers for edge features.
        edge_expansion_units (int): Number of units for expanding edge features.
        angle_expansion_units (int): Number of units for expanding angle features.
        n_classes (int): Number of output classes.
        pooling (bool): Whether to perform pooling on crystal features.
        embedding (bool): Whether to return embedded features.

    Methods:
        _message_passing(edge_fea, angle_fea, nbr_idx):
            Performs message passing on the edge features and angle features.
        forward(data):
            Forward pass of the CEGANN model.
        pool(atom_fea, crys_idx):
            Pooling function for crystal features.
    """

    def __init__(
        self,
        gbf_bond: dict,
        gbf_angle: dict,
        n_conv_edge: int = 3,
        edge_expansion_units: int = 128,
        angle_expansion_units: int = 128,
        n_classes: int = 2,
        pooling: bool = False,
        embedding: bool = False,
    ) -> None:
        super().__init__()

        self.pooling = pooling
        self.embedding = embedding

        edge_features_len = gbf_bond["steps"]
        angle_features_len = gbf_angle["steps"]

        self.gbf_edge = GBFExpansion(gbf_bond)
        self.linear_angle = nn.Linear(angle_features_len, angle_expansion_units)
        self.conv_edge = nn.ModuleList(
            [
                ConvEdge(edge_features_len, angle_features_len)
                for _ in range(n_conv_edge)
            ]
        )

        self.gbf_angle = GBFExpansion(gbf_angle)
        self.linear_edge = nn.Linear(edge_features_len, edge_expansion_units)
        self.conv_angle = nn.ModuleList(
            [
                ConvAngle(edge_features_len, angle_features_len)
                for _ in range(n_conv_edge - 1)
            ]
        )

        self.layer_norm = nn.LayerNorm(
            edge_expansion_units + angle_expansion_units)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout()

        self.output_layer = nn.Linear(
            edge_expansion_units + angle_expansion_units, n_classes)

    def _message_passing(
        self,
        edge_features: torch.Tensor,
        angle_features: torch.Tensor,
        neigh_idx: torch.Tensor
    ) -> tuple[: torch.Tensor, : torch.Tensor]:
        """
        Performs message passing on the edge features and angle features.

        Args:
            edge_features (torch.Tensor): The edge features.
            angle_features (torch.Tensor): The angle features.
            neigh_idx (torch.Tensor): The neighbor indices.

        Returns:
            torch.Tensor: The updated edge features.
            torch.Tensor: The updated angle features.
        """
        edge_features = self.conv_edge[0](edge_features, angle_features, neigh_idx)
        for conv_edge, conv_angle in zip(self.conv_edge[1:], self.conv_angle):
            angle_features = conv_angle(angle_features, edge_features, neigh_idx)
            edge_features = conv_edge(edge_features, angle_features, neigh_idx)

        return edge_features, angle_features

    def forward(self, data: tuple) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Forward pass of the CEGANN model.

        Args:
            data (tuple): Tuple containing bond features, angle features, neighbor indices, and crystal indices.

        Returns:
            torch.Tensor: Output of the model.
            torch.Tensor: Embedded features (if self.embedding is set to True).
        """
        edge_features, angle_features, neigh_idx, crystal_idx = data

        # Create features using Gaussian basis function expansion
        edge_features = self.gbf_edge(edge_features)
        angle_features = self.gbf_angle(angle_features, bond=False)

        # Perform message passing
        edge_features, angle_features = self._message_passing(
            edge_features, angle_features, neigh_idx)

        # Expand edge features and angle features
        edge_features = self.linear_edge(self.dropout(edge_features))
        angle_features = self.linear_angle(self.dropout(angle_features))

        # Sum over edge features and angle features
        edge_features = torch.sum(self.softplus(edge_features), dim=1)
        angle_features = torch.sum(
            self.softplus(
                torch.sum(self.softplus(angle_features), dim=2)
            ),
            dim=1,
        )

        # Concatenate edge features and angle features
        crystal_features = torch.cat([edge_features, angle_features], dim=1)

        if self.pooling:
            crystal_features = self.pool(crystal_features, crystal_idx)

        # Normalize and apply softplus activation
        crystal_features = self.softplus(self.layer_norm(crystal_features))

        if self.embedding:
            embedded_features = crystal_features

        # Apply dropout and linear layer
        output = self.output_layer(self.dropout(crystal_features))

        if self.embedding:
            return output, embedded_features
        else:
            return output

    def pool(self, atom_fea, crys_idx):
        """
        Pooling function for crystal features.

        Args:
            atom_fea (torch.Tensor): Atom-level features.
            crys_idx (list): List of indices mapping crystal features to atom features.

        Returns:
            torch.Tensor: Pooled crystal features.
        """
        summed_fea = [
            torch.mean(atom_fea[idx_map[0]: idx_map[1], :],
                       dim=0, keepdim=True)
            for idx_map in crys_idx
        ]
        return torch.cat(summed_fea, dim=0)
