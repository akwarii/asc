import torch
from torch import nn


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