import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AxialTransformer(nn.Module):
    """
    Workflow:
    - 2D row/col attention over features (N, N) where N >> K
    - 2D row/col attention over graph estimates (T, k^2) -> (T, K)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.padding_idx = 0
        # embed edge types and project features into embed_dim
        self.embed_tokens = nn.Embedding(args.num_edge_types,
                                         args.embed_dim,
                                         padding_idx=self.padding_idx)
        self.embed_feats = nn.Linear(1, args.embed_dim)
        # embed node and edge index
        self.embed_nodes = LearnedPositionalEmbedding(
            self.args.num_vars,
            self.args.embed_dim,
            self.padding_idx,
        )
        self.embed_edges = EdgePositionalEmbedding(
            self.args.num_vars,
            self.args.embed_dim,
            self.padding_idx,
        )
        # embed time = subset index
        self.embed_time = LearnedPositionalEmbedding(
            500,  # max time
        #self.embed_time = SinusoidalPositionalEmbedding(
            self.args.embed_dim,
            self.padding_idx,
            permute=False
        )
        self.dropout_module = nn.Dropout(self.args.dropout)

        # Transformer encoder layers
        layers = []
        for _ in range(self.args.transformer_num_layers):
            # features are NxN so no varying dimension
            feat_layer = AxialTransformerLayer(
                embed_dim=self.args.embed_dim,
                ffn_embed_dim=self.args.ffn_embed_dim,
                n_heads=self.args.n_heads,
                dropout=self.args.dropout,
                max_tokens=args.max_length,
            )
            # graph estimates are TxK so require invariance to T -> scale_rows
            graph_layer = AxialTransformerLayer(
                embed_dim=self.args.embed_dim,
                ffn_embed_dim=self.args.ffn_embed_dim,
                n_heads=self.args.n_heads,
                dropout=self.args.dropout,
                max_tokens=args.max_length,
                scale_rows=True,
            )
            layers.append(AxialBlock(feat_layer, graph_layer))
        self.layers = nn.ModuleList(layers)

        self.emb_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)
        self.emb_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)
        self.feat_layer_norm_before = ESM1bLayerNorm(self.args.embed_dim)
        self.feat_layer_norm_after = ESM1bLayerNorm(self.args.embed_dim)

    def scatter_tokens(self, batch):
        """
            Scatter x from [B, T, #sampled] to [B, T, #unique]
                                    =k*k                 =K
        """
        x = batch["input"]  # [B, T, k*k]
        batch_size, num_trials, _ = x.size()
        num_edges = batch["index"].max() + 1  # [B, K]
        x_expand = torch.zeros(batch_size, num_trials, num_edges,
                device=x.device, dtype=x.dtype)
        index = batch["index"]  # [B, T, k*k]
        assert x.size() == index.size()
        # [B, T, k*k] -> [B, T, K]
        x_expand = x_expand.scatter_(dim=2, index=index, src=x)
        return x_expand

    def forward(self, batch, repr_layers=[]):
        """
            tokens = batch_size, num_trials, num_edges
            feats = batch_size, num_nodes, num_nodes
            mask = batch_size, num_trials
        """
        # expand inputs
        tokens = self.scatter_tokens(batch)  # [B, T, K]
        feats = batch["feats"]   # [B, N, N]
        nodes = batch["unique"]  # [B, K, 2]

        assert tokens.ndim == 3
        batch_size, num_trials, num_edges = tokens.size()
        padding_mask = tokens.eq(self.padding_idx)  # B, R, C = B, T, K
        padding_mask_feats = feats.eq(self.padding_idx)  # B, R, C = B, N, N
        if not padding_mask.any():
            padding_mask = None
        if not padding_mask_feats.any():
            padding_mask_feats = None

        # embed tokens, then (sparse) edge indices and time indices
        edge_embed = self.embed_edges(nodes)[:,None]  # [B, 1, K, d]
        time_embed = self.embed_time(tokens[:,:,0])[:,:,None]  # [B, T, 1, d]
        x = self.embed_tokens(tokens) + edge_embed + time_embed
        # embed features, then node indices along each of 2 dimensions
        row_ebd = self.embed_nodes(feats[:,0])[:,:,None]  # [B, N, 1, d]
        col_ebd = self.embed_nodes(feats[:,:,0])[:,None]  # [B, 1, N, d]
        feats = self.embed_feats(feats[...,None]) + row_ebd + col_ebd

        x = self.emb_layer_norm_before(x)
        x = self.dropout_module(x)
        feats = self.feat_layer_norm_before(feats)
        feats = self.dropout_module(feats)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        if padding_mask_feats is not None:
            feats = feats * (1 - padding_mask_feats.unsqueeze(-1).type_as(feats))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        # B x R x C x D -> R x C x B x D = T, K, B, d
        x = x.permute(1, 2, 0, 3)
        feats = feats.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x, feats = layer(
                x, feats, nodes,
                padding_mask=padding_mask,
                padding_mask_feats=padding_mask_feats,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.permute(2, 0, 1, 3)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D
        feats = self.feat_layer_norm_after(feats)
        feats = feats.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D
        if padding_mask_feats is not None:
            feats = feats * (1 - padding_mask_feats.unsqueeze(-1).type_as(feats))
        return feats

    @property
    def num_layers(self):
        return self.args.transformer_num_layers


class AxialBlock(nn.Module):
    """
        Single block
    """
    def __init__(self, feat_layer, graph_layer):
        super().__init__()
        self.feat_layer = feat_layer
        self.graph_layer = graph_layer
        self.pad = nn.Parameter(torch.randn(1))
        self.linear = nn.Linear(self.graph_layer.embed_dim * 2,
                                self.graph_layer.embed_dim)

    def forward(self, x, feats, nodes, padding_mask, padding_mask_feats):
        """
            x = num_trials, num_edges, batch_size, embed_dim
            feats = num_nodes, num_nodes, batch_size, embed_dim
            nodes = batch_size, 2, num_edges
        """
        # encode x = graph estimates
        x = self.graph_layer(x, padding_mask)

        # update feats with x
        num_nodes, _, batch_size, dim = feats.size()
        # num_nodes+1 to absorb padding 0s
        # initialize everything to self.pad (learned float)
        feats_update = torch.ones(num_nodes+1, num_nodes+1, batch_size, dim,
                                  device=feats.device) * self.pad
        # align features with selected nodes
        idx_0 = torch.arange(batch_size).view(1, 1, batch_size)
        idx_1 = nodes[:, :, 0:1]  # 1 index
        idx_2 = nodes[:, :, 1:2]
        feats_update[idx_1, idx_2, idx_0] = x.mean(0, keepdim=True)
        feats_update = feats_update[1:, 1:, :]  # trim padding
        # concatenate and project back to proper dimension
        feats = torch.cat([feats, feats_update], dim=-1)
        feats = self.linear(feats)

        # encode feats
        feats = self.feat_layer(feats, padding_mask_feats)

        # remove +1 from padding = 0-indexed version of nodes
        # ok because padding_mask removes relevant tokens
        nodes_0 = torch.where(nodes-1 < 0, torch.zeros_like(nodes), nodes-1)
        idx_1 = nodes_0[:, :, 0:1].permute(2,1,0)  # preserve 1 dimension
        idx_2 = nodes_0[:, :, 1:2].permute(2,1,0)
        # feats_subset = T, K, B, d
        feats_subset = feats[idx_1, idx_2, idx_0]
        # add relevant feats to x
        x = x + feats_subset
        return x, feats


class AxialTransformerLayer(nn.Module):
    """
        Single block
    """
    def __init__(self, *,
                 embed_dim: int, n_heads: int, dropout: float,
                 max_tokens: int, ffn_embed_dim: int,
                 scale_rows=False, scale_cols=False,
                 activation_dropout: float = 0.1,):
        super().__init__()
        """
            @param scale_rows  True to scale rows, for invariance to # cols
            @param scale_cols  True to scale cols, for invariance to # rows
        """
        # Save params
        self.embed_dim = embed_dim
        self.dropout = dropout
        # 2D attention over rows and columns
        # Shared arguments for all attention / FFN layers
        attn_kwargs = {
            "embed_dim": embed_dim,
            "num_heads": n_heads,
            "dropout": dropout,
            "max_tokens": max_tokens,
        }
        row_attn = SelfAttention2D(dim=0, use_scaling=scale_rows,
                                   **attn_kwargs)
        col_attn = SelfAttention2D(dim=1, use_scaling=scale_cols,
                                   **attn_kwargs)
        ffn = FeedForwardNetwork(embed_dim, ffn_embed_dim,
                                 activation_dropout=dropout)
        # Add residual wrapper
        self.row_attn = self.build_residual(row_attn)
        self.col_attn = self.build_residual(col_attn)
        self.ffn = self.build_residual(ffn)

    def forward(self, x, padding_mask):
        """
            x = batch_size, num_rows, num_cols, embed_dim
        """
        x = self.row_attn(x, padding_mask=padding_mask)
        x = self.col_attn(x, padding_mask=padding_mask)
        x = self.ffn(x)
        return x

    def build_residual(self, layer: nn.Module):
        """
            Wrap layer with LayerNorm and residual
        """
        return NormalizedResidualBlock(layer,
            self.embed_dim, self.dropout)


class SelfAttention2D(nn.Module):
    """
        Heavily modified from:
        https://github.com/facebookresearch/esm/blob/main/esm/model/msa_transformer.py

        Compute self-attention over rows of a 2D input.
        This module's correctness was tested in src/model/attn_test.py
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        max_tokens: int = 2 ** 16,
        dim=0,
        use_scaling=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens = max_tokens
        assert dim in [0, 1], f"dim {dim} not allowed; 2D inputs only, [0, 1]"
        self.dim = dim
        self.use_scaling = use_scaling
        self.attn_shape = "hnij"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, q):
        if not self.use_scaling:
            return 1.0
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def _batched_forward(
        self,
        x,
        padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_rows = max(1, self.max_tokens // num_cols)
        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self(
                x[start : start + max_rows],
                padding_mask=padding_mask[:, start : start + max_rows]
                if padding_mask is not None
                else None,
            )
            outputs.append(output)
        return outputs

    def compute_attention_weights(
        self,
        x,
        scaling: float,
        padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        q *= scaling
        if padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            q *= 1 - padding_mask.permute(1, 2, 0)[...,None,None].to(q)

        # r = row, i = index into col, n = batch_size, h = heads, d = head_dim
        attn_weights = torch.einsum(f"rinhd,rjnhd->{self.attn_shape}", q, k)

        if padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                padding_mask[:, 0][None,:,None],
                -10000,
            )

        return attn_weights

    def compute_attention_update(
        self,
        x,
        padding_mask=None,
    ):
        """
            x: [R, C, B, d]
            padding_mask: [B, R, C]
        """
        num_rows, num_cols, batch_size, embed_dim = x.size()
        # compute attention weights
        scaling = self.align_scaling(x)
        attn_weights = self.compute_attention_weights(
            x, scaling, padding_mask
        )
        attn_probs = attn_weights.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)
        # apply update
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        context = torch.einsum(f"{self.attn_shape},rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(
        self,
        x,
        padding_mask=None,
    ):
        """
            x: [R, C, B, d]
            padding_mask: [B, R, C]  DIFFERENT from x order ???
        """
        # permute
        if self.dim == 1:
            x = x.permute(1, 0, 2, 3)
            if padding_mask is not None:
                padding_mask = padding_mask.transpose(1, 2)
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if (num_rows * num_cols > self.max_tokens) and not torch.is_grad_enabled():
            output = self._batched_forward(x, padding_mask)
            output = torch.cat(output, dim=0)
        else:
            output = self.compute_attention_update(x, padding_mask)
        # permute back
        if self.dim == 1:
            output = output.permute(1, 0, 2, 3)
        return output


class EdgePositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embed_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        self.max_positions = num_embeddings_
        super().__init__(num_embeddings_, embed_dim, padding_idx)
        # project i,j -> single representation
        self.linear = nn.Linear(embed_dim*2, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor):
        """
            Input is expected to be of size [bsz, seqlen, 2].
            - Randomly shuffle for permutation invariance
            - Tokens should be 1-indexed
        """
        if tokens.max() > self.max_positions:
            raise ValueError(
                f"Sequence length {tokens.max()} above maximum "
                f" sequence length of {self.max_positions}"
            )
        mask = tokens.ne(self.padding_idx).int()
        # randomly shuffle nodes
        perm = torch.randperm(self.max_positions, device=tokens.device)
        tokens = perm[tokens]  # (B, L, 2)
        ebd = F.embedding(
            tokens,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # MLP to combine both forward and backward
        # (B, L, 2, dim) -> (B, L, dim)
        batch_size, length, *_ = ebd.size()
        x = self.linear(ebd.reshape(batch_size, length, -1))
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.linear2(x)
        return x


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embed_dim: int, padding_idx: int,
                 permute: bool = True):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embed_dim, padding_idx)
        self.max_positions = num_embeddings_
        self.permute = permute

    def forward(self, input: torch.Tensor):
        """
            Input is expected to be of size [bsz x seqlen].
            - Randomly shuffle for permutation invariance if self.permute
            Input should be 2D (including batch)
        """
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        assert len(input.size()) == 2, f"{input.size()} but only 2d allowed"
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        # randomly shuffle
        if self.permute:
            perm = torch.randperm(self.max_positions, device=input.device)
            positions = perm[positions]
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, learned=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.weights = None

    def forward(self, x):
        bsz, seq_len, *_ = x.shape
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = self.get_embedding(max_pos)
        self.weights = self.weights.type_as(self._float_tensor)

        positions = self.make_positions(x)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def make_positions(self, x):
        mask = x.ne(self.padding_idx)
        range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
        positions = range_buf.expand_as(x)
        return positions * mask.long() + self.padding_idx * (1 - mask.long())

    def get_embedding(self, num_embeddings):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if self.embed_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb


class NormalizedResidualBlock(nn.Module):
    """
        This class is unchanged
    """
    def __init__(
        self,
        layer: nn.Module,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.layer = layer
        self.dropout_module = nn.Dropout(
            dropout,
        )
        self.layer_norm = ESM1bLayerNorm(self.embed_dim)

    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = self.dropout_module(x)
        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x


class FeedForwardNetwork(nn.Module):
    """
        This class is unchanged
    """
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(
            activation_dropout,
        )
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x


class TopLayer(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.linear2 = nn.Linear(embed_dim, output_dim)

    def forward(self, features):
        x = self.linear(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.linear2(x)
        return x

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm

