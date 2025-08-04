import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing

torch.set_printoptions(precision=5, threshold=2097152, linewidth=1000, sci_mode=False)


class ConstituentNet(nn.Module):
    """
    ConstituentNet Base Model
    """

    def __init__(
        self,
        in_dim: int = 16,
        embbed_dim: int = 128,
        num_heads: int = 2,
        num_classes: int = 5,
        num_transformers: int = 4,
        dropout: float = 0.0,
        is_debug: bool = False,
        num_particles: int = 30,
        activation: str = "ReLU",
        normalization: str = "Batch",
        **kwargs,
    ) -> None:
        super(ConstituentNet, self).__init__()
        self.is_debug = is_debug
        self.input_size = in_dim
        self.channel_in = in_dim
        self.embbed_dim = embbed_dim  # C
        self.num_transformers = num_transformers
        self.normalization = normalization

        self.embedding = nn.Linear(in_dim, embbed_dim)

        if self.normalization == "Batch":
            self.norm = nn.BatchNorm1d(embbed_dim)
        elif self.normalization == "Layer":
            self.norm = nn.LayerNorm(embbed_dim)
        else:
            # Dummy so type checking doesnt complain
            self.norm = nn.BatchNorm1d(embbed_dim)

        self.linear = nn.Linear(embbed_dim, num_classes)

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embbed_dim)
        )  # learned classification token, (1, 1, C)
        self.transformers = nn.ModuleList(
            [
                Transformer(
                    embbed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    is_debug=self.is_debug,
                    num_particles=num_particles,
                    activation=activation,
                    normalization=self.normalization,
                )
                for _ in range(num_transformers)
            ]
        )

        # Add slicer to fix Allo slice issue
        self.slicer = SliceFirstDim()

        torch.set_printoptions(
            precision=5, threshold=2097152, linewidth=1000, sci_mode=False
        )

    def debug_print(self, name: str, t):
        pass
        # if self.is_debug and not self.training:
        #     print(f"\n{name} -> {t.size()}")
        #     print(t)

    def forward(self, x):

        m_batch, seq_len, _ = x.size()
        self.debug_print("input", x)

        # Input layer
        out = self.embedding(x)  # (batch_size, num_particles, embbed_dim)
        self.debug_print("out (after embedding)", out)

        # Append class tokens to input
        cls_tokens = self.cls_token.repeat(m_batch, 1, 1)
        self.debug_print("cls_tokens", cls_tokens)
        out = torch.cat(
            (cls_tokens, out), dim=1
        )  # (batch_size, num_particles+1, embbed_dim)
        self.debug_print("out (after class tokens)", out)

        # Transformer layers
        for transformer in self.transformers:
            out = transformer(out)  # (batch_size, num_particles+1, embbed_dim)
            self.debug_print("out (after transformer layer)", out)

        # out = out[:, 0]  # (batch_size, embbed_dim)
        # self.debug_print("out (after out[:, 0])", out)

        out = self.slicer(out)

        if self.normalization == "Batch" or self.normalization == "Layer":
            out = self.norm(out)
            self.debug_print("out (after norm)", out)

        out = self.linear(out)
        self.debug_print("out (after linear)", out)

        # out = out.squeeze(1)
        # self.debug_print("out (after squeeze())", out)

        final_result = F.log_softmax(out, dim=-1)
        self.debug_print("final_result (softmax)", final_result)

        return final_result


class Transformer(nn.Module):
    """
    Transformer block with self-attention.
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: typing.Optional[int] = None,
        num_heads: int = 1,
        dropout: float = 0.0,
        is_debug: bool = False,
        num_particles: int = 30,
        activation: str = "ReLU",
        normalization: str = "Batch",
    ) -> None:
        super(Transformer, self).__init__()
        self.is_debug = is_debug
        self.in_dim = in_dim
        self.latent_dim = latent_dim if latent_dim is not None else in_dim
        self.channel_in = in_dim
        self.normalization = normalization

        self.self_attention = SelfAttention(
            in_dim,
            latent_dim=self.latent_dim,
            num_heads=num_heads,
            is_debug=self.is_debug,
            num_particles=num_particles,
            normalization=normalization,
        )

        if self.normalization == "Batch":
            self.norm_0 = nn.BatchNorm1d(in_dim)
        elif self.normalization == "Layer":
            self.norm_0 = nn.LayerNorm(in_dim)
        else:
            # Dummy so type checking doesnt complain
            self.norm_0 = nn.BatchNorm1d(in_dim)

        if activation == "ReLU":
            self.activ_0 = nn.ReLU()
        elif activation == "SiLU":
            self.activ_0 = nn.SiLU()
        else:
            raise TypeError("Unknown activation function")

        self.linear_0 = nn.Linear(in_dim, in_dim * 2, bias=False)

        if self.normalization == "Batch":
            self.norm_1 = nn.BatchNorm1d(in_dim * 2)
        elif self.normalization == "Layer":
            self.norm_1 = nn.LayerNorm(in_dim * 2)
        else:
            # Dummy so type checking doesnt complain
            self.norm_1 = nn.BatchNorm1d(in_dim * 2)

        if activation == "ReLU":
            self.activ_1 = nn.ReLU()
        elif activation == "SiLU":
            self.activ_1 = nn.SiLU()
        else:
            raise TypeError("Unknown activation function")

        self.linear_1 = nn.Linear(in_dim * 2, in_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        torch.set_printoptions(
            precision=5, threshold=2097152, linewidth=1000, sci_mode=False
        )

    def debug_print(self, name: str, t):
        pass
        # if self.is_debug and not self.training:
        #     print(f"\nT: {name} -> {t.size()}")
        #     print(t)

    def forward(self, x):
        """
        Args :
            x : input feature maps (batch_m, seq_len, C)
        Returns :
            out : self-attention  + linear-transformation (batch_m, seq_len, C)
            energy: self-attention energy (batch_m, seq_len, seq_len)
        """

        self.debug_print("input", x)

        x = self.self_attention(x)
        self.debug_print("x (after self-attention)", x)

        if self.normalization == "Batch":
            out0 = self.norm_0(x.transpose(1, 2)).transpose(1, 2)
        elif self.normalization == "Layer":
            out0 = self.norm_0(x)
        else:
            out0 = x
        self.debug_print("out0 (after norm_0)", out0)

        out1 = self.activ_0(out0)
        self.debug_print("out1 (after activ_0)", out1)

        out2 = self.linear_0(out1)
        self.debug_print("out2 (after linear_0)", out2)

        if self.normalization == "Batch":
            out3 = self.norm_1(out2.transpose(1, 2)).transpose(1, 2)
        elif self.normalization == "Layer":
            out3 = self.norm_1(out2)
        else:
            out3 = out2
        self.debug_print("out3 (after norm_1)", out3)

        out4 = self.activ_1(out3)
        self.debug_print("out4 (after activ_1)", out4)

        out5 = self.linear_1(out4)
        self.debug_print("out5 (after linear_1)", out5)

        out = x + out5
        self.debug_print("out (after x + out)", out)

        out = self.dropout(out)
        self.debug_print("out (after dropout)", out)

        return out


class SelfAttention(nn.Module):
    """
    Self-attention layer
    """

    def __init__(
        self,
        in_dim: int,
        latent_dim: typing.Optional[int] = None,
        num_heads: int = 1,
        is_debug: bool = False,
        num_particles: int = 30,
        normalization: str = "Batch",
    ) -> None:
        """
        Args :
            in_dim (int) : the channel dimension of queries tensor. (C)
            latent_dim (int) : the latent channel dimension (num_heads * head_dim, default equal to C)
            num_heads (int) : number of attention heads
        """

        super(SelfAttention, self).__init__()
        self.is_debug = is_debug
        self.in_dim = in_dim
        self.channel_in = in_dim  # C
        self.latent_dim = latent_dim if latent_dim is not None else in_dim
        self.head_dim = self.latent_dim // num_heads
        self.heads = num_heads
        self.normalization = normalization
        self.num_particles = num_particles

        if self.normalization == "Batch":
            self.norm = nn.BatchNorm1d(in_dim)
        elif self.normalization == "Layer":
            self.norm = nn.LayerNorm(in_dim)
        else:
            # Dummy so type checking doesnt complain
            self.norm = nn.BatchNorm1d(in_dim)

        self.q = nn.Linear(in_dim, in_dim, bias=False)
        self.k = nn.Linear(in_dim, in_dim, bias=False)
        self.v = nn.Linear(in_dim, in_dim, bias=False)
        self.out = nn.Linear(in_dim, in_dim)

        if self.normalization == "Batch":
            self.pre_exp_norm = nn.BatchNorm1d(
                (self.num_particles + 1) * (self.num_particles + 1)
            )
        elif self.normalization == "Layer":
            self.pre_exp_norm = nn.LayerNorm(
                (self.num_particles + 1) * (self.num_particles + 1)
            )
        else:
            # Dummy so type checking doesnt complain
            self.pre_exp_norm = nn.BatchNorm1d(
                (self.num_particles + 1) * (self.num_particles + 1)
            )

        assert (
            in_dim // num_heads
        ) * num_heads == in_dim, "Embedding dim needs to be divisible by num_heads"
        assert (
            self.head_dim * num_heads == self.latent_dim
        ), "Latent dim needs to be divisible by num_heads."

        torch.set_printoptions(
            precision=5, threshold=2097152, linewidth=1000, sci_mode=False
        )

    def debug_print(self, name: str, t):
        pass
        # if self.is_debug and not self.training:
        #     print(f"\nSA: {name} -> {t.size()}")
        #     print(t)

    def forward(self, x):
        """
        Args :
            x : input feature maps (batch_m, seq_len, C)
        Returns :
            out : self attention value + input feature (batch_m, seq_len, C)
        """

        m_batch, seq_len, C = x.size()
        self.debug_print("input", x)

        # Normalization across channels
        if self.normalization == "Batch":
            out = self.norm(x.transpose(1, 2)).transpose(1, 2)
        elif self.normalization == "Layer":
            out = self.norm(x)
        else:
            out = x
        self.debug_print("out (after norm)", out)

        # Queries, keys, and values
        queries = self.q(out).view(m_batch, seq_len, self.heads, -1)
        self.debug_print("queries", queries)
        keys = self.k(out).view(m_batch, seq_len, self.heads, -1)
        self.debug_print("keys", keys)
        values = self.v(out).view(m_batch, seq_len, self.heads, -1)
        self.debug_print("values", values)

        # Attention softmax(Q^T*K)
        # energy = torch.einsum("nqhc,nkhc->nhqk", [queries, keys])  # q: query len, k: key len
        Q_ = queries.permute(0, 2, 1, 3)  # nqhc -> nhqc
        K_ = keys.permute(0, 2, 3, 1)  # nkhc -> nhck
        energy = torch.matmul(Q_, K_)  # nhqk
        self.debug_print("energy", energy)

        energy_post = energy
        attention = F.softmax(energy_post / (self.head_dim ** (1 / 2)), dim=-1)

        # Output
        # out = torch.einsum("nhql,nlhc->nqhc", [attention, values])
        # attention: nhql
        V_ = values.permute(0, 2, 1, 3)  # nlhc -> nhlc
        out = torch.matmul(attention, V_)  # nhqc
        out = out.permute(0, 2, 1, 3)  # nhqc -> nqhc
        self.debug_print("out (after einsum)", out)

        out = out.reshape(m_batch, seq_len, -1)
        self.debug_print("out (after reshape)", out)

        out = self.out(out)
        self.debug_print("out (after out())", out)

        final_sum = out + x
        self.debug_print("final_sum (before returning)", final_sum)

        return final_sum


class SliceFirstDim(nn.Module):
    def forward(self, inp):
        # Only for out[:, 0, :]
        return inp[:, 0, :]
