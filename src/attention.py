import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init="linear"):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)  # (bs,nc,128)

        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(w_init)
        )

    def forward(self, x):
        # b, n, size = x.size()
        # x = x.view(b * n, size)
        return self.linear_layer(x)


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):  # 4
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k  # 4
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        attn = torch.softmax(attn, dim=-1)  # (4*bs,nt,nc)

        # Dropout
        # attn = self.attn_dropout(attn)

        # Get Context Vector
        result = torch.bmm(attn, value)
        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden  # 128
        self.num_hidden_per_attn = num_hidden // h  # 128/4=32
        self.h = h  # 4

        self.key = Linear(num_hidden, num_hidden, bias=False)  # (bs,nc,128)
        self.value = Linear(num_hidden, num_hidden, bias=False)  # (bs,nc,128)
        self.query = Linear(num_hidden, num_hidden, bias=False)  # (bs,nc,128)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(
        self, key, value, query
    ):  # query: target_x(bs*nt,x_size), key: context_x(bs,nc,x_size), value: representation(bs,  ,12)
        batch_size = key.size(0)  # bs
        seq_k = key.size(1)  # nc
        seq_q = query.size(1)  # nt
        residual = query  # (bs,nt,x_size)

        # Make multihead
        # self.key((bs,nc,x_size))=(bs*nc,128)
        key = self.key(key).view(
            batch_size, seq_k, self.h, self.num_hidden_per_attn
        )  # (bs,nc,4,32)
        value = self.value(value).view(
            batch_size, seq_k, self.h, self.num_hidden_per_attn
        )  # (bs,nc,4,32)
        query = self.query(query).view(
            batch_size, seq_q, self.h, self.num_hidden_per_attn
        )  # (bs,nt,4,32)

        key = key.permute(2, 0, 1, 3).reshape(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).reshape(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).reshape(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.reshape(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).reshape(batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = torch.cat([residual, result], dim=-1)  # (bs,nt,128*2)

        # Final linear
        result = self.final_linear(result)  # (bs,nt,128*2)-->(bs,nt,128)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns
