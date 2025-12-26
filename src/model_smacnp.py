from attention import *
import torch
import torch.nn as nn


class LocationEncoder(nn.Module):
    """
    Mean-Location Encoder [w]
    """

    def __init__(self, num_hidden):
        super(LocationEncoder, self).__init__()
        self.input_projection = Linear(3, int(num_hidden))  # s+y
        self.context_projection = Linear(2, int(num_hidden))  # s
        self.target_projection = Linear(int(num_hidden), num_hidden)

    def forward(self, context_x, context_y, target_x):
        encoder_input = torch.cat(
            [context_x[..., 0:2], context_y], dim=-1
        )  # concat context location (x), context value (y)
        encoder_input = self.input_projection(
            encoder_input
        )  # (bs, nc, 3)--> (bs, nc, num_hidden)
        value = encoder_input  # (bs, nc, num_hidden)
        key = self.context_projection(context_x[..., 0:2])  # (bs, nc, num_hidden)
        query = self.context_projection(target_x[..., 0:2])  # (bs, nt, num_hidden)
        query = torch.unsqueeze(
            query, axis=2
        )  # (bs, nt, num_hidden) --> (bs, nt, 1, num_hidden)
        key = torch.unsqueeze(key, axis=1)  # (bs, nc, 2) --> (bs, 1, nc, num_hidden)
        weights = -torch.abs((key - query) * 0.5)  # [bs, nt, nc, num_hidden]
        weights = torch.sum(weights, axis=-1)  # [bs, nt, nc]
        weight = torch.softmax(weights, dim=-1)  # [bs, nt, nc]
        rep = torch.matmul(
            weight, value
        )  # (bs, nt, nc)*(bs, nc, hidden_number) = (bs, nt, hidden_number)
        rep = self.target_projection(rep)  # (bs, nt, hidden_number)

        return rep


class DeterministicEncoder(nn.Module):
    """
    Mean-Attribute Encoder [r]
    """

    def __init__(self, input_dim, num_hidden):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim, num_hidden)
        self.context_projection = Linear(input_dim - 1, num_hidden)
        self.target_projection = Linear(input_dim - 1, num_hidden)

    def forward(self, context_x, context_y, target_x):
        encoder_input = torch.cat(
            [context_x[..., 2:], context_y], dim=-1
        )  # concat context attribute (x), context value (y)
        encoder_input = self.input_projection(encoder_input)
        query = self.target_projection(target_x[..., 2:])
        keys = self.context_projection(context_x[..., 2:])

        for attention in self.cross_attentions:  # cross attention layer
            query, _ = attention(keys, encoder_input, query)  # (bs, nt, hidden_number)

        return query


class VarianceEncoder(nn.Module):
    """
    Variance Encoder [v]
    """

    def __init__(self, input_dim, num_hidden):
        super(VarianceEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(input_dim + 2, num_hidden)
        self.context_projection = Linear(input_dim + 2, num_hidden)
        self.target_projection = Linear(input_dim + 2, num_hidden)

    def forward(self, context_x, target_x):
        encoder_input = self.input_projection(
            context_x
        )  # (bs, nc, s+x) --> (bs, nc, num_hidden)
        query = self.target_projection(
            target_x
        )  # (bs, nt, s+x) --> (bs, nt, num_hidden)
        keys = self.context_projection(
            context_x
        )  # (bs, nc, s+x) --> (bs, nc, num_hidden)

        for attention in self.cross_attentions:  # cross attention layer
            query, _ = attention(keys, encoder_input, query)  # (bs, nt, hidden_number)

        return query


class Decoder(nn.Module):
    """
    Dencoder
    """

    def __init__(self, x_size, y_size, num_hidden):
        super(Decoder, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.num_hidden = num_hidden
        self.attribute = Linear(self.x_size, int(self.num_hidden / 4))
        self.location = Linear(2, int(self.num_hidden / 4))
        self.decoder1 = nn.Sequential(
            nn.Linear(2 * self.num_hidden + int(self.num_hidden / 2), num_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_hidden, 1),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.num_hidden + int(self.num_hidden / 2), num_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_hidden, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, target_x, r, w, v):
        """context_x : (batch_size, n_context, x_size)
        context_y : (batch_size, n_context, y_size)
        target_x : (batch_size, n_target, x_size)
        """

        bs, nt, x_size = target_x.shape  # (bs, nt, x_size)
        t_x = self.attribute(target_x[..., 2:])
        s_x = self.location(target_x[..., 0:2])
        z_tx = torch.cat([t_x, s_x], dim=-1)

        z1_tx = torch.cat(
            [torch.cat([w, z_tx], dim=-1), r], dim=-1
        )  # cat(x*, s*, r, w)
        z1_tx = z1_tx.view((bs * nt, 2 * self.num_hidden + int(self.num_hidden / 2)))
        decoder1 = self.decoder1(z1_tx)
        decoder1 = decoder1.view((bs, nt, 1))  # (bs, nt, 1)
        mu = decoder1[:, :, 0]

        z2_tx = torch.cat([z_tx, v], dim=-1)  # cat(x*, s*, v)
        z2_tx = z2_tx.view((bs * nt, self.num_hidden + int(self.num_hidden / 2)))
        decoder2 = self.decoder2(z2_tx)
        decoder2 = decoder2.view((bs, nt, 1))  # (bs, nt, 1)
        log_sigma = decoder2[:, :, 0]

        sigma = 0.1 + 0.9 * self.softplus(
            log_sigma
        )  # variance sigma=0.1+0.9*log(1+exp(log_sigma))

        return mu, sigma


class SpatialNeuralProcess(nn.Module):
    def __init__(self, x_size, y_size, num_hidden):
        super(SpatialNeuralProcess, self).__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.num_hidden = num_hidden
        self.determine = DeterministicEncoder(
            self.x_size + self.y_size, self.num_hidden
        )
        self.location = LocationEncoder(num_hidden)
        self.decoder = Decoder(self.x_size, self.y_size, self.num_hidden)
        self.variance = VarianceEncoder(self.x_size, self.num_hidden)

    def forward(self, context_x, context_y, target_x):
        """context_x : (batch_size, n_context, x_size)
        context_y : (batch_size, n_context, y_size)
        target_x : (batch_size, n_target, x_size)
        """
        r = self.determine(context_x, context_y, target_x)  # mean-attribute encoder
        w = self.location(context_x, context_y, target_x)  # mean-location encoder
        v = self.variance(context_x, target_x)  # variance encoder
        mu, sigma = self.decoder(target_x, r, w, v)  # decoder

        return mu, sigma


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.kl_div = nn.KLDivLoss()

    def forward(self, mu, sigma, target_y):
        """mu : (bs, n_target)
        sigma : (bs, n_target)
        target_y : (bs, n_target)
        """
        loss = 0.0
        bs = mu.shape[0]
        nt = mu.shape[1]
        if target_y is not None:
            for i in range(bs):
                dist1 = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=mu[i], covariance_matrix=torch.diag(sigma[i])
                )
                log_prob = dist1.log_prob(target_y[i])
                loss = -log_prob / nt  # torch.mean(log_prob)
        else:
            loss = None
        loss = loss / bs

        return loss
