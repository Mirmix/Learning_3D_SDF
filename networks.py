import torch


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, higher_dimension=4, ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.higher_dimension = higher_dimension
        self.Softplus = torch.nn.Softplus(beta=10)
        if (higher_dimension):  # for fairness
            self.fc0 = torch.nn.Linear(self.input_size, higher_dimension * input_size * 2)
            self.fc1 = torch.nn.Linear(self.higher_dimension * input_size * 2, self.hidden_size)
        else:
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc7 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc8 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        if (self.higher_dimension):  # for fairness
            hidden = self.fc1(self.Softplus(self.fc0(x)))
        else:
            hidden = self.fc1(x)
        softplus = self.Softplus(hidden)
        softplus2 = self.Softplus(self.fc2(softplus))
        softplus3 = self.Softplus(self.fc3(softplus2))
        output = self.fc8(softplus3)
        return output


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=3, omega0=0.1):
        """
        I took some ideas from PyTorch3D implementation for implementing this
        Personally i would like to think omega0 as scaling factor.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class FeedforwardNeRF(torch.nn.Module):
    # For 3D SDF regression, with out any Radiance Fields of course
    def __init__(self, n_harmonic_functions, input_size, hidden_size):
        super().__init__()
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)
        self.input_size = n_harmonic_functions * input_size * 2
        self.hidden_size = hidden_size
        self.Softplus = torch.nn.Softplus(beta=10)
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc7 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc8 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = self.harmonic_embedding(x)
        hidden = self.fc1(x)
        softplus = self.Softplus(hidden)
        softplus2 = self.Softplus(self.fc2(softplus))
        softplus3 = self.Softplus(self.fc3(softplus2))
        output = self.fc8(softplus3)

        return output


class FeedforwardSIREN(torch.nn.Module):
    def __init__(self, n_harmonic_functions, input_size, hidden_size):
        super().__init__()
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)
        self.input_size = n_harmonic_functions * input_size * 2
        self.hidden_size = hidden_size
        self.Softplus = torch.nn.Softplus(beta=10)
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc5 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc6 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc7 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc8 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = self.harmonic_embedding(x)
        hidden = self.fc1(x)

        softplus_siren = torch.sin(hidden)
        softplus2_siren = torch.sin(self.fc2(softplus_siren))
        softplus3_siren = torch.sin(self.fc3(softplus2_siren))
        output = self.fc8(softplus3_siren)

        return output
