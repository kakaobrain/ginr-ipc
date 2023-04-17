import torch
import torch.nn as nn


class LatentMapping(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.num_patches = config.n_patches
        self.type = config.type
        self.input_dim = input_dim

        if self.config.use_pe:
            self.pos_emb_latent = nn.Parameter(torch.zeros(1, self.num_patches, self.config.latent_dim))
            self.pos_emb_latent.data.normal_(mean=0.0, std=1.0)
        else:
            self.pos_emb_latent = None

        if self.type == "linear":
            self.latent_mapping = nn.Linear(input_dim, self.config.latent_dim)
        else:
            raise NotImplementedError

    def forward(self, xs):
        latent = self.latent_mapping(xs)
        latent = latent + self.pos_emb_latent if self.config.use_pe else latent
        return latent
