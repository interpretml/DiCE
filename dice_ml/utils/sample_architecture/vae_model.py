import torch
import torch.utils.data
from torch import nn


class CF_VAE(nn.Module):

    def __init__(self, d, encoded_size):

        super(CF_VAE, self).__init__()

        self.encoded_size = encoded_size
        self.data_size = len(d.ohe_encoded_feature_names)
        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, \
            self.cont_minx, self.cont_maxx, self.cont_precisions = d.get_data_params_for_gradient_dice()

        flattened_indexes = [item for sublist in self.encoded_categorical_feature_indexes for item in sublist]
        self.encoded_continuous_feature_indexes = [ix for ix in range(len(self.minx[0])) if ix not in flattened_indexes]
        self.encoded_start_cat = len(self.encoded_continuous_feature_indexes)

        # Plus 1 to the input encoding size and data size to incorporate the target class label
        self.encoder_mean = nn.Sequential(
            nn.Linear(self.data_size+1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size)
            )

        self.encoder_var = nn.Sequential(
            nn.Linear(self.data_size+1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
            nn.Sigmoid()
            )

        # Plus 1 to the input encoding size and data size to incorporate the target class label
        self.decoder_mean = nn.Sequential(
            nn.Linear(self.encoded_size+1, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, self.data_size),
            nn.Sigmoid()
            )

    def encoder(self, x):
        mean = self.encoder_mean(x)
        logvar = 0.5 + self.encoder_var(x)
        return mean, logvar

    def decoder(self, z):
        mean = self.decoder_mean(z)
        return mean

    def sample_latent_code(self, mean, logvar):
        eps = torch.randn_like(logvar)
        return mean + torch.sqrt(logvar)*eps

    def normal_likelihood(self, x, mean, logvar, raxis=1):
        return torch.sum(-0.5 * ((x - mean)*(1./logvar)*(x-mean) + torch.log(logvar)), axis=1)

    def forward(self, x, c):
        c = c.view(c.shape[0], 1)
        c = torch.tensor(c).float()
        res = {}
        mc_samples = 50
        em, ev = self.encoder(torch.cat((x, c), 1))
        res['em'] = em
        res['ev'] = ev
        res['z'] = []
        res['x_pred'] = []
        res['mc_samples'] = mc_samples
        for _ in range(mc_samples):
            z = self.sample_latent_code(em, ev)
            x_pred = self.decoder(torch.cat((z, c), 1))
            res['z'].append(z)
            res['x_pred'].append(x_pred)

        return res

    def compute_elbo(self, x, c, pred_model):
        c = torch.tensor(c).float()
        c = c.view(c.shape[0], 1)
        em, ev = self.encoder(torch.cat((x, c), 1))
        kl_divergence = 0.5*torch.mean(em**2 + ev - torch.log(ev) - 1, axis=1)

        z = self.sample_latent_code(em, ev)
        dm = self.decoder(torch.cat((z, c), 1))
        log_px_z = torch.tensor(0.0)

        x_pred = dm
        return torch.mean(log_px_z), torch.mean(kl_divergence), x, x_pred, torch.argmax(pred_model(x_pred), dim=1)


class AutoEncoder(nn.Module):

    def __init__(self, d, encoded_size):

        super(AutoEncoder, self).__init__()

        self.encoded_size = encoded_size
        self.data_size = len(d.encoded_feature_names)
        self.encoded_categorical_feature_indexes = d.get_data_params()[2]

        self.encoded_continuous_feature_indexes = []
        for i in range(self.data_size):
            valid = 1
            for v in self.encoded_categorical_feature_indexes:
                if i in v:
                    valid = 0
            if valid:
                self.encoded_continuous_feature_indexes.append(i)

        self.encoded_start_cat = len(self.encoded_continuous_feature_indexes)

        self.encoder_mean = nn.Sequential(
            nn.Linear(self.data_size, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size)
        )

        self.encoder_var = nn.Sequential(
            nn.Linear(self.data_size, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
            nn.Sigmoid()
         )

        self.decoder_mean = nn.Sequential(
            nn.Linear(self.encoded_size, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, self.data_size),
            nn.Sigmoid()
            )

    def encoder(self, x):
        mean = self.encoder_mean(x)
        logvar = 0.05 + self.encoder_var(x)
        return mean, logvar

    def decoder(self, z):
        mean = self.decoder_mean(z)
        return mean

    def sample_latent_code(self, mean, logvar):
        eps = torch.randn_like(logvar)
        return mean + torch.sqrt(logvar)*eps

    def normal_likelihood(self, x, mean, logvar, raxis=1):
        return torch.sum(-0.5 * ((x - mean)*(1./logvar)*(x-mean) + torch.log(logvar)), axis=1)

    def forward(self, x):
        res = {}
        mc_samples = 50
        em, ev = self.encoder(x)
        res['em'] = em
        res['ev'] = ev
        res['z'] = []
        res['x_pred'] = []
        res['mc_samples'] = mc_samples
        for _ in range(mc_samples):
            z = self.sample_latent_code(em, ev)
            x_pred = self.decoder(z)
            res['z'].append(z)
            res['x_pred'].append(x_pred)

        return res
