import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.fc0 = nn.Sequential(
            nn.Linear(9,180),
            nn.Linear(180, 180),
            nn.Dropout(0.2),
            nn.Linear(180, 180),
            nn.Dropout(0.2),
            nn.Linear(180, 180),
        )
        self.fc = nn.Linear(180, 90)

    def forward(self, x):

        x = nn.functional.relu(self.fc0(x))

        x = self.fc(x)  # Shape: (batch_size, latent_dim)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc0 = nn.Sequential(
            nn.Linear(90, 180),
            nn.Dropout(0.2),
            nn.Linear(180, 180),
            nn.Linear(180, 180),
            nn.Linear(180, 180),
            nn.Dropout(0.2),
            nn.Linear(180, 180)
        )

        self.fc_2 = nn.Linear(180, 9)

    def forward(self, x):
        # Input shape: (batch_size, latent_dim)
        x = nn.functional.relu(self.fc0(x))
        x = self.fc_2(x)

        return x


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
