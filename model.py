import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, n_features, n_filters, kernel_size, latent_dim):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_filters * n_features,
            kernel_size=kernel_size,
            padding='same'  # In PyTorch, we need to calculate padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=n_filters * n_features,
            out_channels=n_filters * n_features,
            kernel_size=kernel_size,
            padding='same'
        )
        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)  # GlobalMaxPooling1D
        self.fc = nn.Linear(n_filters * n_features, latent_dim)

    def forward(self, x):
        # Input shape: (batch_size, n_window, n_features)
        # For Conv1d in PyTorch, we need (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)  # Change to (batch_size, n_features, n_window)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Global max pooling over the sequence dimension
        x = self.global_max_pool(x)  # Shape: (batch_size, n_filters*n_features, 1)
        x = x.squeeze(-1)  # Remove the last dimension: (batch_size, n_filters*n_features)

        x = self.fc(x)  # Shape: (batch_size, latent_dim)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_window, n_filters, n_features, kernel_size):
        super(Decoder, self).__init__()

        self.n_window_after_pool2 = n_window
        self.n_filters_after_conv2 = n_filters * n_features

        self.fc = nn.Linear(latent_dim, self.n_window_after_pool2 * self.n_filters_after_conv2)
        self.relu = nn.ReLU()

        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=self.n_filters_after_conv2,
            out_channels=n_filters * n_features,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # To maintain similar dimensions
        )

        self.conv = nn.Conv1d(
            in_channels=n_filters * n_features,
            out_channels=n_filters * n_features,
            kernel_size=kernel_size,
            padding='same'
        )

        # self.final_conv = nn.Conv1d(
        #     in_channels=n_filters * n_features,
        #     out_channels=n_features,
        #     kernel_size=1  # 1x1 convolution to reduce channels to n_features
        # )
        self.fc_2 = nn.Linear(n_filters * n_features, n_features)

    def forward(self, x):
        # Input shape: (batch_size, latent_dim)
        x = self.fc(x)
        x = self.relu(x)

        # Reshape to (batch_size, channels, sequence_length)
        x = x.view(-1, self.n_filters_after_conv2, self.n_window_after_pool2)

        x = self.relu(self.conv_transpose(x))
        x = self.relu(self.conv(x))

        # Final convolution to get the right number of features
        #x = self.final_conv(x)
        x = self.relu(self.fc_2(x))

        # Reshape back to original format: (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)

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
