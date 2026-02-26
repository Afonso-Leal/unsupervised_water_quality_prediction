import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from preprossesing import WaterQualityDataOrganization
from trainer import Trainer
from model_cnn import EncoderDecoder, Decoder, Encoder
from utils import *

n_window = 30
n_features = 9

kernel_size = 5
latent_dim = 15
n_filters = 20
batch_size = 32

#UNCOMMENT TO RESET PICKLED DATA FOR CNN
# wq_org = WaterQualityDataOrganization(import_data())
# wq_org.preprocessing()
# wq_org.shape_to_3d(n_window)
#
# X_train_dataset = WaterQualityDataset(wq_org.x_train_3d,wq_org.x_train_3d)
# X_val_dataset = WaterQualityDataset(wq_org.x_val_3d,wq_org.x_val_3d)
# X_test_dataset = WaterQualityDataset(wq_org.x_test_3d,wq_org.x_test_3d)
#
# pickle_variable(X_train_dataset,"./picles/train_dataset_3d.pickle")
# pickle_variable(X_val_dataset,"./picles/val_dataset_3d.pickle")
# pickle_variable(X_test_dataset,"./picles/test_dataset_3d.pickle")

X_train_dataset = unpickle_variable("./picles/train_dataset_3d.pickle")
X_val_dataset = unpickle_variable("./picles/val_dataset_3d.pickle")
X_test_dataset = unpickle_variable("./picles/test_dataset_3d.pickle")

#Create the models
encoder = Encoder(n_features, n_filters, kernel_size, latent_dim)
decoder = Decoder(latent_dim, n_window, n_filters, n_features, kernel_size)
model = EncoderDecoder(encoder, decoder)
print(model)

criterion = nn.MSELoss()  # Change based on your task
optimizer = torch.optim.Adam(model.parameters(), lr= 0.0001)

prob_train = Trainer(
    model = model,
    train_dataset = X_train_dataset,
    val_dataset = X_val_dataset,
    batch_size = batch_size,
    criterion = criterion,
    optimizer = optimizer
)

train_losses, val_losse = prob_train.train(num_epochs = 10)

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losse, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs model cnn')
plt.xlabel('Epoch')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.grid(True)
plt.show()


