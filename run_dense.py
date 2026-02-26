import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from preprossesing import WaterQualityDataOrganization
from trainer import Trainer
from model_dense import EncoderDecoder, Decoder, Encoder
from utils import *

batch_size = 32

wq_org = WaterQualityDataOrganization(import_data())
wq_org.preprocessing()

X_train_dataset = WaterQualityDataset(wq_org.x_train.to_numpy(),wq_org.x_train.to_numpy())
X_val_dataset = WaterQualityDataset(wq_org.x_val.to_numpy(),wq_org.x_val.to_numpy())
X_test_dataset = WaterQualityDataset(wq_org.x_test.to_numpy(),wq_org.x_test.to_numpy())

encoder = Encoder()
decoder = Decoder()
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
plt.title('Training and Validation Loss Over Epochs model dense')
plt.xlabel('Epoch')
plt.ylabel('Loss (Mean Squared Error)')
plt.legend()
plt.grid(True)
plt.show()