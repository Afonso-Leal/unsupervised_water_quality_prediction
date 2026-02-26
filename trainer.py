import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            train_dataset,
            val_dataset=None,
            batch_size: int = 32,
            criterion: nn.Module = nn.MSELoss,
            optimizer: torch.optim.Optimizer = None,
            val_split: float = 0.2,  # If val_dataset not provided, split from train
            device: str = None,
            log_dir: str = './logs'
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model.to(self.device)

        # Setup datasets
        if val_dataset is None:
            # Split training data into train/val
            train_size = int((1 - val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )
        else:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        # Setup dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )

        # Setup loss function and optimizer
        self.criterion = criterion
        self.optimizer = optimizer

        # For logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def train_epoch(self):
        """Train for one epoch and return average loss"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            #data = data.double()
            #target = target.double()
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass

            output = self.model(data.float())
            loss = self.criterion(output, target.float())
            self.optimizer.zero_grad()
            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Log to tensorboard
            self.writer.add_scalar('Loss/train_batch', loss.item(), self.global_step)
            self.global_step += 1

        return total_loss / num_batches

    def validate(self):
        """Validate the model and return average loss"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating')
            for data, target in pbar:
                #data = data.double()
                #target = target.double()
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data.float())
                loss = self.criterion(output, target)

                # Track loss
                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches

    def train(self, num_epochs: int):
        """Main training loop returning losses for both train and val"""
        train_losses = []
        val_losses = []
        self.model = self.model.float()

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')

            # Train
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            val_losses.append(val_loss)

            # Log epoch losses
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Log to tensorboard
            self.writer.add_scalars('Loss/epoch', {
                'train': train_loss,
                'val': val_loss
            }, epoch)

        self.writer.close()
        return train_losses, val_losses