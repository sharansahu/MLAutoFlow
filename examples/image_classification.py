import torch
import pytorch_lightning as pl
import os

from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import datasets, transforms

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

class LightningMNISTClassifier(pl.LightningModule):

  def __init__(self, lr_rate):
    super(LightningMNISTClassifier, self).__init__()

    self.test_losses = []
    self.val_losses = []
    
    # mnist images are (1, 28, 28) (channels, width, height) 
    self.layer_1 = torch.nn.Linear(28 * 28, 128)
    self.layer_2 = torch.nn.Linear(128, 256)
    self.layer_3 = torch.nn.Linear(256, 10)
    self.lr_rate = lr_rate

  def forward(self, x):
      batch_size, channels, width, height = x.size()
      
      # (b, 1, 28, 28) -> (b, 1*28*28)
      x = x.view(batch_size, -1)

      # layer 1 (b, 1*28*28) -> (b, 128)
      x = self.layer_1(x)
      x = torch.relu(x)

      # layer 2 (b, 128) -> (b, 256)
      x = self.layer_2(x)
      x = torch.relu(x)

      # layer 3 (b, 256) -> (b, 10)
      x = self.layer_3(x)

      # probability distribution over labels
      x = torch.softmax(x, dim=1)

      return x

  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)

  def training_step(self, train_batch, batch_idx):
      x, y = train_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)

      logs = {'train_loss': loss}
      return {'loss': loss, 'log': logs}

  def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.val_losses.append(loss)  # Accumulate validation losses
      self.log('val_loss', loss, on_step=False, on_epoch=True)
      return loss

  def test_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.test_losses.append(loss)  # Accumulate test losses
      self.log('test_loss', loss, on_step=False, on_epoch=True)
      return loss

  def on_validation_epoch_end(self):
      if self.val_losses:
        avg_loss = torch.stack(self.val_losses).mean()
        self.log('val_loss', avg_loss)
        self.val_losses = []  # Reset for the next epoch
      else:
          self.log('val_loss', 0)  # Log a default value if no losses were collected

  def on_test_epoch_end(self):
      if self.test_losses:
        avg_loss = torch.stack(self.test_loss).mean()
        self.log('test_loss', avg_loss)
        self.test_losses = []  # Reset for the next epoch
      else:
        self.log('test_loss', 0)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
    lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95),
                    'name': 'expo_lr'}
    return [optimizer], [lr_scheduler]

def prepare_data():
  # transforms for images
  transform=transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))])
    
  # prepare transforms standard to MNIST
  mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
  mnist_train = [mnist_train[i] for i in range(2200)]
  
  mnist_train, mnist_val = random_split(mnist_train, [2000, 200])

  mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
  mnist_test = [mnist_test[i] for i in range(3000,4000)]

  return mnist_train, mnist_val, mnist_test

if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(), "trained-model")

    train, val, test = prepare_data()
    train_loader, val_loader, test_loader = DataLoader(train, batch_size=64), DataLoader(val, batch_size=64), DataLoader(test, batch_size=64)
    model = LightningMNISTClassifier(lr_rate=1e-3)

    # Learning Rate Logger
    lr_logger = LearningRateMonitor()
    # Set Early Stopping
    early_stopping = EarlyStopping('val_loss', mode='min', patience=5)
    # saves checkpoints to 'model_path' whenever 'val_loss' has a new min
    checkpoint_callback = ModelCheckpoint(dirpath=model_path, filename='mnist_{epoch}-{val_loss:.2f}',
                                        monitor='val_loss', mode='min', save_top_k=3)

    trainer = pl.Trainer(max_epochs=30, profiler="simple", callbacks=[lr_logger, early_stopping, checkpoint_callback], 
                        default_root_dir=model_path) #gpus=1

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)