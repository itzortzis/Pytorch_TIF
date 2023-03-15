# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn

from skimage import exposure


"""## The Dataset class"""

# Dataset:
# --------
# The overwritten torch class to dynamically retrive
# x's and y's from the given dataset
#
# --> d: numpy array containing the imgs and the
#        corresponding annotation masks
#        Shape: (num_of_images, img_width, img_height,
#               1 img-channel + 1 mask-channels)
# <-- x, y: retrieve image-mask pair on-demand when
#           calling __getitem__ function

class Dataset(torch.utils.data.Dataset):
  def __init__(self, d):
    self.dtst = d

  def __len__(self):
    return len(self.dtst)

  def __getitem__(self, index):
    obj = self.dtst[index, :, :, :]
    e = np.where(obj[:, :, 0] < 0.3)
    t = exposure.equalize_hist(obj[:, :, 0])
    t[e] = 0
    # t1 = t - obj[:, :, 0]
    x = torch.from_numpy(t)
    y = torch.from_numpy(obj[:, :, 1])

    return x, y


"""## The DiceLoss layer"""

class DiceLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
    super(DiceLoss, self).__init__()

  def forward(self, inputs, targets, smooth=0):

    inputs = torch.squeeze(inputs, 1)
    thresh = nn.Threshold(0.8, 0)
    inputs = thresh(inputs)
    inputs = torch.ceil(inputs)

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

    return 1 - dice


"""## The framework core"""

import torch.nn.functional as F

class training():

  def __init__(self, comps):

    self.components = comps
    self.init_components()
    self.thresh_act = self.create_threshold_activation()
    self.model = self.model.cuda()


  # Init_components:
  # ----------------
  # Initialization of the basic components used for the
  # training and the validation procedures.
  def init_components(self):
    self.thresh    = self.components['threshold']
    self.epochs    = self.components['epochs']
    self.model     = self.components['model']
    self.opt       = self.components['opt']
    self.loss_fn   = self.components['loss_fn']
    self.train_ldr = self.components['train_ldr']
    self.valid_ldr = self.components['valid_ldr']


  def create_threshold_activation(self):

    return nn.Threshold(self.thresh, 0)


  # Main_training:
  # --------------
  # The supervisor of the training procedure.
  def main_training(self):

    for epoch in range(self.epochs + 1):

      tr_score, tr_loss = self.epoch_training()
      ts_score, ts_loss = self.epoch_validation()

      print("Epoch: ", epoch)
      print("Training: ", tr_score.item(), tr_loss.item())
      print("Validation: ", ts_score.item(), ts_loss.item())
      print()



  # Prepare_data:
  # -------------
  # This function reshapes, formats and prepare the x and y tensors
  # in a proper way to fit the model requirements.
  #
  # --> x: the inputs of the model
  # --> y: the targets of the model
  # <-- x: the modified inputs
  # <-- y: the modified targets
  def prepare_data(self, x, y):
    x = torch.unsqueeze(x, 1)

    x = x.to(torch.float32)
    y = y.to(torch.float32)

    x = x.cuda()
    y = y.cuda()

    return x, y


  # Epoch_training:
  # ---------------
  # This function is used for implementing the training
  # procedure during a single epoch.
  #
  # <-- epoch_score: performance score achieved during
  #                  the training
  # <-- epoch_loss: the loss function score achieved during
  #                 the training
  def epoch_training(self):
    print("Epoch training")
    self.model.train(True)
    current_score = 0.0
    current_loss = 0.0

    for x, y in self.train_ldr:
      x, y = self.prepare_data(x, y)

      self.opt.zero_grad()
      outputs = self.model(x)
      loss = self.loss_fn(outputs, y)
      loss.backward()
      self.opt.step()

      score, batches = self.batch_mean_score(outputs, y)
      current_score += score * batches
      current_loss  += loss * batches

    epoch_score = current_score / len(self.train_ldr.dataset)
    epoch_loss  = current_loss / len(self.train_ldr.dataset)

    return epoch_score, epoch_loss


  # Epoch_validation:
  # ---------------
  # This function is used for implementing the validation
  # procedure during a single epoch.
  #
  # <-- epoch_score: performance score achieved during
  #                  the validation
  # <-- epoch_loss: the loss function score achieved during
  #                 the validation
  def epoch_validation(self):

    print("Epoch validation")
    self.model.train(False)
    current_score = 0.0
    current_loss = 0.0

    for x, y in self.valid_ldr:
      x, y = self.prepare_data(x, y)

      with torch.no_grad():
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)

      score, batches = self.batch_mean_score(outputs, y)
      current_score += score * batches
      current_loss  += loss * batches

    epoch_score = current_score / len(self.valid_ldr.dataset)
    epoch_loss  = current_loss / len(self.valid_ldr.dataset)

    return epoch_score, epoch_loss

  def batch_mean_score(self, predb, yb):

    batch_size = len(predb)
    score_sum = 0
    idx = 0
    for i in range(len(predb)):
      d_score = self.dice_score(predb[i, 0, :, :], yb[i, :, :])
      score_sum += d_score
      idx += 1
    score_mean = score_sum / idx

    return score_mean, idx


  # Dice_score:
  # ------------
  # Given the predictions and the targets of the model, this function
  # calculates the dice score.
  #
  # --> preds: tensor containing the predictions of the model
  # --> targets: tensor containing the targets of the model
  # <-- dice: the calculated dice score
  def dice_score(self, preds, targets, smooth=0):

    preds = torch.squeeze(preds, 1)
    preds = self.thresh_act(preds)
    preds = torch.ceil(preds)

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2.*intersection + smooth)/(preds.sum() + targets.sum() + smooth)

    return dice
