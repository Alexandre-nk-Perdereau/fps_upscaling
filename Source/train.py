from math import exp
from pathlib import Path
from os.path import join

import torch

from config import models_directory
from dataset import FrameUpscalingDataset
from models import YModel, Discriminator


def train(folder_list, model_name="", epoch_number=10, batch_size=8, num_workers=4, epoch_start=0):
    """
    train a Ymodel using the MSE approach.
    :param folder_list: (list[string]) folders containing the training datasets
    :param model_name:  (string) save name of the model, if length=0 then savename = ymodel_MSE_epochnumberX.pt
    :param epoch_number:    (int) number of epoch
    :param batch_size:  (int)   batch size
    :param num_workers: (int) number of workers
    :param epoch_start (int) if different from 0 allows you to resume a training already started

    """
    # If I want to use batch>1 I need to write a collate_fn or to use only images with the same resolution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: " + str(device))
    if model_name == "":
        model_name = "ymodel_" + "MSE"
    save_directory = join(models_directory, model_name)
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    temp_directory = join(save_directory, "temp")
    Path(temp_directory).mkdir(parents=True, exist_ok=True)

    model = YModel()
    optimiser = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()))

    model.to(device)
    if epoch_start != 0:
        model_temp = join(temp_directory, "epoch" + str(epoch_start - 1) + ".pt")
        print("loading " + model_temp)
        model.load_state_dict(torch.load(model_temp))
    criterion = torch.nn.MSELoss().to(device)

    train_dataset = FrameUpscalingDataset(folder_list, 0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    losses = []
    batch_length = len(train_loader)

    for epoch in range(epoch_start, epoch_number):
        print("beginning of epoch " + str(epoch))
        loss_avg = 0
        model.train()

        for i, (previous_tensor, following_tensor, target_tensor) in enumerate(train_loader):
            previous_tensor = previous_tensor.to(device)
            following_tensor = following_tensor.to(device)
            target_tensor = target_tensor.to(device)

            interpolation_tensor = model(previous_tensor, following_tensor)

            loss = criterion(interpolation_tensor, target_tensor)
            optimiser.zero_grad()
            loss.backward()

            loss_avg += loss.item()
            optimiser.step()

        loss_avg /= batch_length
        print(loss_avg)
        losses.append(loss_avg)

        torch.save(model.state_dict(), join(temp_directory, "epoch" + str(epoch) + '.pt'))
    torch.save(model.state_dict(), join(save_directory, model_name + "_epochnumber" + str(epoch_number) + '.pt'))
    loss_file = open(join(save_directory, "loss_" + str(epoch_start) + "to" + str(epoch_start) + ".txt"), 'w')
    loss_file.write(str(losses))
    loss_file.close()


def train_gan(folder_list, model_name="", epoch_number=10, batch_size=8, num_workers=4, gamma=1,epoch_start=0):
    """
    train a Ymodel using the GAN approach.
    :param folder_list: (list[string]) folders containing the training datasets
    :param model_name:  (string) save name of the model, if length=0 then savename = ymodel_GAN_epochnumberX_gammaY.pt
    :param epoch_number:    (int) number of epoch
    :param batch_size:  (int)   batch size
    :param num_workers: (int) number of workers
    :param gamma:   (float) factor of the exponantial decay of the weight of the MSE
    :param epoch_start (int) if different from 0 allows you to resume a training already started
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "":
        model_name = "ymodel_" + "GAN" + "_gamma" + str(gamma)
    save_directory = join(models_directory, model_name)
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    temp_directory = join(save_directory, "temp")
    Path(temp_directory).mkdir(parents=True, exist_ok=True)

    model = YModel()
    discriminator = Discriminator()
    optimiser = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()))
    discriminator_optimiser = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()))

    model.to(device)
    discriminator.to(device)
    if epoch_start != 0:
        model_temp = join(temp_directory, "epoch" + str(epoch_start - 1) + ".pt")
        discriminator_temp = join(temp_directory, "discriminator_epoch" + str(epoch_start - 1) + ".pt")
        print("loading " + model_temp)
        model.load_state_dict(torch.load(model_temp))
        print("loading " + discriminator_temp)
        discriminator.load_state_dict(torch.load(discriminator_temp))

    criterion = torch.nn.MSELoss().to(device)
    adversarial_loss_criterion = torch.nn.BCELoss().to(device)

    train_dataset = FrameUpscalingDataset(folder_list, 0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    losses = []
    batch_length = len(train_loader)

    for epoch in range(epoch_start, epoch_number):
        print("beginning of epoch " + str(epoch))
        loss_avg = 0
        model.train()
        discriminator.train()

        for i, (previous_tensor, following_tensor, target_tensor) in enumerate(train_loader):
            previous_tensor = previous_tensor.to(device)
            following_tensor = following_tensor.to(device)
            target_tensor = target_tensor.to(device)

            interpolation_tensor = model(previous_tensor, following_tensor)

            interpolation_discrimination = discriminator(interpolation_tensor.detach()) # we don't want to backpropagate to the generator network (model)
            # it is a generated tensor, the discriminator should return 0
            actual_discrimination = discriminator(target_tensor)
            # it is a tensor coming from an actual image, the discriminator should return 1

            discriminator_loss_fake = adversarial_loss_criterion(interpolation_discrimination,
                                                                 torch.zeros_like(interpolation_discrimination))
            discriminator_loss_actual = adversarial_loss_criterion(actual_discrimination,
                                                                   torch.ones_like(actual_discrimination))

            discriminator_loss = (discriminator_loss_fake + discriminator_loss_actual) / 2
            discriminator_loss.backward(retain_graph=True)  # to change later
            discriminator_optimiser.step()

            adversarial_loss = adversarial_loss_criterion(interpolation_discrimination, torch.ones_like(
                actual_discrimination))  # minimising this criterion -> the interpolated image seems actual (if the
            # discriminator is good)

            alpha = exp(-gamma * epoch)
            loss = alpha * criterion(interpolation_tensor, target_tensor) + adversarial_loss.detach()   # detach because we don't want the discriminator to learn that this image is actual
            optimiser.zero_grad()
            loss.backward()

            loss_avg += loss.item()

            optimiser.step()

        loss_avg /= batch_length
        print(loss_avg)
        losses.append(loss_avg)

        torch.save(model.state_dict(), join(temp_directory, "epoch" + str(epoch) + '.pt'))
        torch.save(discriminator.state_dict(), join(temp_directory, "discriminator_epoch" + str(epoch) + '.pt'))

    torch.save(model.state_dict(), join(save_directory, model_name + "_epochnumber" + str(epoch_number) + '.pt'))
    torch.save(discriminator.state_dict(), join(save_directory, "discriminator_" + model_name + "_epochnumber" + str(epoch_number) + '.pt'))

    loss_file = open(join(save_directory, "loss_" + str(epoch_start) + str(epoch_start) + ".txt"), 'w')
    loss_file.write(str(losses))
    loss_file.close()


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    # train(['240p_sintel'], batch_size=16, epoch_number=20, epoch_start=15)
    train_gan(['240p_sintel'], batch_size=16, epoch_number=20, gamma=0.5, epoch_start=12)
