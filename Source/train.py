from pathlib import Path
from os.path import join

import torch

from config import datafolder_path, models_directory
from dataset import FrameUpscalingDataset
from models import YModel


def train(folder_list, model_name="", epoch_number=10, approach="MSE", batch_size=8, num_workers=4):
    # If I want to use batch>1 I need to write a collate_fn or to use only images with the same resolution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "":
        model_name = "ymodel_" + approach + "_epochnumber" + str(epoch_number)
    save_directory = join(models_directory, model_name)
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    temp_directory = join(save_directory, "temp")
    Path(temp_directory).mkdir(parents=True, exist_ok=True)

    model = YModel()
    optimiser = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()))

    model.to(device)
    criterion = torch.nn.MSELoss().to(device)

    train_dataset = FrameUpscalingDataset(folder_list, 0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    losses = []
    batch_length = len(train_loader)

    for epoch in range(epoch_number):
        print("beginning of epoch "+str(epoch))
        loss_avg = 0
        model.train()

        for i, (previous_tensor, following_tensor, target_tensor) in enumerate(train_loader):
            previous_tensor = previous_tensor.to(device)
            following_tensor = following_tensor.to(device)
            target_tensor = target_tensor.to(device)

            interpolation_tensor = model(previous_tensor, following_tensor)
            if interpolation_tensor.shape != target_tensor.shape:
                print("PROBLEM")
                print(interpolation_tensor.shape)
                print(target_tensor.shape)
                print(previous_tensor.shape)
                print(following_tensor.shape)

            loss = criterion(interpolation_tensor, target_tensor)
            optimiser.zero_grad()
            loss.backward()

            loss_avg += loss.item()

            optimiser.step()
        loss_avg /= i + 1
        print(loss_avg)
        losses.append(loss_avg)

        torch.save(model, join(temp_directory, "epoch" + str(epoch)))
    torch.save(model, join(temp_directory, model_name))
    loss_file = open(join(save_directory, "loss.txt"), 'w')
    loss_file.write(str(losses))
    loss_file.close()


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    train(['240p_sintel'], batch_size=16)
