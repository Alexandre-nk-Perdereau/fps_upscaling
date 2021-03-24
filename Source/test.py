from pathlib import Path
from os.path import join

import numpy
import torch

from config import models_directory, debug_path
from dataset import FrameUpscalingDataset
from models import YModel

import torchvision.transforms.transforms as transforms



def test(model_name, test_folders, debug_images=False):
    model = YModel()
    model.load_state_dict(torch.load(join(join(models_directory, model_name), model_name + '.pt')))
    model.eval()

    test_set = FrameUpscalingDataset(test_folders, 1)

    criterion = torch.nn.MSELoss()
    test_length = len(test_set)
    losses = numpy.zeros((test_length, 1))

    transforms_to_pilimage = transforms.ToPILImage(mode="RGB")

    for i, (previous_tensor, following_tensor, target_tensor) in enumerate(test_set):
        previous_tensor = previous_tensor.unsqueeze(0)
        following_tensor = following_tensor.unsqueeze(0)
        target_tensor = target_tensor.unsqueeze(0)

        interpolated_tensor = model(previous_tensor, following_tensor)
        loss = criterion(interpolated_tensor, target_tensor)
        losses[i] = loss.item()



        if debug_images:
            target_tensor = target_tensor.squeeze(0)
            interpolated_tensor = interpolated_tensor.squeeze(0)
            comparison_tensor = torch.cat((target_tensor, interpolated_tensor), dim=1)
            comparison_tensor * 255
            comparison_image = transforms_to_pilimage(comparison_tensor)

            save_name = join(debug_path, str(i) + ".png")
            comparison_image.save(save_name)


    print(numpy.mean(losses[i]))






if __name__ == '__main__':
    test('ymodel_MSE_epochnumber10', "240p_spring", debug_images=True)
