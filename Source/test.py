from pathlib import Path
from os.path import join

import torch

from config import models_directory
from dataset import FrameUpscalingDataset
from models import YModel


def test(model_name, test_folders, debug_images=False):
    model = YModel()
    model.load_state_dict(torch.load(join(join(models_directory, model_name), model_name + '.pt')))
    model.eval()




if __name__ == '__main__':
    test('ymodel_MSE_epochnumber10', '240_spring')
