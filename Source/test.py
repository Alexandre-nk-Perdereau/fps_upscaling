from os.path import join

import numpy
import torch
from skimage.metrics import structural_similarity

from config import models_directory, debug_path
from dataset import FrameUpscalingDataset
from models import YModel
from numpy import log10, swapaxes

import torchvision.transforms.transforms as transforms


def test(model_name, model_epoch, test_folders, debug_images=False, device="cuda", measure_ssim=False):
    """
    calculate the mse on a test set.
    :param model_name:  (string) name under which the model is saved.
    :param model_epoch (int) load the model trained in epoch model_epoch
    :param test_folders:    (list[string]) folders that contains the test data
    :param debug_images:    (boolean)   if True, save the comparison image actual image - interpolated image
    :param device:  (string) cuda or cpu
    """
    device = torch.device(device)
    model = YModel()
    model.load_state_dict(torch.load(join(join(models_directory, model_name + "/temp"), "epoch" + str(model_epoch) + '.pt')))
    model.to(device)
    model.eval()

    test_set = FrameUpscalingDataset(test_folders, 1)

    criterion = torch.nn.MSELoss().to(device)
    test_length = len(test_set)
    losses = numpy.zeros((test_length, 1))
    if measure_ssim:
        ssim_list = numpy.zeros((test_length, 1))

    transforms_to_pilimage = transforms.ToPILImage(mode="RGB")

    for i, (previous_tensor, following_tensor, target_tensor) in enumerate(test_set):
        previous_tensor = previous_tensor.to(device)
        following_tensor = following_tensor.to(device)
        target_tensor = target_tensor.to(device)

        previous_tensor = previous_tensor.unsqueeze(0)
        following_tensor = following_tensor.unsqueeze(0)
        target_tensor = target_tensor.unsqueeze(0)

        interpolated_tensor = model(previous_tensor, following_tensor)

        loss = criterion(interpolated_tensor, target_tensor)  # loss = MSE
        losses[i] = loss.item()

        if debug_images:
            target_tensor = target_tensor.squeeze(0)
            interpolated_tensor = interpolated_tensor.squeeze(0)
            comparison_tensor = torch.cat((target_tensor, interpolated_tensor), dim=1)
            comparison_tensor * 255
            comparison_image = transforms_to_pilimage(comparison_tensor.cpu())
            save_name = join(debug_path, str(i) + ".png")
            comparison_image.save(save_name)

        if measure_ssim:
            ssim_list[i] = structural_similarity(swapaxes(target_tensor.squeeze(0).cpu().detach().numpy(), 0, 2),
                                                 swapaxes(interpolated_tensor.squeeze(0).cpu().detach().numpy(), 0, 2),
                                                 multichannel=True)

    print("mse: " + str(numpy.mean(losses)))
    print('psnr: ' + str(numpy.mean(10 * log10(1 / losses))))
    if measure_ssim:
        print("ssim :" + str(numpy.mean(ssim_list)))


if __name__ == '__main__':
    print("MSE")
    test("ymodel_MSE", 19, "240p_spring", debug_images=True, measure_ssim=False)
    # print("GAN")
    # test('ymodel_GAN_epochnumber10_gamma0.5', 19, "240p_spring", debug_images=False, measure_ssim=True)
