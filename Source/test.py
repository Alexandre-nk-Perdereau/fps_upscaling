from os.path import join

import cv2
import numpy
import torch
from skimage.metrics import structural_similarity

from config import models_directory, debug_path, datafolder_path
from dataset import FrameUpscalingDataset
from models import YModel
from numpy import log10, swapaxes
from PIL import Image

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
    model.load_state_dict(
        torch.load(join(join(models_directory, model_name + "/temp"), "epoch" + str(model_epoch) + '.pt')))
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


def video_interpolation(video_name, output_name, model_name, model_epoch, device="cuda"):
    video_complete_path = join(join(datafolder_path, "Videos"), video_name)
    vid_cap = cv2.VideoCapture(video_complete_path)
    input_fps = vid_cap.get(cv2.CAP_PROP_FPS)
    definition = (int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    device = torch.device(device)
    model = YModel()
    model.load_state_dict(
        torch.load(join(join(models_directory, model_name + "/temp"), "epoch" + str(model_epoch) + '.pt')))
    model.to(device)
    model.eval()

    save_name = join(join(datafolder_path, "Videos"), output_name)
    # writer = cv2.VideoWriter(savename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), input_fps//divide_frame_factor , output_resolution)

    writer = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), input_fps*2, definition)
    to_tensor = transforms.ToTensor()
    tensor_list = []
    while vid_cap.isOpened():
        success, image = vid_cap.read()

        if success:
            image_umat = Image.fromarray(image)

            if len(tensor_list) < 2:
                tensor_list.append(to_tensor(image_umat).to(device))

                if len(tensor_list) ==2:
                    interpolated_tensor = model(tensor_list[0][None, ...], tensor_list[1][None, ...])
                    interpolated_array = (interpolated_tensor.squeeze(0).permute(1, 2, 0).cpu().detach() * 255).numpy().astype(numpy.uint8)
                    writer.write(interpolated_array)


            else:
                tensor_list[0] = tensor_list[1].detach().clone()
                tensor_list[1] = to_tensor(image_umat).to(device)

                interpolated_tensor = model(tensor_list[0][None, ...], tensor_list[1][None, ...])
                interpolated_array = (interpolated_tensor.squeeze(0).permute(1, 2, 0).cpu().detach() * 255).numpy().astype(numpy.uint8)
                writer.write(interpolated_array)

            writer.write(image)

        else:
            break
    vid_cap.release()
    writer.release()


if __name__ == '__main__':
    print("MSE")
    # test("ymodel_MSE", 19, "240p_spring", debug_images=True, measure_ssim=False)
    video_interpolation("spring240pframedividedby2.avi", "spring240pMSE.avi", "ymodel_MSE", 29)

    print("GAN")
    # test('ymodel_GAN_gamma0.5', 19, "240p_spring", debug_images=False, measure_ssim=True)
    video_interpolation("spring240pframedividedby2.avi", "spring240pGAN.avi", "ymodel_GAN_gamma0.5", 29)


