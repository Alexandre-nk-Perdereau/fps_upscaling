import torch
from torch import nn

from dataset import FrameUpscalingDataset, TRAINING_SET


class YModel(nn.Module):
    def __init__(self):
        super(YModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2, padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, padding_mode='zeros')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, padding_mode='zeros')
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, padding_mode='zeros')
        self.maxpool = nn.MaxPool2d((2, 2), padding=0)
        self.relu = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=2, padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=2, kernel_size=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=2, kernel_size=2, padding=0)
        # the article use 3x3 kernels instead
        self.finalconv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, previous_input, following_input):

        left_branch = self.conv1(previous_input)  # 1=0.5^0
        left_branch = self.relu(left_branch)
        residual_pow0 = left_branch
        left_branch = self.maxpool(left_branch)  # 0.5^1
        residual_pow1 = left_branch
        left_branch = self.conv2(left_branch)  # 0.5^1
        left_branch = self.relu(left_branch)
        left_branch = self.conv3(left_branch)  # 0.5^1
        left_branch = self.relu(left_branch)
        left_branch = self.maxpool(left_branch)  # 0.5^2
        residual_pow2 = left_branch
        left_branch = self.conv4(left_branch)  # 0.5^2
        left_branch = self.relu(left_branch)
        left_branch = self.conv5(left_branch)  # 0.5^2
        left_branch = self.relu(left_branch)
        left_branch = self.maxpool(left_branch)  # 0.5^3
        # residual_pow3 = left_branch

        right_branch = self.conv1(following_input)
        right_branch = self.relu(right_branch)
        residual_pow0 = residual_pow0 + right_branch
        right_branch = self.maxpool(right_branch)
        residual_pow1 = residual_pow1 + right_branch
        right_branch = self.conv2(right_branch)
        right_branch = self.relu(right_branch)
        right_branch = self.conv3(right_branch)
        right_branch = self.relu(right_branch)
        right_branch = self.maxpool(right_branch)
        residual_pow2 = residual_pow2 + right_branch
        right_branch = self.conv4(right_branch)
        right_branch = self.relu(right_branch)
        right_branch = self.conv5(right_branch)
        right_branch = self.relu(right_branch)
        right_branch = self.maxpool(right_branch)

        y = left_branch + right_branch

        y = self.deconv1(y)
        y = self.relu(y)
        y = y + residual_pow2
        y = self.deconv2(y)
        y = self.relu(y)
        y = y + residual_pow1
        y = self.deconv3(y)
        y = self.relu(y)
        y = y + residual_pow0
        y = self.finalconv(y)
        y = self.tanh(y)
        y = self.relu(y)

        return y


if __name__ == '__main__':
    test_ds = FrameUpscalingDataset(['test_ds'], TRAINING_SET)
    # test_previous_tensor, test_following_tensor, _ = test_ds[1000]
    # print(test_previous_tensor.shape)
    # test_previous_tensor = torch.rand(3, 582, 1280)
    # test_following_tensor = torch.rand(3, 582, 1280)

    # test_model = YModel()
    # test_model.eval()
    # test_output = test_model(test_previous_tensor[None, ...], test_following_tensor[None, ...])
    # print(test_output.shape)
    # test_net = nn.ConvTranspose2d(in_channels=3, out_channels=3,stride=3, kernel_size=3, padding=0)
    # test_result = test_net(test_previous_tensor[None, ...])
