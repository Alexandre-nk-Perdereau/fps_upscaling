from torch.utils.data import Dataset
from config import datasets_path
from os.path import join
from PIL import Image
import torchvision.transforms.transforms as transforms
from config import debug_path

TRAINING_SET = 0
TEST_SET = 1


class FrameUpscalingDataset(Dataset):

    def __init__(self, folder_list, set):
        self.triplets = []

        if set == TRAINING_SET:
            self.transformations = transforms.Compose([transforms.ToTensor()])
        else:
            self.transformations = transforms.Compose([transforms.ToTensor()])

        for folder in folder_list:
            triplets_folder = open(join(datasets_path + folder, "triplets.txt"), 'r')
            triplet_strings = (triplets_folder.read()).split(';')
            for triplet_string in triplet_strings:
                triplet = triplet_string.split(',')
                for i in range(3):
                    triplet[i] = join(datasets_path + folder, triplet[i])
                self.triplets.append(triplet)

    def __getitem__(self, i):
        triplet = self.triplets[i]

        previous_image = Image.open(triplet[0])
        following_image = Image.open(triplet[2])
        target_image = Image.open(triplet[1])

        previous_tensor = self.transformations(previous_image)
        following_tensor = self.transformations(following_image)
        target_tensor = self.transformations(target_image)
        # I will need to change it when I will add random transforms to apply the same transforms to the triplet

        return previous_tensor, following_tensor, target_tensor

    def __len__(self):
        return len(self.triplets)


if __name__ == "__main__":
    test_ds = FrameUpscalingDataset(['test_ds'], TRAINING_SET)
    test_tensor, _, _ = test_ds[1000]
    test_image = transforms.ToPILImage()(test_tensor)
    test_image.save(join(debug_path, "test_image.jpg"))
