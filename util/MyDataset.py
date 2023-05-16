from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import util


class MyDataset(Dataset):
    def __init__(self, label_file, img_path, net_img_size,
                 transform=None, label_bias=0, color_mode="RGB", enhance=0):
        file_list = open(label_file)
        images = []
        for i in file_list:
            i = i.strip()
            words = i.split()
            images.append((img_path + "\\" + words[0], int(words[1]) - label_bias))
        self.images = images
        self.transform = transform
        self.net_img_size = net_img_size
        self.color_mode = color_mode
        self.enhance = enhance

    def __getitem__(self, index):
        file, label = self.images[index]
        image = Image.open(file).convert(self.color_mode)
        image = util.resize_fit(image, self.net_img_size)
        if self.enhance > 0:
            enh_col = ImageEnhance.Contrast(image)
            contrast = self.enhance
            image = enh_col.enhance(contrast)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)