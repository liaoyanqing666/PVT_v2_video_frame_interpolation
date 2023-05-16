import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision.transforms import ToTensor
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


class dataset(Dataset):
    def __init__(self, train=True, max_num=10000000):
        super(dataset, self).__init__()
        root = 'vimeo_triplet/sequences'
        self.root = root
        if train:
            data_dic_txt = 'vimeo_triplet/tri_trainlist.txt'
        else:
            data_dic_txt = 'vimeo_triplet/tri_testlist.txt'

        with open(data_dic_txt, 'r') as f:
            lines = f.readlines()
        self.lines = lines
        self.length = len(lines)
        self.length = min(self.length, max_num)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        line = self.lines[item]
        line = line[:-1]
        x1 = Image.open(os.path.join(self.root, line, 'im1.png'))
        x2 = Image.open(os.path.join(self.root, line, 'im2.png'))
        x3 = Image.open(os.path.join(self.root, line, 'im3.png'))
        return ToTensor()(x1), ToTensor()(x2), ToTensor()(x3)


if __name__ == "__main__":
    print('begin')
    # test whether all of the pictures are readable
    data = dataset(train=True)
    dataloader = DataLoader(data, batch_size=16)
    for i, (x1, x2, x3) in enumerate(dataloader):
        if i % 10 == 0:
            print(i)
    data = dataset(train=False)
    dataloader = DataLoader(data, batch_size=16)
    for i, (x1, x2, x3) in enumerate(dataloader):
        if i % 10 == 0:
            print(i)

    # for i, (x1, x2, x3) in enumerate(dataloader):
    #     print(x1)
    #     print(x1.min())
    #     print(x1.max())
    #     print(x1.shape)
    #     x1 = x1[0]
    #     img_array = x1.numpy().transpose((1, 2, 0))
    #     plt.imshow(img_array)
    #     plt.show()
