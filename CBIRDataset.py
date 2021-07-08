from myImports import *
from utils import Utils

utils = Utils()


class CBIRDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame

        self.transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        row = self.dataFrame.iloc[key]
        img = Image.open(row['image'])
        new_img = utils.pre_processing(img)
        image = self.transformations(new_img)
        return image

    def __len__(self):
        return len(self.dataFrame.index)