from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def download(self, save_dir):
        raise NotImplementedError(f'There is no download codes in {self}')

    def __str__(self):
        return self.__name__
