"""Placeholder case-level PET/CT dataset used by inference."""

from torch.utils.data import Dataset


class PairedImageFoldersByCase(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Implement PairedImageFoldersByCase for the released PET/CT data format."
        )

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)


def get_case_dataloader(*args, **kwargs):
    raise NotImplementedError(
        "Implement get_case_dataloader after defining the public dataset format."
    )
