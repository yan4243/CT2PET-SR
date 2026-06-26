"""Placeholder paired PET/CT slice dataset.

Implement ``PairedImageFolders`` so that every item returns:
``pet, pet_affine, ct, ct_affine, key``.
"""

from torch.utils.data import Dataset


class PairedImageFolders(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Implement PairedImageFolders for the released PET/CT data format."
        )

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError(index)
