"""
Spatial transformation utilities for aligning CT to PET coordinate system.
Lightweight implementation without complex registration - pure coordinate transformation and resampling.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pydicom
from typing import Tuple, Optional, Union


def align_ct_to_pet(
    ct_image: Union[np.ndarray, torch.Tensor],
    pet_image: Union[np.ndarray, torch.Tensor],
    ct_dicom_path: str,
    pet_dicom_path: str,
    output_size: Optional[Tuple[int, int]] = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Transform a CT image into the PET coordinate system and resolution.
    Uses DICOM metadata for spatial transformation and resampling without
    registration optimization.
    
    Args:
        ct_image: CT image with shape (H, W) or (C, H, W)
        pet_image: PET image with shape (H, W) or (C, H, W), used to define the target space
        ct_dicom_path: Path to the CT DICOM file used to read coordinate information
        pet_dicom_path: Path to the PET DICOM file used to read coordinate information
        output_size: Output image size; defaults to the PET image size
        device: Compute device
    
    Returns:
        Aligned CT image in the PET coordinate system and resolution
    """
    # Convert to torch tensors
    if isinstance(ct_image, np.ndarray):
        ct_tensor = torch.from_numpy(ct_image).float()
    else:
        ct_tensor = ct_image.float()
    
    if isinstance(pet_image, np.ndarray):
        pet_tensor = torch.from_numpy(pet_image).float()
    else:
        pet_tensor = pet_image.float()
    
    # Ensure 4D tensor shape (B, C, H, W)
    if ct_tensor.ndim == 2:
        ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)
    elif ct_tensor.ndim == 3:
        ct_tensor = ct_tensor.unsqueeze(0)
    
    if pet_tensor.ndim == 2:
        pet_tensor = pet_tensor.unsqueeze(0).unsqueeze(0)
    elif pet_tensor.ndim == 3:
        pet_tensor = pet_tensor.unsqueeze(0)
    
    # Move tensors to the requested device
    ct_tensor = ct_tensor.to(device)
    pet_tensor = pet_tensor.to(device)
    
    # Read DICOM metadata
    ct_affine = get_affine_matrix(ct_dicom_path)
    pet_affine = get_affine_matrix(pet_dicom_path)
    
    # Determine the output size
    if output_size is None:
        output_size = (pet_tensor.shape[2], pet_tensor.shape[3])
    
    # Generate a sampling grid in the PET coordinate system
    grid = create_sampling_grid(
        output_size=output_size,
        ct_shape=ct_tensor.shape[-2:],
        ct_affine=ct_affine,
        pet_affine=pet_affine,
        device=device
    )
    
    # Resample with grid_sample
    aligned_ct = F.grid_sample(
        ct_tensor,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    
    return aligned_ct.squeeze()


def get_affine_matrix(dicom_path: str) -> np.ndarray:
    """
    Read an affine transformation matrix from a DICOM file.
    The matrix maps image coordinates to physical coordinates.
    
    Args:
        dicom_path: Path to the DICOM file
    
    Returns:
        A 4x4 affine transformation matrix
    """
    ds = pydicom.dcmread(dicom_path)
    
    # Read the required DICOM attributes
    image_position = np.array(ds.ImagePositionPatient, dtype=float)  # Physical position of the first voxel
    image_orientation = np.array(ds.ImageOrientationPatient, dtype=float)  # Image direction cosines
    pixel_spacing = np.array(ds.PixelSpacing, dtype=float)  # Pixel spacing [row, column]
    
    # Build direction vectors
    row_cosine = image_orientation[:3]  # Row direction cosine
    col_cosine = image_orientation[3:]  # Column direction cosine
    
    # Scale by the physical pixel spacing
    row_vec = row_cosine * pixel_spacing[0]
    col_vec = col_cosine * pixel_spacing[1]
    
    # Build the 4x4 affine matrix
    affine = np.eye(4)
    affine[:3, 0] = col_vec  # Column direction (x-axis)
    affine[:3, 1] = row_vec  # Row direction (y-axis)
    affine[:3, 3] = image_position  # Translation (origin)
    
    return affine


def create_sampling_grid(
    output_size: Tuple[int, int],
    ct_shape: Tuple[int, int],
    ct_affine: np.ndarray,
    pet_affine: np.ndarray,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create a sampling grid that maps PET coordinates to CT coordinates.
    
    Args:
        output_size: Output image size (H, W)
        ct_shape: CT image size (H, W)
        ct_affine: CT affine transformation matrix
        pet_affine: PET affine transformation matrix
        device: Compute device
    
    Returns:
        Sampling grid with shape (1, H, W, 2) and values in [-1, 1]
    """
    H, W = output_size
    
    # Create a coordinate grid in PET image space
    y_coords = torch.linspace(0, H - 1, H, device=device)
    x_coords = torch.linspace(0, W - 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Convert to homogeneous coordinates (H, W, 4)
    ones = torch.ones_like(grid_x)
    zeros = torch.zeros_like(grid_x)
    coords_pet = torch.stack([grid_x, grid_y, zeros, ones], dim=-1)  # (H, W, 4)
    
    # Convert affine matrices to torch tensors
    pet_affine_tensor = torch.from_numpy(pet_affine).float().to(device)
    ct_affine_tensor = torch.from_numpy(ct_affine).float().to(device)
    ct_affine_inv = torch.linalg.inv(ct_affine_tensor)
    
    # Coordinate transformation: PET image -> physical space -> CT image
    coords_pet_flat = coords_pet.reshape(-1, 4, 1)  # (H*W, 4, 1)
    coords_physical = torch.matmul(pet_affine_tensor, coords_pet_flat)  # (H*W, 4, 1)
    coords_ct = torch.matmul(ct_affine_inv, coords_physical)  # (H*W, 4, 1)
    coords_ct = coords_ct.squeeze(-1).reshape(H, W, 4)  # (H, W, 4)
    
    # Extract x and y coordinates
    ct_x = coords_ct[..., 0]
    ct_y = coords_ct[..., 1]
    
    # Normalize to [-1, 1]
    ct_h, ct_w = ct_shape
    grid_x_norm = 2.0 * ct_x / (ct_w - 1) - 1.0
    grid_y_norm = 2.0 * ct_y / (ct_h - 1) - 1.0
    
    # Assemble the format required by grid_sample: (1, H, W, 2)
    grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1).unsqueeze(0)
    
    return grid


def align_ct_to_pet_simple(
    ct_image: Union[np.ndarray, torch.Tensor],
    pet_shape: Tuple[int, int],
    ct_pixel_spacing: Tuple[float, float],
    pet_pixel_spacing: Tuple[float, float],
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Simplified alignment based only on pixel spacing, assuming both images are
    already aligned in the same physical space. Suitable for coarsely aligned
    data or cases that do not require an exact spatial transformation.
    
    Args:
        ct_image: CT image with shape (H, W) or (C, H, W)
        pet_shape: PET image size (H, W)
        ct_pixel_spacing: CT pixel spacing (row, column) in mm
        pet_pixel_spacing: PET pixel spacing (row, column) in mm
        device: Compute device
    
    Returns:
        Resampled CT image
    """
    # Convert to a torch tensor
    if isinstance(ct_image, np.ndarray):
        ct_tensor = torch.from_numpy(ct_image).float()
    else:
        ct_tensor = ct_image.float()
    
    # Ensure 4D tensor shape (B, C, H, W)
    if ct_tensor.ndim == 2:
        ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)
    elif ct_tensor.ndim == 3:
        ct_tensor = ct_tensor.unsqueeze(0)
    
    ct_tensor = ct_tensor.to(device)
    
    # Resize directly to the target size
    aligned_ct = F.interpolate(
        ct_tensor,
        size=pet_shape,
        mode='bilinear',
        align_corners=True
    )
    
    return aligned_ct.squeeze()


def align_ct_to_pet_with_affine(
    ct_image: Union[np.ndarray, torch.Tensor],
    ct_affine: np.ndarray,
    pet_affine: np.ndarray,
    pet_shape: Tuple[int, int],
    device: str = 'cuda',
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Transform a CT image into the PET coordinate system and resolution using
    precomputed affine matrices (2D version). This avoids repeated DICOM reads
    when the dataset already returns affine matrices.
    
    Args:
        ct_image: CT image with shape (H, W), (C, H, W), or (B, C, H, W)
        ct_affine: CT affine transformation matrix [4, 4]
        pet_affine: PET affine transformation matrix [4, 4]
        pet_shape: PET image size (H, W)
        device: Compute device
        mode: Interpolation mode ('bilinear' or 'nearest')
    
    Returns:
        Aligned CT image in the PET coordinate system and resolution.
        The output dimensionality matches the input dimensionality.
    """
    # Convert to a torch tensor
    if isinstance(ct_image, np.ndarray):
        ct_tensor = torch.from_numpy(ct_image).float()
    else:
        ct_tensor = ct_image.float()
    
    # Record the original dimensionality
    original_ndim = ct_tensor.ndim
    
    # Ensure 4D tensor shape (B, C, H, W)
    if ct_tensor.ndim == 2:
        ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)
    elif ct_tensor.ndim == 3:
        ct_tensor = ct_tensor.unsqueeze(0)
    
    # Move to the requested device
    ct_tensor = ct_tensor.to(device)
    
    # Ensure affine matrices are NumPy arrays
    if isinstance(ct_affine, torch.Tensor):
        ct_affine = ct_affine.cpu().numpy()
    if isinstance(pet_affine, torch.Tensor):
        pet_affine = pet_affine.cpu().numpy()
    
    # Create the sampling grid
    grid = create_sampling_grid(
        output_size=pet_shape,
        ct_shape=ct_tensor.shape[-2:],
        ct_affine=ct_affine,
        pet_affine=pet_affine,
        device=device
    )
    
    # Expand the grid to match the batch size
    if grid.shape[0] != ct_tensor.shape[0]:
        grid = grid.expand(ct_tensor.shape[0], -1, -1, -1)
    
    # Resample with grid_sample
    aligned_ct = F.grid_sample(
        ct_tensor,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=True
    )
    
    # Restore the original dimensionality
    if original_ndim == 2:
        aligned_ct = aligned_ct.squeeze(0).squeeze(0)
    elif original_ndim == 3:
        aligned_ct = aligned_ct.squeeze(0)
    
    return aligned_ct


def align_ct_to_pet_batch(
    ct_batch: torch.Tensor,
    ct_affines: Union[np.ndarray, torch.Tensor],
    pet_affines: Union[np.ndarray, torch.Tensor],
    pet_shape: Tuple[int, int],
    device: str = 'cuda',
    mode: str = 'bilinear'
) -> torch.Tensor:
    """
    Transform a batch of CT images into the PET coordinate system (2D version).
    Each sample uses its own affine matrices.
    
    Args:
        ct_batch: Batch of CT images (B, C, H, W)
        ct_affines: Batch of CT affine matrices (B, 4, 4)
        pet_affines: Batch of PET affine matrices (B, 4, 4)
        pet_shape: PET image size (H, W)
        device: Compute device
        mode: Interpolation mode ('bilinear' or 'nearest')
    
    Returns:
        Batch of aligned CT images (B, C, pet_H, pet_W)
    """
    batch_size = ct_batch.shape[0]
    ct_batch = ct_batch.to(device)
    
    # Convert affine matrices to NumPy arrays
    if isinstance(ct_affines, torch.Tensor):
        ct_affines = ct_affines.cpu().numpy()
    if isinstance(pet_affines, torch.Tensor):
        pet_affines = pet_affines.cpu().numpy()
    
    aligned_list = []
    
    for i in range(batch_size):
        ct_single = ct_batch[i:i+1]  # (1, C, H, W)
        ct_affine = ct_affines[i]  # (4, 4)
        pet_affine = pet_affines[i]  # (4, 4)
        
        # Create the sampling grid
        grid = create_sampling_grid(
            output_size=pet_shape,
            ct_shape=ct_single.shape[-2:],
            ct_affine=ct_affine,
            pet_affine=pet_affine,
            device=device
        )
        
        # Resample
        aligned = F.grid_sample(
            ct_single,
            grid,
            mode=mode,
            padding_mode='zeros',
            align_corners=True
        )
        aligned_list.append(aligned)
    
    return torch.cat(aligned_list, dim=0)


# Usage example
if __name__ == "__main__":
    """
    Usage examples:
    
    # Full version using DICOM metadata
    from utils.spatial_transform import align_ct_to_pet
    
    ct_aligned = align_ct_to_pet(
        ct_image=ct_img,
        pet_image=pet_img,
        ct_dicom_path='path/to/ct.dcm',
        pet_dicom_path='path/to/pet.dcm',
        device='cuda'
    )
    
    # Simplified version using resizing only
    from utils.spatial_transform import align_ct_to_pet_simple
    
    ct_aligned = align_ct_to_pet_simple(
        ct_image=ct_img,
        pet_shape=(400, 400),
        ct_pixel_spacing=(0.977, 0.977),
        pet_pixel_spacing=(4.073, 4.073),
        device='cuda'
    )
    
    # Version using precomputed affine matrices (2D, suitable for training loops)
    from utils.spatial_transform import align_ct_to_pet_with_affine, align_ct_to_pet_batch
    
    # Single image
    ct_aligned = align_ct_to_pet_with_affine(
        ct_image=ct_img,  # (H, W) or (C, H, W)
        ct_affine=ct_affine,  # (4, 4)
        pet_affine=pet_affine,  # (4, 4)
        pet_shape=(400, 400),
        device='cuda'
    )
    
    # Batch processing
    ct_aligned_batch = align_ct_to_pet_batch(
        ct_batch=ct_batch,  # (B, C, H, W)
        ct_affines=ct_affines,  # (B, 4, 4)
        pet_affines=pet_affines,  # (B, 4, 4)
        pet_shape=(400, 400),
        device='cuda'
    )
    """
    pass
