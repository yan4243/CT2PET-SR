# Dataset integration TODO

The original dataset modules were stored outside the source repository. They
are deliberately represented as placeholders in this initial snapshot.

Required follow-up:

1. Choose the public data format.
2. Implement the slice-level dataset.
3. Group slices by `case_id` for inference.
4. Document PET and CT normalization.
5. Document affine conventions.
6. Provide train/validation/test manifest examples.
7. Add a synthetic-data smoke example.

Do not silently replace affine PET/CT alignment with image resizing. The
training and inference paths expect modality-specific 4×4 affine matrices.
