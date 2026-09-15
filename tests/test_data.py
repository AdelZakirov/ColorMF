import numpy as np

from src.data import adm_center_crop


def test_adm_center_crop_repeated_box_then_bicubic_geometry():
    # Large non-square input forces two BOX reductions before the final resize.
    image = np.arange(1024 * 768 * 3, dtype=np.uint32).reshape(1024, 768, 3)
    image = (image % 256).astype(np.uint8)
    cropped = adm_center_crop(image, 128)
    assert cropped.shape == (128, 128, 3)
    # A one-shot bicubic resize is intentionally not the official result.
    import cv2
    stretched = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
    assert not np.array_equal(cropped, stretched)
