import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch

from Load_Dataset import ImageToImage2D, RandomGenerator


class DatasetConfigTests(unittest.TestCase):
    def test_mask_resolution_supports_tif_image_with_png_mask(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            img_dir = os.path.join(tmp_dir, "img")
            mask_dir = os.path.join(tmp_dir, "labelcol")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            open(os.path.join(img_dir, "case_001.tif"), "wb").close()
            open(os.path.join(mask_dir, "case_001.png"), "wb").close()

            dataset = ImageToImage2D(
                dataset_path=tmp_dir,
                task_name="MoNuSeg",
                row_text={},
                joint_transform=None,
                image_size=64,
                cache_dir=tmp_dir,
            )

            self.assertEqual(dataset._resolve_mask_filename("case_001.tif"), "case_001.png")

    def test_random_generator_returns_normalized_tensors(self) -> None:
        generator = RandomGenerator(output_size=[64, 64])
        sample = {
            "image": np.full((32, 48, 3), 255, dtype=np.uint8),
            "label": np.ones((32, 48), dtype=np.uint8),
            "text": torch.randn(5, 32),
            "attributes": torch.randn(4),
        }

        transformed = generator(sample)

        self.assertEqual(transformed["image"].shape, (3, 64, 64))
        self.assertEqual(transformed["label"].shape, (64, 64))
        self.assertEqual(transformed["image"].dtype, torch.float32)
        self.assertEqual(transformed["label"].dtype, torch.int64)
        self.assertGreaterEqual(float(transformed["image"].min().item()), 0.0)
        self.assertLessEqual(float(transformed["image"].max().item()), 1.0)


if __name__ == "__main__":
    unittest.main()
