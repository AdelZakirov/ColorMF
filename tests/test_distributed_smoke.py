import unittest

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class DistributedSmokeTests(unittest.TestCase):
    def test_environment_reports_available_devices(self):
        # The actual Lightning DDP smoke run belongs to the pilot environment;
        # this test prevents a local CPU-only checkout from claiming GPUs.
        self.assertGreaterEqual(torch.cuda.device_count(), 0)

    @unittest.skipUnless(torch.cuda.device_count() > 1, "requires at least two GPUs")
    def test_two_gpu_lightning_batch(self):
        from src.lightning_module import PMFColorizerModule

        class SyntheticDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                generator = torch.Generator().manual_seed(index)
                return {
                    "ab": torch.randn(2, 16, 16, generator=generator),
                    "L": torch.randn(1, 16, 16, generator=generator),
                    "image_id": str(index),
                }

        class SyntheticDataModule(pl.LightningDataModule):
            def train_dataloader(self):
                return DataLoader(SyntheticDataset(), batch_size=1)

        module = PMFColorizerModule(
            model={
                "resolution": 16,
                "patch_size": 4,
                "hidden_size": 32,
                "depth": 2,
                "heads": 4,
                "aux_head_depth": 1,
                "pca_channels": 8,
            },
            warmup_steps=1,
            max_steps=1,
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=2,
            strategy="ddp",
            max_steps=1,
            limit_train_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(module, datamodule=SyntheticDataModule())


if __name__ == "__main__":
    unittest.main()
