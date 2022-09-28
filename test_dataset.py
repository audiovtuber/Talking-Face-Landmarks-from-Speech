import torch

from dataset import GridDataset, GridDataModule

def test_grid_samples_are_different():
    data_module = GridDataModule(batch_size=32)
    data_module.setup('fit')
    dataloader = data_module.train_dataloader()

    x, y = next(iter(dataloader))
    for idx in range(31):
        assert not torch.allclose(x[idx], x[idx+1])
        assert not torch.allclose(y[idx], y[idx+1])

if __name__ == '__main__':
    pass
