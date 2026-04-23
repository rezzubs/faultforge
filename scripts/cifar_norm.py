"""A script to compute the mean and standard deviation of the CIFAR-10 dataset."""

import torch
from torchvision import datasets, transforms


def main():
    dataset = datasets.CIFAR10(
        "./data", train=True, transform=transforms.ToTensor(), download=True
    )

    data = torch.stack([image for image, _ in dataset])  # ToTensor gives (N, C, H, W)
    mean = data.mean(dim=[0, 2, 3])

    # unbiased=False because we have the full population.
    std = data.std(dim=[0, 2, 3], unbiased=False)

    # 10 decimal places gives us full f32 precision
    torch.set_printoptions(precision=10)
    print(mean)
    print(std)


if __name__ == "__main__":
    main()
