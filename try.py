import torch
import torchvision
import torchvision.transforms as transforms



if __name__ == '__main__':
    # 定义数据预处理的转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]之间
    ])

    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    # 类别标签
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 遍历训练集并显示图像和标签
    for images, labels in trainloader:
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            # 反归一化
            image = image / 2 + 0.5
            image = image.numpy()

            # 显示图像和标签
            import matplotlib.pyplot as plt
            plt.imshow(image.transpose(1, 2, 0))
            plt.title(classes[label])
            plt.show()
