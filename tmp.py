import utils
import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

src_transform = transforms.Compose([
        # transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
# a = datasets.ImageFolder('data/src_data_path/train', src_transform)
# a = utils.data_load('data/src_data_path')
train_loader_fake = torch.utils.data.DataLoader(datasets.FakeData(100,(3,256,256), transform=src_transform), batch_size=8, shuffle=True, drop_last=True)
# print(a.__dict__)
print(datasets.FakeData(100,(3,256,256), transform=src_transform).__dict__)
# # print(a.targets)
# b = torch.utils.data.DataLoader(a, batch_size=8, shuffle=True)

# for t in b:
#     print(t)
#     break
a = 1
b = 2
t = [a,b]
# for *[t[i] for i in range(2)] in train_loader_fake:
for t[0], t[1] in train_loader_fake:
	print(t[0])
	print(t[1])
	break
print(b)
t = [a,b]
# for *[t[i] for i in range(2)] in train_loader_fake:
for a, b in train_loader_fake:
	print(t[0])
	print(t[1])
	break
print(b)