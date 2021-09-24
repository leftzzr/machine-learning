#mnist手写数据集三层全连接神经网络
import torch
import numpy
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Batch_net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# class Batch_net(nn.Module):
#     def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
#         super(Batch_net, self).__init__()
#         self.layer1 = nn.Linear(in_dim, n_hidden_1)
#         self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
#         self.layer3 = nn.Linear(n_hidden_2, out_dim)
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x

batch_size = 64
learning_rate = 1e-2
num_epoches = 20


data_tf = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

train_dataset = datasets.MNIST(root='./data', train=True, download=True,transform=transforms.ToTensor())

test_dataset = datasets.MNIST(root='./data', train=False,transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = Batch_net(28*28, 300, 100, 10)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


model.eval()
eval_loss = 0
eval_acc = 0


for data in test_loader:


    img, label = data
    img = img.view(img.size(0),-1)
    #img = transforms.ToTensor()(img)  # PILImage->tensor
    #label = torch.from_numpy(numpy.array(label))
    # image = torch.Tensor.permute(image, (0, 1, 2))  # tensor下维度转换
    # print(image.shape)

    #print(type(data))
    #print(type(img))
    #img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data*label.size(0)
    _,pred = torch.max(out,1)

    num_correct = (pred == label).sum()
    #print("c",num_correct)
    eval_acc += num_correct.item()
    print('test Loss: {:.6f}, Eval Acc: {:.6f}'.format(eval_loss/(len(test_dataset)), eval_acc/(len(test_dataset))))
























