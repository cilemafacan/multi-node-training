import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
import argparse
from datetime import datetime

from torch.nn.parallel import DistributedDataParallel as DDP

class ConvNet(nn.Module):
    def __init__(self,num_classes=10):
        super(ConvNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32,num_classes)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=2, type=int, help='number of nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    args = parser.parse_args()

    args.world_size = args.nodes * args.gpus
    os.environ['MASTER_ADDR'] = '10.0.0.8'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl',init_method='env://',world_size=args.world_size,rank=rank) 

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    batch_size = 100
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=args.world_size,rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0 , pin_memory=True, sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)  
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, 10, i+1, total_step, loss.item()))
            
    if gpu == 0:
        print('Training completed in {}'.format(datetime.now() - start))

if __name__ == '__main__':
    main()
    print("Done")


