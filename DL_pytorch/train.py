import torch
import torchvision
from utils.get_transform import get_transform 
from XXDataset import XXXDataset
from networks.XXXNet import XXXNet
import torch.optim as optim
import torch.nn as nn
from settings import params

def train(dataloader, net, criterion, optimizer):

    for epoch in range(params['max_epoches']):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % params['print_freq'] == 0:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0

    print('Finished Training')

if __name__ == '__main__':
    # prepare dataset
    transform = get_transform()
    trainset = XXXDataset(root='./data/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # prepare network
    net = XXXNet()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=params['lr'], momentum=params['momentum'])

    # train
    train(trainloader, net, criterion, optimizer)

    # save model
    save_model_path = os.path.join('models', params['model_name'])
    torch.save(net.state_dict(), save_model_path)
