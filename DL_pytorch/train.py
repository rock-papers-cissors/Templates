import torch
import torchvision
from utils.get_transform import get_transform 
from XXDataset import XXXDataset
from networks.XXXNet import XXXNet
import torch.optim as optim
import torch.nn as nn

def train(dataloader, net, criterion, optimizer, params, device):

    for epoch in range(params['max_epoches']):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i+1) % params['print_freq'] == 0:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0

        if (epoch+1)%params['lr_freq'] == 0:
            adjust_learning_rate(optimizer, epoch, params['lr'], params['lr_decay'], params['lr_freq'])

    print('Finished Training')


def adjust_learning_rate(optimizer, epoch, lr_init, lr_decay, lr_freq):
    lr = lr_init * ( lr_decay ** (epoch // lr_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main(params):
    # reproducitiblity
    torch.manual_seed(0) 
    np.random.seed(0) 
    torch.backends.cudnn.deterministic = False # cuDNN deterministically select an algorithm, possibly at the cost of reduced performance
    torch.set_deterministic(True) # optional, some operation is without a deterministic alternative and would throw an error

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device', device)


    # prepare dataset
    transform = get_transform()
    trainset = XXXDataset(root='./data/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # prepare network
    net = XXXNet().to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=params['lr'], momentum=params['momentum'])

    # train
    train(trainloader, net, criterion, optimizer, params, device)

    # save model
    save_model_path = os.path.join('models', params['model_name'])
    torch.save(net.state_dict(), save_model_path)


if __name__ == '__main__':
    from settings import params, fix_settings
    fix_settings()
    main(params)
    
