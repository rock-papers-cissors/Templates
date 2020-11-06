import torch
import torchvision
from utils.get_transform import get_transfrom
from XXDataset import XXXDataset
from networks.XXXNet import XXXNet
import torch.optim as optim
import torch.nn as nn
from settings import params, fix_settings

def test(dataloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))

if __name__ == '__main__':

    fix_settings()

    transform = get_transform()
    testset = XXXDataset(root='./data/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    # prepare network and load saved model
    net = XXXNet()
    save_model_path = os.path.join('models', params['model_name'])
    net.load_state_dict(torch.load(save_model_path))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # train
    return test(testloader, net)
