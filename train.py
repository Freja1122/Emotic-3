import torch
from model import Net
from dataset import Emotic
import torch.optim as optim
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = Emotic('train')
train_dataloader = DataLoader(train_dataset, batch_size=40, shuffle=True)
val_dataset = Emotic('val')
val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=True)

net = Net().to(device)
image_feature_extraction_pretrained_model_file = 'resnet50_places365.pth.tar'
checkpoint = torch.load(image_feature_extraction_pretrained_model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
net.image_feature_extraction.load_state_dict(state_dict)
#net.image_feature_extraction = torch.load('image_feature_extraction_pretrained.pkl')

#criterion_dis = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)


def train(epoch, optimizer):
    net.train()
    loss_total = 0
    for idx, (image, body, dis_label, con_label) in enumerate(train_dataloader):
        image, body, dis_label, con_label = image.to(device), body.to(device), dis_label.to(device), con_label.to(device)
        dis, con = net(body, image)
        optimizer.zero_grad()
        loss = torch.sum(loss_fn(dis, dis_label)) + torch.sum(loss_fn(con, con_label))
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        loss_total += loss.item()
        if (idx + 1) % 25 == 0:
            print(epoch, idx + 1, loss_total/((idx+1)*40))
    print('train', epoch, loss_total / len(train_dataset))


def val(epoch):
    net.eval()
    loss = 0
    for idx, (image, body, dis_label, con_label) in enumerate(val_dataloader):
        image, body, dis_label, con_label = image.to(device), body.to(device), dis_label.to(device), con_label.to(device)
        dis, con = net(body, image)
        loss += (torch.sum(loss_fn(dis, dis_label)).item() + torch.sum(loss_fn(con, con_label)))
    loss /= len(val_dataset)
    print('val', epoch, loss)
    torch.save(net, 'net/net' + str(epoch) + '.pkl')
    return loss


def main():
    count_ascend = 0
    print('start')
    last_loss = 1e100
    #net.load_state_dict(torch.load('net/net1.pth'))
    lr = 1e-3
    for epoch in range(100):
        optimizer = optim.Adam(net.parameters(), lr=lr)
        train(epoch, optimizer)
        torch.save(net, 'net/net' + str(epoch) + '.pkl')
        #val(epoch)
		if epoch % 3 == 2:
            lr *= 0.9

if __name__ == '__main__':
    main()

