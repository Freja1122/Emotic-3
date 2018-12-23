import torch
from dataset import Emotic
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataloader = DataLoader(Emotic('test'))

net = torch.load('net5/net4.pkl')

diss = []
diss_label = []


def test():
    net.eval()
    for image, body, dis_label, con_label in tqdm(dataloader):
        image, body, dis_label, con_label = image.to(device), body.to(device), dis_label.to(device), con_label.to(device)
        dis, con = net(body, image)
        dis_label = dis_label.type_as(dis)
        diss.append([dis[0][i].item() for i in range(26)])
        diss_label.append([dis_label[0][i].item() for i in range(26)])


if __name__ == '__main__':
    test()
    with open('net_output.pkl', 'wb') as f1:
        pickle.dump(diss, f1, pickle.HIGHEST_PROTOCOL)
    with open('artificial_label.pkl', 'wb') as f2:
        pickle.dump(diss_label, f2, pickle.HIGHEST_PROTOCOL)
