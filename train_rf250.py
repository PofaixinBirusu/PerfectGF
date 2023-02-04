import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import OpenGF250
from utils import WeightedBCELoss, processbar, get_inputs, validate_gradient
from model import SCFNet
from evaluate import evaluate
from dataloader import get_dataloader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 1
epoch = 100
learning_rate = 0.01
min_learning_rate = 0.0001
learning_rate_decay_gamma = 0.95
loss_fn = WeightedBCELoss()
params_save_path = "./params/SCFNet-RF250.pth"

net = SCFNet()
net.to(device)
net.load_state_dict(torch.load(params_save_path))
# optimizer = torch.optim.SGD(lr=learning_rate, params=net.parameters(), momentum=0.9)
optimizer = torch.optim.Adam(lr=learning_rate, params=net.parameters())

train_dataset = OpenGF250("./data/OpenGF_Exp", dir="train", pts_num=50000)
# train_loader, _ = get_dataloader(train_dataset, batch_size, num_workers=2, shuffle=True, neighborhood_limits=None)
train_loader, _ = get_dataloader(train_dataset, batch_size, num_workers=2, shuffle=True, neighborhood_limits=train_dataset.config.neighborhood_limits)
val_dataset = OpenGF250("./data/OpenGF_Exp", dir="val", pts_num=50000)
val_loader, _ = get_dataloader(val_dataset, batch_size, num_workers=2, shuffle=False, neighborhood_limits=val_dataset.config.neighborhood_limits)


def update_lr(optimizer, gamma=0.5):
    global learning_rate
    learning_rate = max(learning_rate*gamma, min_learning_rate)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    print("lr update finished  cur lr: %.5f" % learning_rate)


if __name__ == '__main__':
    max_oa = 0
    for epoch_count in range(1, 1+epoch):
        net.train()
        loss_val, precision_val, recall_val, acc_val, processed = 0, 0, 0, 0, 0
        num_iter = int(len(train_loader.dataset) // train_loader.batch_size)
        c_loader_iter = train_loader.__iter__()
        for c_iter in range(num_iter):
            inputs = c_loader_iter.next()
            inputs = get_inputs(inputs, device)
            clz = inputs["cls"]
            predict = net(inputs)
            loss, result = loss_fn(predict, clz)

            optimizer.zero_grad()
            loss.backward()
            if validate_gradient(net):
                optimizer.step()
            else:
                print("\ngradient error")
            optimizer.step()

            loss_val += loss.item()
            processed += result.shape[0]
            loss_precision_recall_sum = torch.sum(result, dim=0)
            precision_val, recall_val = precision_val+loss_precision_recall_sum[1].item(), recall_val+loss_precision_recall_sum[2].item()
            for i in range(len(predict)):
                acc_val += (predict[i].view(-1).round() == clz[i]).sum(dim=0).item() / predict[i].shape[0]
            cur_mean_acc, cur_mean_precision, cur_mean_recall = acc_val/processed, precision_val/processed, recall_val/processed
            print("\r进度：%s  本批loss:%.5f   precision: %.5f  recall: %.5f  acc: %.5f" % (processbar(processed, len(train_dataset)), loss.item(), cur_mean_precision, cur_mean_recall, cur_mean_acc), end="")
        mean_acc, mean_precision, mean_recall = acc_val/len(train_dataset), precision_val/len(train_dataset), recall_val/len(train_dataset)
        print("\nepoch: %d  本轮loss:%.5f   precision: %.5f  recall: %.5f  acc: %.5f" % (epoch_count, loss_val, mean_precision, mean_recall, mean_acc))
        print("开始测试...")
        OA, IoU1, IoU2, mIoU, precision, recall = evaluate(net, val_loader)
        f1_score = 2*precision*recall/(precision+recall)
        if max_oa < OA:
            max_oa = OA
            print("save...")
            torch.save(net.state_dict(), params_save_path)
            print("save finished !!!")
        # 每轮后 lr = lr * 0.95
        update_lr(optimizer, learning_rate_decay_gamma)