import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import OpenGF150In250
from utils import WeightedBCELoss, processbar, get_inputs, validate_gradient
from model import SCFMRFNet
from evaluate import evaluate, evaluate_mrf
from dataloader import get_dataloader_fmr
from torch.cuda.amp import autocast, GradScaler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 1
epoch = 100
learning_rate = 0.01
min_learning_rate = 0.0001
learning_rate_decay_gamma = 0.95
loss_fn = WeightedBCELoss(with_log=True)
params_save_path = "./params/SCFNet-MRF150In250.pth"

net = SCFMRFNet(mini_rf_params_path="./params/SCFNet-RF150-oldcfg.pth")
net.to(device)
net.load()
# net.load_state_dict(torch.load(params_save_path))
# optimizer = torch.optim.SGD(lr=learning_rate, params=net.parameters(), momentum=0.9)
optimizer = torch.optim.Adam(lr=learning_rate, params=net.parameters())

train_dataset = OpenGF150In250("./data/OpenGF_Exp", dir="train", pts_num=50000)
train_loader = get_dataloader_fmr(train_dataset, batch_size, num_workers=4, shuffle=True)
# train_loader, _ = get_mrf_dataloader(train_dataset, batch_size, num_workers=0, shuffle=True, neighborhood_limits=train_dataset.config.neighborhood_limits)
val_dataset = OpenGF150In250("./data/OpenGF_Exp", dir="val", pts_num=50000)
val_loader = get_dataloader_fmr(val_dataset, batch_size, num_workers=4, shuffle=False)

scaler = GradScaler()


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
        item = np.random.permutation(len(train_dataset))
        optimizer.zero_grad()
        for inputs, sub_patch_inputs, sub_idx in train_loader:
        # for c_iter in range(num_iter):
        #     inputs = c_loader_iter.next()
        #     inputs = get_inputs(inputs, device)
        #     i = item[idx].item()
        #     inputs, sub_patch_inputs, sub_idx = train_dataset[i]
            sub_idx = [torch.from_numpy(idx).long().to(device) for idx in sub_idx]
            inputs = get_inputs(inputs, device)
            sub_patch_inputs = [get_inputs(sub, device) for sub in sub_patch_inputs]
            clz = inputs["cls"]
            with autocast():
                predict = net(inputs, sub_patch_inputs, sub_idx)
                loss, result = loss_fn(predict, clz)
            # optimizer.zero_grad()
            # loss.backward()
            # if validate_gradient(net):
            #     optimizer.step()
            # else:
            #     print("\ngradient error")
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_val += loss.item()
            processed += len(predict)
            loss_precision_recall_sum = torch.sum(result[:1], dim=0)
            precision_val, recall_val = precision_val+loss_precision_recall_sum[1].item(), recall_val+loss_precision_recall_sum[2].item()
            for i in range(len(predict)):
                acc_val += (torch.sigmoid(predict[i]).view(-1).round() == clz[i]).sum(dim=0).item() / predict[i].shape[0]
            cur_mean_acc, cur_mean_precision, cur_mean_recall = acc_val/processed, precision_val/processed, recall_val/processed
            print("\r进度：%s  本批loss:%.5f   precision: %.5f  recall: %.5f  acc: %.5f" % (processbar(processed, len(train_dataset)), loss.item(), cur_mean_precision, cur_mean_recall, cur_mean_acc), end="")
        mean_acc, mean_precision, mean_recall = acc_val/len(train_dataset), precision_val/len(train_dataset), recall_val/len(train_dataset)
        print("\nepoch: %d  本轮loss:%.5f   precision: %.5f  recall: %.5f  acc: %.5f" % (epoch_count, loss_val, mean_precision, mean_recall, mean_acc))
        print("开始测试...")
        OA, IoU1, IoU2, mIoU, precision, recall = evaluate_mrf(net, val_loader)
        f1_score = 2*precision*recall/(precision+recall)
        if max_oa < OA:
            max_oa = OA
            print("save...")
            torch.save(net.state_dict(), params_save_path)
            print("save finished !!!")
        # 每轮后 lr = lr * 0.95
        update_lr(optimizer, learning_rate_decay_gamma)