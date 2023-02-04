import numpy as np
import torch
from utils import processbar, WeightedBCELoss, get_inputs
# from model import SCFNet
# from dataset import OpenGF250
# from torch.utils.data import DataLoader
import warnings

device = torch.device("cuda")
seg_classes = {'All Scenes': [0, 1]}
loss_fn = WeightedBCELoss(with_log=True)

warnings.filterwarnings("ignore")


def evaluate(net, test_loader):
    net.eval()
    loss_val, process, correct, process_pts = 0, 0, 0, 0
    precision_val, recall_val, oa, iou1, iou2 = 0, 0, 0, 0, 0
    with torch.no_grad():
        num_iter = int(len(test_loader.dataset) // test_loader.batch_size)
        c_loader_iter = test_loader.__iter__()
        for c_iter in range(num_iter):
            inputs = c_loader_iter.next()
            inputs = get_inputs(inputs, device)
            clz = inputs["cls"]
            predict = net(inputs)
            loss, result = loss_fn(predict, clz)
            for i in range(len(clz)):
                pred = predict[i].round().view(-1)
                pred_ground_idx, pred_noground_path = torch.nonzero(pred == 1).contiguous().view(-1), torch.nonzero(pred == 0).contiguous().view(-1)
                P, F = clz[i][pred_ground_idx], clz[i][pred_noground_path]
                TP, FP = torch.sum(P, dim=0), P.shape[0] - torch.sum(P, dim=0)
                FN, TN = torch.sum(F, dim=0), F.shape[0] - torch.sum(F, dim=0)
                # | TP, FN |
                # | FP, TN |
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                # 非地面点IoU
                IoU1 = TN / (TN + FN + FP)
                # 地面点IoU
                IoU2 = TP / (TP + FP + FN)

                oa += accuracy.item()
                iou1 += IoU1.item()
                iou2 += IoU2.item()
                precision_val += result[i][1].item()
                recall_val += result[i][2].item()
            process += len(clz)
            print("\rtest: %s, OA: %.5f  IoU1: %.5f  IoU2: %.5f  mIoU: %.5f  precision: %.5f  recall: %.5f" % (processbar(process, len(test_loader.dataset)), oa/process, iou1/process, iou2/process, (iou1+iou2)/(2*process), precision_val/process, recall_val/process), end="")

    OA, IoU1, IoU2, mIoU, precision, recall = oa/process, iou1/process, iou2/process, (iou1+iou2)/(2*process), precision_val/process, recall_val/process
    print("\ntest finished!  OA: %.5f  IoU1: %.5f  IoU2: %.5f  mIoU: %.5f  precision: %.5f  recall: %.5f" % (OA, IoU1, IoU2, mIoU, precision, recall))

    return OA, IoU1, IoU2, mIoU, precision, recall


def evaluate_mrf(net, test_loader):
    net.eval()
    loss_val, process, correct, process_pts = 0, 0, 0, 0
    precision_val, recall_val, oa, iou1, iou2 = 0, 0, 0, 0, 0
    with torch.no_grad():
        for inputs, sub_patch_inputs, sub_idx in test_loader:
            # for c_iter in range(num_iter):
            #     inputs = c_loader_iter.next()
            #     inputs = get_inputs(inputs, device)
            sub_idx = [torch.from_numpy(idx).long().to(device) for idx in sub_idx]
            inputs = get_inputs(inputs, device)
            sub_patch_inputs = [get_inputs(sub, device) for sub in sub_patch_inputs]
            clz = inputs["cls"]
            predict = net(inputs, sub_patch_inputs, sub_idx)
            loss, result = loss_fn(predict, clz)
            for j in range(len(clz)):
                pred = torch.sigmoid(predict[j]).round().view(-1)
                pred_ground_idx, pred_noground_path = torch.nonzero(pred == 1).contiguous().view(-1), torch.nonzero(pred == 0).contiguous().view(-1)
                P, F = clz[j][pred_ground_idx], clz[j][pred_noground_path]
                TP, FP = torch.sum(P, dim=0), P.shape[0] - torch.sum(P, dim=0)
                FN, TN = torch.sum(F, dim=0), F.shape[0] - torch.sum(F, dim=0)
                # | TP, FN |
                # | FP, TN |
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                # 非地面点IoU
                IoU1 = TN / (TN + FN + FP)
                # 地面点IoU
                IoU2 = TP / (TP + FP + FN)

                oa += accuracy.item()
                iou1 += IoU1.item()
                iou2 += IoU2.item()
                precision_val += result[j][1].item()
                recall_val += result[j][2].item()
            process += len(clz)
            print("\rtest: %s, OA: %.5f  IoU1: %.5f  IoU2: %.5f  mIoU: %.5f  precision: %.5f  recall: %.5f" % (processbar(process, len(test_loader.dataset)), oa/process, iou1/process, iou2/process, (iou1+iou2)/(2*process), precision_val/process, recall_val/process), end="")

    OA, IoU1, IoU2, mIoU, precision, recall = oa/process, iou1/process, iou2/process, (iou1+iou2)/(2*process), precision_val/process, recall_val/process
    print("\ntest finished!  OA: %.5f  IoU1: %.5f  IoU2: %.5f  mIoU: %.5f  precision: %.5f  recall: %.5f" % (OA, IoU1, IoU2, mIoU, precision, recall))

    return OA, IoU1, IoU2, mIoU, precision, recall


if __name__ == '__main__':
    # import warnings
    # warnings.filterwarnings("ignore")
    # params_save_path = "./params/SCFNet-OpenGF8192.pth"
    # net = SCFNet()
    # net.to(device)
    # net.load_state_dict(torch.load(params_save_path))
    # val_dataset = OpenGF("./OpenGF_8192", dir="val")
    # val_loader = DataLoader(dataset=val_dataset, batch_size=5, shuffle=False)
    # OA, mAcc, _, mIoU, precision, recall = evaluate(net, val_loader)
    pass