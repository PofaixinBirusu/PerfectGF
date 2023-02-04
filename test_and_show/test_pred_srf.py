import numpy as np
import open3d as o3d
import torch
import laspy
from model import SCFNet
from utils import get_point_cloud
from utils import get_inputs
import warnings
from dataset import package
from config import Config

device = torch.device("cuda:0")
params_save_path = "../params/SCFNet-RF250.pth"
net = SCFNet()
net.to(device)
net.load_state_dict(torch.load(params_save_path))
net.eval()
inp_pt_num = 50000

cfg = Config()


def predict_patch(patch):
    pts_num = patch.shape[0]
    pred = torch.zeros((pts_num, )).to(device)
    if pts_num < inp_pt_num:
        inputs = package([patch], [np.ones(shape=(patch.shape[0], 1))], [np.zeros(shape=(1, 1))], batch_size=1,
                         neighborhood_limits=cfg.neighborhood_limits, cfg=cfg)[0]
        inputs = get_inputs(inputs, device)
        sub_pred = net(inputs)
        pred[:] = sub_pred[0].view(-1)
        return pred
    for st in range(0, pts_num-inp_pt_num+1, inp_pt_num):
        ed = st + inp_pt_num
        if pts_num - ed < inp_pt_num:
            ed = pts_num
        sub_patch = patch[st:ed, :]
        with torch.no_grad():
            if sub_patch.shape[0] > 70000:
                sub_patch_1 = sub_patch[:sub_patch.shape[0]//2, :]
                sub_patch_2 = sub_patch[sub_patch.shape[0]//2:, :]
                inputs1 = package([sub_patch_1], [np.ones(shape=(sub_patch_1.shape[0], 1))], [np.zeros(shape=(1, 1))], batch_size=1,
                                 neighborhood_limits=cfg.neighborhood_limits, cfg=cfg)[0]
                inputs2 = package([sub_patch_2], [np.ones(shape=(sub_patch_2.shape[0], 1))], [np.zeros(shape=(1, 1))], batch_size=1,
                                 neighborhood_limits=cfg.neighborhood_limits, cfg=cfg)[0]
                inputs1 = get_inputs(inputs1, device)
                inputs2 = get_inputs(inputs2, device)
                sub_pred_1 = net(inputs1)
                sub_pred_2 = net(inputs2)
                # sub_pred_1 = net(torch.Tensor(sub_patch_1).unsqueeze(0).to(device), torch.ones(1, sub_patch_1.shape[0], 1).to(device)).view(-1)
                # sub_pred_2 = net(torch.Tensor(sub_patch_2).unsqueeze(0).to(device), torch.ones(1, sub_patch_2.shape[0], 1).to(device)).view(-1)
                pred[st:st+sub_patch.shape[0]//2] = sub_pred_1[0].view(-1)
                pred[st+sub_patch.shape[0]//2:ed] = sub_pred_2[0].view(-1)
            else:
                inputs = package([sub_patch], [np.ones(shape=(sub_patch.shape[0], 1))], [np.zeros(shape=(1, 1))], batch_size=1,
                                 neighborhood_limits=cfg.neighborhood_limits, cfg=cfg)[0]
                inputs = get_inputs(inputs, device)
                sub_pred = net(inputs)
                pred[st:ed] = sub_pred[0].view(-1)
    return pred


def predict_test_scene(xyz, xcross, ycross, sep, save_path):
    coor_max, coor_min = np.max(xyz, axis=0), np.min(xyz, axis=0)
    x_shift, y_shift = xyz[:, 0] - coor_min[0], xyz[:, 1] - coor_min[1]
    cnt = 0
    xrange, yrange = range(0, xcross - sep, sep // 2), range(0, ycross - sep, sep // 2)
    # print(list(xrange), list(yrange))
    pred, pred_num = torch.zeros((xyz.shape[0],)).to(device), torch.zeros((xyz.shape[0],)).to(device)
    # overlap split
    for y in yrange:
        y_end = y + sep
        if ycross - y_end < sep // 2:
            y_end = ycross
        y_selected = (y_shift >= y) & (y_shift < y_end)
        for x in xrange:
            x_end = x + sep
            if xcross - x_end < sep // 2:
                x_end = xcross
            idx = (x_shift >= x) & (x_shift < x_end) & y_selected
            idx = np.nonzero(idx)[0]
            sub_xyz, sub_cls = xyz[idx], clz[idx]
            shuffle_idx = np.random.permutation(sub_xyz.shape[0])#[:50000]
            sub_xyz, sub_cls, idx = sub_xyz[shuffle_idx], sub_cls[shuffle_idx], idx[shuffle_idx]
            sub_xyz = (sub_xyz - np.mean(sub_xyz, axis=0, keepdims=True)) / (sep // 2) * 1.5
            # coor_max, coor_min = np.max(sub_xyz, axis=0), np.min(sub_xyz, axis=0)
            # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2]))

            patch_pred = predict_patch(sub_xyz)
            gt = torch.Tensor(sub_cls).to(device)
            # patch_pred[patch_pred < 0.5] = 0
            # patch_pred[patch_pred >= 0.5] = 1
            acc = (patch_pred.round() == gt).sum(dim=0).item() / sub_xyz.shape[0]
            # print((patch_pred.round() == gt).sum(dim=0).item(), sub_xyz.shape[0])
            cnt += 1
            print("%d / %d   acc: %.5f" % (cnt, len(xrange) * len(yrange), acc))
            # torch.cuda.empty_cache()
            pred[idx] += patch_pred
            pred_num[idx] += 1
            # # 画出错误
            # # print(patch_pred[torch.nonzero(patch_pred.round() == 0).view(-1)[(gt[torch.nonzero(patch_pred.round() == 0).view(-1)] == 1)]])
            # # sub_pc = get_point_cloud(sub_xyz, [0, 0.651, 0.929], True)
            # sub_pc = get_point_cloud(sub_xyz, [0.8, 0.8, 0.8], True)
            # np.asarray(sub_pc.colors)[torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()] = np.array(
            #     [1, 0.706, 0])
            #
            # np.asarray(sub_pc.points)[torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()[
            #     (gt[torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()] == 1).cpu().numpy()]] += 0.05
            # # 蓝色是地面，但预测成了物体
            # np.asarray(sub_pc.colors)[torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()[
            #     (gt[torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()] == 1).cpu().numpy()]] = np.array(
            #     [0, 0, 1])
            # np.asarray(sub_pc.points)[torch.nonzero(patch_pred.round() == 1).view(-1).cpu().numpy()[
            #     (gt[torch.nonzero(patch_pred.round() == 1).view(-1).cpu().numpy()] == 0).cpu().numpy()]] += 0.05
            # # 红色是物体，但预测成了地面
            # np.asarray(sub_pc.colors)[torch.nonzero(patch_pred.round() == 1).view(-1).cpu().numpy()[
            #     (gt[torch.nonzero(patch_pred.round() == 1).view(-1).cpu().numpy()] == 0).cpu().numpy()]] = np.array(
            #     [1, 0, 0])
            # # print(torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()[
            # #     (gt[torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()] == 1).cpu().numpy()])
            # o3d.draw_geometries([sub_pc], width=1500, height=1200, window_name="look wrong")

    # valid_idx = (pred_num > 0)
    # pred = pred[valid_idx] / pred_num[valid_idx]
    pred = pred / pred_num
    gt = torch.Tensor(clz).to(device)
    acc = (pred.round() == gt).sum(dim=0).item() / pred.shape[0]
    print("finish, acc: %.5f" % acc)
    print(pred.shape)
    np.save(save_path, pred.cpu().numpy())


def evaluate(pred, clz):
    pred, clz = torch.Tensor(pred).round().view(-1), torch.Tensor(clz)
    pred_ground_idx, pred_noground_path = torch.nonzero(pred == 1).view(-1), torch.nonzero(pred == 0).view(-1)
    P, F = clz[pred_ground_idx], clz[pred_noground_path]
    TP, FP = torch.sum(P, dim=0), P.shape[0] - torch.sum(P, dim=0)
    FN, TN = torch.sum(F, dim=0), F.shape[0] - torch.sum(F, dim=0)
    # | TP, FN |
    # | FP, TN |
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # 非地面点IoU
    IoU1 = TN / (TN + FN + FP)
    # 地面点IoU
    IoU2 = TP / (TP + FP + FN)
    print("accuracy: %.5f  IoU1(非地面): %.5f  IoU2(地面): %.5f" % (accuracy.item(), IoU1.item(), IoU2.item()))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    #######################   T1   ########################
    # laz = laspy.read("./OpenGF_Test/T1.laz")
    # xyz, clz = laz.xyz, np.asarray(laz.classification)-1
    # coor_max, coor_min = np.max(xyz, axis=0), np.min(xyz, axis=0)
    # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2]))
    # xcross, ycross, sep = 2635, 2505, 500
    # predict_test_scene(xyz, xcross, ycross, sep, None)
    # pred = np.load("./OpenGF_Test_Result/t1.npy")
    # pred_aug = np.load("./OpenGF_Test_Result/t1_100.npy")
    # pred = (pred*0.6+pred_aug*0.4)
    # clz = torch.Tensor(clz)
    # for th in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]:
    #     print(th)
    #     pred_ = torch.Tensor(np.copy(pred))
    #     pred_[pred_ < th] = 0
    #     pred_[pred_ >= th] = 1
    #     print((pred_.round() == clz).sum(dim=0).item() / clz.shape[0])


    # pred2 = np.load("./OpenGF_Test_Result/t1_100.npy")
    # for w1 in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #     w2 = 1-w1
    #     print(w1, w2)
    #     pred_fuse = (pred*w1+pred2*w2)
    #     evaluate(pred_fuse, clz)

    # #######################   T2   ########################
    laz = laspy.read("../OpenGF_Test/t2_withoutnoise.laz")
    xyz, clz = laz.xyz, np.asarray(laz.classification)-1
    # np.save("./test_data/T2.npy", np.concatenate([xyz, (clz-1).reshape(-1, 1)], axis=1))
    coor_max, coor_min = np.max(xyz, axis=0), np.min(xyz, axis=0)
    print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f" % (coor_min[0], coor_max[0], coor_max[0] - coor_min[0], coor_min[1], coor_max[1], coor_max[1] - coor_min[1], coor_min[2], coor_max[2], coor_max[2] - coor_min[2]))
    xcross, ycross, sep = 940, 1226, 250
    predict_test_scene(xyz, xcross, ycross, sep, "../OpenGF_Test_Result/t2-250F.npy")