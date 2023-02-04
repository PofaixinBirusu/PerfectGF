import numpy as np
import open3d as o3d
import laspy
import torch
from utils import get_point_cloud


def show(pred, clz, window_name):
    # sub_pc = get_point_cloud(sub_xyz, [0, 0.651, 0.929], True)
    np.asarray(pc.colors)[:, :] = np.array([0.8, 0.8, 0.8])
    patch_pred = torch.Tensor(pred)
    gt = torch.Tensor(clz)
    pred[torch.nonzero(patch_pred.round() == 1).view(-1).numpy()[torch.topk(torch.Tensor(xyz[torch.nonzero(patch_pred.round() == 1).view(-1).numpy()][:, 2]), k=370)[1].numpy()]] = 0

    np.asarray(pc.colors)[torch.nonzero(patch_pred.round() == 1).view(-1).cpu().numpy()] = np.array(
        [1, 0.706, 0])

    np.asarray(pc.points)[torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()[
        (gt[torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()] == 1).cpu().numpy()]] += 0.05
    # 蓝色是地面，但预测成了物体
    np.asarray(pc.colors)[torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()[
        (gt[torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()] == 1).cpu().numpy()]] = np.array(
        [0, 0, 1])
    np.asarray(pc.points)[torch.nonzero(patch_pred.round() == 1).view(-1).cpu().numpy()[
        (gt[torch.nonzero(patch_pred.round() == 1).view(-1).cpu().numpy()] == 0).cpu().numpy()]] += 0.05
    # 红色是物体，但预测成了地面
    np.asarray(pc.colors)[torch.nonzero(patch_pred.round() == 1).view(-1).cpu().numpy()[
        (gt[torch.nonzero(patch_pred.round() == 1).view(-1).cpu().numpy()] == 0).cpu().numpy()]] = np.array(
        [1, 0, 0])
    # print(torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()[
    #     (gt[torch.nonzero(patch_pred.round() == 0).view(-1).cpu().numpy()] == 1).cpu().numpy()])
    o3d.visualization.draw_geometries([pc], width=1500, height=1200, window_name=window_name)


print("\n#################### t2 ####################")
laz = laspy.read("../OpenGF_Test/t2_withoutnoise.laz")
xyz, clz = laz.xyz, np.asarray(laz.classification) - 1
coor_max, coor_min = np.max(xyz, axis=0), np.min(xyz, axis=0)
print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f" % (
coor_min[0], coor_max[0], coor_max[0] - coor_min[0], coor_min[1], coor_max[1], coor_max[1] - coor_min[1], coor_min[2],
coor_max[2], coor_max[2] - coor_min[2]))

pc = get_point_cloud(xyz, [0.8, 0.8, 0.8], True)
print("\n50 receptive field result:")
pred50 = np.load("../OpenGF_Test_Result/t2-50A.npy")
show(pred50, clz, "50 receptive field result")

print("\n150 receptive field result:")
pred150 = np.load("../OpenGF_Test_Result/t2-150F.npy")
show(pred150, clz, "150 receptive field result")

print("\n250 receptive field result:")
pred250 = np.load("../OpenGF_Test_Result/t2-250F.npy")
show(pred250, clz, "250 receptive field result")

print("\nadd 50 to 150 result:")
pred50to150 = 0.4 * pred50 + 0.6 * pred150
show(pred50to150, clz, "add 50 to 150 result")

print("\nadd 50 to 250 result:")
pred50to250 = 0.4 * pred50 + 0.6 * pred250
show(pred50to250, clz, "add 50 to 250 result")

print("\nadd 150 to 250 result:")
pred150to250 = 0.4 * pred150 + 0.6 * pred250
show(pred150to250, clz, "add 150 to 250 result")

print("\n150 -> 250  result:")
pred150in250 = np.load("../OpenGF_Test_Result/t2-150in250G.npy")
show(pred150in250, clz, "150 -> 250  result")

print("\nadd 50 to 150->250")
pred50to150in250 = 0.4 * pred50 + 0.6 * pred150in250
show(pred50to150in250, clz, "add 50 to 150->250  result")