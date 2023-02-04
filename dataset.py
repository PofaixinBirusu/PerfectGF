import numpy as np
import open3d as o3d
import torch
import os
import laspy
from torch.utils import data
from scipy.spatial.transform import Rotation
from utils import get_point_cloud, get_inputs
from scipy.spatial.transform import Rotation
from config import Config, Config150, ConfigMFR
from dataloader import get_dataloader


class OpenGF250(data.Dataset):
    def __init__(self, root, dir="train", pts_num=50000):
        for _, _, filelist in os.walk(root+"/"+dir):
            self.filelist = [root+"/"+dir+"/"+filename for filename in filelist]
        self.dir = dir
        self.sample_num = pts_num
        self.config = Config()

    def __len__(self):
        return len(self.filelist) * 36

    def __getitem__(self, index):
        file_idx = index // 36
        laz = laspy.read(self.filelist[file_idx])
        name = self.filelist[file_idx][self.filelist[file_idx].rfind("/")+1:self.filelist[file_idx].rfind(".")]
        inp, clz = laz.xyz, np.asarray(laz.classification)
        valid_idx = (clz != 0)
        inp, clz = inp[valid_idx], clz[valid_idx]-1
        # print(inp.shape)
        # pc = get_point_cloud(inp, color=[0, 0.651, 0.929], estimate_normal=True)
        # o3d.draw_geometries([pc], width=1000, height=800, window_name="open gf")
        coor_max, coor_min = np.max(inp, axis=0), np.min(inp, axis=0)
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], index))
        # print(inp.shape, clz.shape)
        # print(clz)
        range_idx = index % 36
        xrange = [[0, 250], [50, 300], [100, 350], [150, 400], [200, 450], [250, 500]]
        yrange = [[0, 250], [50, 300], [100, 350], [150, 400], [200, 450], [250, 500]]
        x_shift, y_shift = inp[:, 0] - coor_min[0], inp[:, 1] - coor_min[1]
        # print(list(xrange), list(yrange))
        # overlap split
        y = yrange[range_idx // 6]
        x = xrange[range_idx % 6]

        y_selected = (y_shift >= y[0]) & (y_shift < y[1])
        idx = (x_shift >= x[0]) & (x_shift < x[1]) & y_selected
        idx = np.nonzero(idx)[0]
        idx = idx[np.random.permutation(idx.shape[0])[:self.sample_num]]
        sub_xyz, sub_cls = inp[idx], clz[idx]
        sub_xyz = (sub_xyz - np.mean(sub_xyz, axis=0, keepdims=True)) / 125 * 1.5
        # coor_max, coor_min = np.max(sub_xyz, axis=0), np.min(sub_xyz, axis=0)
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], index))

        if self.dir == "train":
            euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
            euler_ab[1] = 0
            euler_ab[2] = 0
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            # print(rot_ab)
            sub_xyz = np.matmul(rot_ab, sub_xyz.T).T
        feats = np.ones(shape=(sub_xyz.shape[0], 1))
        # sub_pc = get_point_cloud(sub_xyz, color=[0, 0.651, 0.929], estimate_normal=True)
        # o3d.draw_geometries([sub_pc], width=1000, height=800, window_name="pc")
        return sub_xyz, feats, sub_cls


class OpenGF150(data.Dataset):
    def __init__(self, root, dir="train", pts_num=30000):
        for _, _, filelist in os.walk(root+"/"+dir):
            self.filelist = [root+"/"+dir+"/"+filename for filename in filelist]
        self.dir = dir
        self.sample_num = pts_num
        self.config = Config()
        # self.config = Config150()

    def __len__(self):
        return len(self.filelist) * 36

    def __getitem__(self, index):
        file_idx = index // 36
        laz = laspy.read(self.filelist[file_idx])
        name = self.filelist[file_idx][self.filelist[file_idx].rfind("/")+1:self.filelist[file_idx].rfind(".")]
        inp, clz = laz.xyz, np.asarray(laz.classification)
        valid_idx = (clz != 0)
        inp, clz = inp[valid_idx], clz[valid_idx]-1
        # print(inp.shape)
        # pc = get_point_cloud(inp, color=[0, 0.651, 0.929], estimate_normal=True)
        # o3d.draw_geometries([pc], width=1000, height=800, window_name="open gf")
        coor_max, coor_min = np.max(inp, axis=0), np.min(inp, axis=0)
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], index))
        # print(inp.shape, clz.shape)
        # print(clz)
        range_idx = index % 36
        xrange = [[0, 150], [70, 220], [140, 290], [210, 360], [280, 430], [350, 500]]
        yrange = [[0, 150], [70, 220], [140, 290], [210, 360], [280, 430], [350, 500]]
        x_shift, y_shift = inp[:, 0] - coor_min[0], inp[:, 1] - coor_min[1]
        # print(list(xrange), list(yrange))
        # overlap split
        y = yrange[range_idx // 6]
        x = xrange[range_idx % 6]

        y_selected = (y_shift >= y[0]) & (y_shift < y[1])
        idx = (x_shift >= x[0]) & (x_shift < x[1]) & y_selected
        idx = np.nonzero(idx)[0]
        idx = idx[np.random.permutation(idx.shape[0])[:self.sample_num]]
        sub_xyz, sub_cls = inp[idx], clz[idx]
        sub_xyz = (sub_xyz - np.mean(sub_xyz, axis=0, keepdims=True)) / 75 * 1.5
        # coor_max, coor_min = np.max(sub_xyz, axis=0), np.min(sub_xyz, axis=0)
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], index))

        if self.dir == "train":
            euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
            euler_ab[1] = 0
            euler_ab[2] = 0
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            # print(rot_ab)
            sub_xyz = np.matmul(rot_ab, sub_xyz.T).T
        feats = np.ones(shape=(sub_xyz.shape[0], 1))
        # sub_pc = get_point_cloud(sub_xyz, color=[0, 0.651, 0.929], estimate_normal=True)
        # o3d.draw_geometries([sub_pc], width=1000, height=800, window_name="pc")
        return sub_xyz, feats, sub_cls


class PackageDataset(data.Dataset):
    def __init__(self, pts_list, feats_list, clz_list, cfg):
        self.pts_list, self.feats_list, self.clz_list = pts_list, feats_list, clz_list
        self.config = cfg

    def __len__(self):
        return len(self.pts_list)

    def __getitem__(self, item):
        return self.pts_list[item], self.feats_list[item], self.clz_list[item]


def package(pts_list, feats_list, clz_list, batch_size, neighborhood_limits, cfg, num_workers=0, device=torch.device("cuda:0")):
    sub_patch_loader, _ = get_dataloader(
        PackageDataset(pts_list, feats_list, clz_list, cfg),
        batch_size,
        num_workers=num_workers, shuffle=False,
        neighborhood_limits=neighborhood_limits
    )
    package_loader_iter = sub_patch_loader.__iter__()
    package_inputs = []
    for i in range(len(pts_list) // batch_size):
        # package_inputs.append(get_inputs(package_loader_iter.next(), device))
        package_inputs.append(package_loader_iter.next())
    return package_inputs


class OpenGF150In250(data.Dataset):
    def __init__(self, root, dir="train", pts_num=10000):
        for _, _, filelist in os.walk(root+"/"+dir):
            self.filelist = [root+"/"+dir+"/"+filename for filename in filelist]
        self.dir = dir
        self.sample_num = pts_num
        self.config = ConfigMFR()
        self.config150 = Config()

    def __len__(self):
        return len(self.filelist) * 36

    def __getitem__(self, index):
        file_idx = index // 36
        laz = laspy.read(self.filelist[file_idx])
        inp, clz = laz.xyz, np.asarray(laz.classification)
        valid_idx = (clz != 0)
        inp, clz = inp[valid_idx], clz[valid_idx]-1
        # print(inp.shape)
        # pc = get_point_cloud(inp, color=[0, 0.651, 0.929], estimate_normal=True)
        # o3d.draw_geometries([pc], width=1000, height=800, window_name="open gf")
        coor_max, coor_min = np.max(inp, axis=0), np.min(inp, axis=0)
        # print("x: %.3f - %.3f, %.3f   y: %.3f - %.3f, %.3f  z: %.3f - %.3f, %.3f  %d" % (coor_min[0], coor_max[0], coor_max[0]-coor_min[0], coor_min[1], coor_max[1], coor_max[1]-coor_min[1], coor_min[2], coor_max[2], coor_max[2]-coor_min[2], index))
        # print(inp.shape, clz.shape)
        # print(clz)
        range_idx = index % 36
        xrange = [[0, 250], [50, 300], [100, 350], [150, 400], [200, 450], [250, 500]]
        yrange = [[0, 250], [50, 300], [100, 350], [150, 400], [200, 450], [250, 500]]
        x_shift, y_shift = inp[:, 0] - coor_min[0], inp[:, 1] - coor_min[1]
        # print(list(xrange), list(yrange))
        # overlap split
        y = yrange[range_idx // 6]
        x = xrange[range_idx % 6]

        y_selected = (y_shift >= y[0]) & (y_shift < y[1])
        idx = (x_shift >= x[0]) & (x_shift < x[1]) & y_selected
        idx = np.nonzero(idx)[0]
        idx = idx[np.random.permutation(idx.shape[0])[:self.sample_num]]
        xyz, clz = inp[idx], clz[idx]
        coor_min = np.min(xyz, axis=0)
        xyz = xyz - np.asarray([[coor_min[0].item(), coor_min[1].item(), 0]])
        # sub_idx记录对应的sub_xyz是从xyz中的哪几个idx中取的
        sub_xyz, sub_idx, sub_clz, sub_feats = [], [], [], []
        sub_xrange = [[0, 150], [100, 250]]
        sub_yrange = [[0, 150], [100, 250]]

        rot_ab = None
        if self.dir == "train":
            euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
            euler_ab[1] = 0
            euler_ab[2] = 0
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()

        for i in range(len(sub_xrange)):
            sub_x_selected_idx = (xyz[:, 0] >= sub_xrange[i][0]) & (xyz[:, 0] < sub_xrange[i][1])
            for j in range(len(sub_yrange)):
                sub_selected_idx = (xyz[:, 1] >= sub_yrange[j][0]) & (xyz[:, 1] < sub_yrange[j][1]) & sub_x_selected_idx
                sub_selected_idx = np.nonzero(sub_selected_idx)[0]
                sub_patch = xyz[sub_selected_idx]
                sub_patch = (sub_patch - np.mean(sub_patch, axis=0, keepdims=True)) / 75 * 1.5
                if self.dir == "train":
                    sub_patch = np.matmul(rot_ab, sub_patch.T).T
                sub_xyz.append(sub_patch)
                # print(sub_patch.shape)
                # sub_pc = get_point_cloud(sub_patch, [0, 0.651, 0.929], True)
                # o3d.draw_geometries([sub_pc], window_name="sub%d" % (i*3+j+1), width=1000, height=800)
                sub_clz.append(clz[sub_selected_idx])
                sub_idx.append(sub_selected_idx)
                sub_feats.append(np.ones(shape=(sub_patch.shape[0], 1)))

        sub_patch_inputs = package(sub_xyz, sub_feats, sub_clz, batch_size=2, neighborhood_limits=self.config150.neighborhood_limits, cfg=self.config150, num_workers=0)

        xyz = (xyz - np.mean(xyz, axis=0, keepdims=True)) / 125 * 1.5
        # print(sum([subxyz.shape[0] for subxyz in sub_xyz])+xyz.shape[0])
        if self.dir == "train":
            xyz = np.matmul(rot_ab, xyz.T).T
        inputs = package([xyz], [np.zeros(shape=(1, 1))], [clz], batch_size=1, neighborhood_limits=self.config.neighborhood_limits, cfg=self.config, num_workers=0)

        return inputs[0], sub_patch_inputs, sub_idx


if __name__ == '__main__':
    opengf = OpenGF250("E:/OpenGF_Exp", dir="train")
    from dataloader import get_dataloader
    from utils import get_inputs
    train_loader, _ = get_dataloader(opengf, 1, num_workers=0, shuffle=True,
                                     neighborhood_limits=opengf.config.neighborhood_limits)
    c_loader_iter = train_loader.__iter__()
    num_iter = int(len(train_loader.dataset) // train_loader.batch_size)
    for c_iter in range(num_iter):
        ##################################
        # load inputs to device.
        inputs = c_loader_iter.next()
        pts_list = inputs["points"]
        nei_list = inputs["neighbors"]
        poor_list = inputs["pools"]
        upsample = inputs["upsamples"]
        for i in range(len(upsample)):
            print(upsample[i].shape)
        for i, pts in enumerate(pts_list):
            print(pts.shape, nei_list[i].shape, poor_list[i].shape)
            pc = get_point_cloud(pts.numpy(), [0, 0.651, 0.929], estimate_normal=True)
            o3d.draw_geometries([pc], width=1000, height=800, window_name="pc")