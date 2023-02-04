import numpy as np
import torch
from torch import nn


def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def closest_pool(x, inds):
    """
    Pools features from the closest neighbors. WARNING: this function assumes the neighbors are ordered.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] Only the first column is used for pooling
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get features for each pooling location [n2, d]
    return gather(x, inds[:, 0])


def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features


def relative_pos_transforming(xyz, neigh_idx, neighbor_xyz):
    # (n1+n2+n3) x 3   (n1+n2+n3) x k    (n1+n2+n3) x k x 3
    # batch_size, npts = xyz.shape[0], xyz.shape[1]
    batch_size, npts = xyz.shape[0], xyz.shape[1]
    # xyz_tile = xyz.view(batch_size, npts, 1, 3).repeat([1, 1, neigh_idx.shape[-1], 1])
    bn = xyz.shape[0]
    xyz_tile = xyz.view(bn, 1, 3).repeat([1, neigh_idx.shape[-1], 1])
    # (n1+n2+n3) x k x 3
    relative_xyz = xyz_tile - neighbor_xyz
    # (n1+n2+n3) x k x 1
    relative_alpha = torch.unsqueeze(torch.atan2(relative_xyz[:, :, 1], relative_xyz[:, :, 0]), dim=2)
    # (n1+n2+n3) x k
    relative_xydis = torch.sqrt(torch.sum(torch.square(relative_xyz[:, :, :2]), dim=-1))
    # (n1+n2+n3) x k x 1
    relative_beta = torch.unsqueeze(torch.atan2(relative_xyz[:, :, 2], relative_xydis), dim=-1)
    # (n1+n2+n3) x k x 1
    relative_dis = torch.sqrt(torch.sum(torch.square(relative_xyz), dim=2, keepdim=True))
    # (n1+n2+n3) x k x 7
    relative_info = torch.cat([relative_dis, xyz_tile, neighbor_xyz], dim=2)
    # print(relative_dis.shape)
    # negative exp of geometric distance
    exp_dis = torch.exp(-relative_dis)

    # volume of local region
    # (n1+n2+n3)
    local_volume = torch.pow(torch.max(torch.max(relative_dis, dim=2)[0], dim=1)[0], 3)

    return relative_info, relative_alpha, relative_beta, exp_dis, local_volume


class LPR(nn.Module):
    def __init__(self):
        super(LPR, self).__init__()

    def forward(self, xyz, neigh_idx, lengths):
        # x:         (n1+n2+n3) x 3
        # neigh_idx: (n1+n2+n3) x k
        # lengths:   [n1, n2, n3]
        batch_size, npts = xyz.shape[0], xyz.shape[1]
        # neighbor_xyz = gather_neighbour(xyz, neigh_idx)
        # (n1+n2+n3) x k x 3
        # neighbor_xyz = torch.cat((xyz, torch.zeros_like(xyz[:1, :].to(xyz.device)) + 1e6), 0)[neigh_idx, :]
        neighbor_xyz = torch.cat((xyz, torch.zeros_like(xyz[:1, :].to(xyz.device))), 0)[neigh_idx, :]
        # Relative position transforming
        # (n1+n2+n3) x k x 7  (n1+n2+n3) x k x 1  (n1+n2+n3) x k x 1  (n1+n2+n3) x k x 1    (n1+n2+n3)
        relative_info, relative_alpha, relative_beta, geometric_dis, local_volume = relative_pos_transforming(xyz, neigh_idx, neighbor_xyz)

        # Local direction calculation (angle)
        # (n1+n2+n3) x 3
        neighbor_mean = torch.mean(neighbor_xyz, dim=1)
        # 自己和邻域中心的连接作为方向
        # (n1+n2+n3) x 3
        direction = xyz - neighbor_mean
        # (n1+n2+n3) x k x 3
        direction_tile = direction.view(direction.shape[0], 1, 3).repeat([1, neigh_idx.shape[-1], 1])
        # (n1+n2+n3) x k x 1
        direction_alpha = torch.atan2(direction_tile[:, :, 1], direction_tile[:, :, 0]).unsqueeze(dim=2)
        # (n1+n2+n3) x k
        direction_xydis = torch.sqrt(torch.sum(torch.square(direction_tile[:, :, :2]), dim=2))
        # (n1+n2+n3) x k x 1
        direction_beta = torch.atan2(direction_tile[:, :, 2], direction_xydis).unsqueeze(dim=2)

        # Polar angle updating
        angle_alpha = relative_alpha - direction_alpha
        angle_beta = relative_beta - direction_beta
        # (n1+n2+n3) x k x 2
        angle_updated = torch.cat([angle_alpha, angle_beta], dim=2)
        # Generate local spatial representation
        # (n1+n2+n3) x k x 9
        local_rep = torch.cat([angle_updated, relative_info], dim=2)

        # Calculate volume ratio for GCF
        # (n1+n2+n3)
        global_dis = torch.sqrt(torch.sum(torch.square(xyz), dim=1))
        # # b x 1
        # global_volume = torch.pow(torch.max(global_dis, dim=0)[0], 3).unsqueeze(dim=1)
        prefix = 0
        global_volume = []
        for i in range(len(lengths)):
            length = lengths[i]
            global_volume.append(torch.max(global_dis[prefix:prefix+length], dim=0)[0].view(1).repeat([length]))
            prefix += length
        # (n1+n2+n3, )
        global_volume = torch.cat(global_volume, dim=0)

        # (n1+n2+n3) x n x 1
        lg_volume_ratio = (local_volume / global_volume).unsqueeze(dim=1)
        # (n1+n2+n3) x k x 9   (n1+n2+n3) x k x 1    (n1+n2+n3) x 1
        return local_rep, geometric_dis, lg_volume_ratio


class MLP(nn.Module):
    def __init__(self, channels):
        super(MLP, self).__init__()
        self.mlp = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.relu = nn.ModuleList()
        for i in range(len(channels)-1):
            self.mlp.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=1, stride=1))
            self.bn.append(nn.InstanceNorm1d(channels[i+1]))
            self.relu.append(nn.LeakyReLU(0.2))
        self.layer_num = len(channels)-1

    def forward(self, x):
        # n x c
        x = x.unsqueeze(0).permute([0, 2, 1])
        for i in range(self.layer_num):
            x = self.mlp[i](x)
            x = self.bn[i](x)
            x = self.relu[i](x)
        x = x.permute([0, 2, 1])
        return x.squeeze(0)



class DDAP(nn.Module):
    def __init__(self, d_in, d_out):
        super(DDAP, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(9, d_in, kernel_size=1, stride=1),
            nn.InstanceNorm1d(d_in),
            # nn.BatchNorm1d(d_in),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_in, d_out//2, kernel_size=1, stride=1),
            nn.InstanceNorm1d(d_out//2),
            # nn.BatchNorm1d(d_out//2),
            nn.LeakyReLU(0.2)
        )
        self.dap_fc1 = nn.Linear(d_in*2+2, d_in*2, bias=False)
        self.dap_fc2 = nn.Linear(d_out+2, d_out, bias=False)
        # self.dap_conv1d1 = nn.Sequential(
        #     nn.Linear(d_in*2, d_out//2),
        #     # nn.InstanceNorm1d(d_out//2),
        #     nn.BatchNorm1d(d_out//2),
        #     nn.LeakyReLU(0.2)
        # )
        self.dap_conv1d1 = MLP([d_in*2, d_out//2])
        # self.dap_conv1d2 = nn.Sequential(
        #     nn.Linear(d_out, d_out),
        #     # nn.InstanceNorm1d(d_out),
        #     nn.BatchNorm1d(d_out),
        #     nn.LeakyReLU(0.2)
        # )
        self.dap_conv1d2 = MLP([d_out, d_out])
        self.softmax = nn.Softmax(dim=1)
        self.local_polar_representation = LPR()

    def dualdis_att_pool(self, feature_set, f_dis, g_dis, opid):
        # (n1+n2+n3) x k x [id1: 2c, d_out]  (n1+n2+n3) x k x 1  (n1+n2+n3) x k x 1  [d_out//2, d_out]
        # batch_size = feature_set.shape[0]
        # num_points = feature_set.shape[1]
        num_neigh = feature_set.shape[1]
        d = feature_set.shape[2]
        d_dis = g_dis.shape[2]

        # (n1+n2+n3) x k x d
        f_reshaped = feature_set.view(-1, num_neigh, d)
        # (n1+n2+n3) x k x 1
        f_dis_reshaped = f_dis.view(-1, num_neigh, d_dis) * 0.1
        # (n1+n2+n3) x k x 1
        g_dis_reshaped = g_dis.view(-1, num_neigh, d_dis)
        concat = torch.cat([g_dis_reshaped, f_dis_reshaped, f_reshaped], dim=2)

        # weight learning
        # (n1+n2+n3) x k x d
        if opid == 1:
            att_activation = self.dap_fc1(concat)
        else:
            att_activation = self.dap_fc2(concat)
        att_scores = self.softmax(att_activation)
        # dot product
        f_lc = f_reshaped * att_scores
        # sum
        # (n1+n2+n3) x d
        f_lc = torch.sum(f_lc, dim=1)
        # # b x n x d
        # f_lc = f_lc.view(batch_size, num_points, d)
        # # shared MLP
        # # b x n x d_out
        # f_lc = f_lc.permute([0, 2, 1])

        if opid == 1:
            f_lc = self.dap_conv1d1(f_lc)
        else:
            f_lc = self.dap_conv1d2(f_lc)
        # f_lc = f_lc.permute([0, 2, 1])
        # (n1+n2+n3) x d
        return f_lc

    def forward(self, xyz, feature, neigh_idx, lengths):
        # xyz:       (n1+n2+n3) x 3
        # feature:   (n1+n2+n3) x c
        # neigh_idx: (n1+n2+n3) x k
        # LPR
        # (n1+n2+n3) x k x 9, (n1+n2+n3) x k x 1, (n1+n2+n3) x 1
        local_rep, g_dis, lg_volume_ratio = self.local_polar_representation(xyz, neigh_idx, lengths)
        # (n1+n2+n3) x k x c
        local_rep = self.conv1(local_rep.permute([0, 2, 1])).permute([0, 2, 1])
        # (n1+n2+n3) x k x c
        # f_neighbours = gather_neighbour(feature, neigh_idx)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        f_neighbours = gather(torch.cat((feature, torch.zeros_like(feature[:1, :]).to(feature.device)), 0), neigh_idx)
        # (n1+n2+n3) x k x (c+c)
        f_concat = torch.cat([f_neighbours, local_rep], dim=2)
        # (n1+n2+n3) x k x 1
        f_dis = self.cal_feature_dis(feature, f_neighbours)
        # (n1+n2+n3) x d_out//2
        f_lc = self.dualdis_att_pool(f_concat, f_dis, g_dis, opid=1)

        # 2
        # (n1+n2+n3) x k x d_out//2
        local_rep = self.conv2(local_rep.permute([0, 2, 1])).permute([0, 2, 1])
        # (n1+n2+n3) x k x d_out//2
        # f_neighbours = gather_neighbour(f_lc, neigh_idx)
        # f_neighbours = f_lc[neigh_idx]
        f_neighbours = gather(torch.cat((f_lc, torch.zeros_like(feature[:1, :]).to(feature.device)), 0), neigh_idx)
        # (n1+n2+n3) x k x d_out
        f_concat = torch.cat([f_neighbours, local_rep], dim=2)
        # (n1+n2+n3) x k x 1
        f_dis = self.cal_feature_dis(f_lc, f_neighbours)
        # (n1+n2+n3) x d_out
        f_lc = self.dualdis_att_pool(f_concat, f_dis, g_dis, opid=2)

        return f_lc, lg_volume_ratio

    def cal_feature_dis(self, feature, f_neighbours):
        """
        Calculate the feature distance
        """
        # feature:     (n1+n2+n3) x c
        # f_neighbour: (n1+n2+n3) x k x c
        # b x n x c, b x n x k x c
        # batch_size, n, c = feature.shape[0], feature.shape[1], feature.shape[2]
        bn, c = feature.shape[0], feature.shape[1]
        # (n1+n2+n3) x k x c
        feature_tile = feature.view(bn, 1, c).repeat([1, f_neighbours.shape[1], 1])

        # (n1+n2+n3) x k x c
        feature_dist = feature_tile - f_neighbours

        feature_dist = torch.mean(torch.abs(feature_dist), dim=2).unsqueeze(dim=2)
        feature_dist = torch.exp(-feature_dist)
        # (n1+n2+n3) x k x 1
        return feature_dist


class SCF(nn.Module):
    def __init__(self, d_in, d_out):
        super(SCF, self).__init__()
        # self.conv1 = nn.Sequential(
        #     # nn.Conv1d(d_in, d_out//2, kernel_size=1, stride=1),
        #     nn.Linear(d_in, d_out // 2),
        #     # nn.InstanceNorm1d(d_out//2),
        #     nn.BatchNorm1d(d_out//2),
        #     nn.LeakyReLU(0.2)
        # )
        self.conv1 = MLP([d_in, d_out//2])
        # self.conv2 = nn.Sequential(
        #     nn.Linear(d_out, d_out*2),
        #     # nn.InstanceNorm1d(d_out*2),
        #     nn.BatchNorm1d(d_out*2),
        #     nn.LeakyReLU(0.2)
        # )
        self.conv2 = MLP([d_out, d_out*2])
        # self.conv3 = nn.Sequential(
        #     nn.Linear(d_in, d_out*2),
        #     # nn.InstanceNorm1d(d_out*2),
        #     nn.BatchNorm1d(d_out*2),
        #     nn.LeakyReLU(0.2)
        # )
        self.conv3 = MLP([d_in, d_out*2])
        # self.conv4 = nn.Sequential(
        #     nn.Linear(4, d_out*2),
        #     # nn.InstanceNorm1d(d_out*2),
        #     nn.BatchNorm1d(d_out*2),
        #     nn.LeakyReLU(0.2)
        # )
        self.conv4 = MLP([4, d_out*2])
        # self.conv5 = nn.Sequential(
        #     nn.Linear(d_out*4, d_out),
        #     # nn.InstanceNorm1d(d_out),
        #     nn.BatchNorm1d(d_out),
        #     nn.LeakyReLU(0.2)
        # )
        self.conv5 = MLP([d_out*4, d_out])
        self.local_context_learning = DDAP(d_out//2, d_out)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, xyz, feature, neigh_idx, lengths):
        """
        SCF
        """
        # (n1+n2+n3) x 3, (n1+n2+n3) x c, (n1+n2+n3) x k
        # Local Contextual Features
        # MLP 1
        # (n1+n2+n3) x c
        f_pc = self.conv1(feature)
        # Local Context Learning (LPR + DDAP)
        # (n1+n2+n3) x dout, (n1+n2+n3) x 1
        f_lc, lg_volume_ratio = self.local_context_learning(xyz, f_pc, neigh_idx, lengths)
        # MLP 2
        # f_lc = helper_tf_util.conv2d(f_lc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training, activation_fn=None)
        # (n1+n2+n3) x 2d_out
        f_lc = self.conv2(f_lc)
        # MLP Shotcut
        # shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training)
        # (n1+n2+n3) x 2d_out
        shortcut = self.conv3(feature)
        # Global Contextual Features
        # # (n1+n2+n3)  x 4
        f_gc = torch.cat([xyz, lg_volume_ratio], dim=1)
        # f_gc = helper_tf_util.conv2d(f_gc, d_out * 2, [1, 1], name + 'lg', [1, 1], 'VALID', activation_fn=None, bn=True, is_training=is_training)
        # # (n1+n2+n3) x 2d_out
        f_gc = self.conv4(f_gc)
        # b x n x d_out 我很确定这一行是他写错了
        # return self.leaky_relu(torch.cat([f_lc + shortcut, f_gc], dim=2))
        return self.conv5(torch.cat([f_lc + shortcut, f_gc], dim=1))


# class UpSample(nn.Module):
#     def __init__(self, in_channel, mlp):
#         super(UpSample, self).__init__()
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.InstanceNorm1d(out_channel))
#             last_channel = out_channel
#         self.leaky_relu = nn.LeakyReLU(0.2)
#
#     #                 多    少    多       少
#     def forward(self, xyz1, xyz2, features1, features2):
#         # xyz1 = xyz1.permute(0, 2, 1)
#         # xyz2 = xyz2.permute(0, 2, 1)
#         #
#         # points2 = points2.permute(0, 2, 1)
#         B, N, C = xyz1.shape
#         _, S, _ = xyz2.shape
#
#         if S == 1:
#             interpolated_points = features2.repeat(1, N, 1)
#         else:
#             dists = square_distance(xyz1, xyz2)
#             dists, idx = dists.sort(dim=-1)
#             dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
#
#             dist_recip = 1.0 / (dists + 1e-8)
#             norm = torch.sum(dist_recip, dim=2, keepdim=True)
#             weight = dist_recip / norm
#             interpolated_points = torch.sum(index_points(features2, idx) * weight.view(B, N, 3, 1), dim=2)
#
#         if features1 is not None:
#             # points1 = points1.permute(0, 2, 1)
#             new_points = torch.cat([features1, interpolated_points], dim=-1)
#         else:
#             new_points = interpolated_points
#
#         new_points = new_points.permute(0, 2, 1)
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points = self.leaky_relu(bn(conv(new_points)))
#         new_points = new_points.permute(0, 2, 1)
#         return new_points

class NearestUpsampleBlock(nn.Module):

    def __init__(self, layer_ind):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        return

    def forward(self, x, batch):
        return closest_pool(x, batch['upsamples'][self.layer_ind - 1])


class UpSample(nn.Module):
    def __init__(self, d_in, d_out):
        super(UpSample, self).__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(d_in, d_out),
        #     # nn.InstanceNorm1d(d_out),
        #     nn.BatchNorm1d(d_out),
        #     nn.LeakyReLU(0.2)
        # )
        self.mlp = MLP([d_in, d_out])

    def forward(self, features_will_upsample, upsample_idx, features_upsampled):
        return self.mlp(torch.cat([closest_pool(features_will_upsample, upsample_idx), features_upsampled], dim=1))


class SCFNet(nn.Module):
    def __init__(self):
        super(SCFNet, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(1, 8),
        #     # nn.InstanceNorm1d(8),
        #     nn.BatchNorm1d(8),
        #     nn.LeakyReLU()
        # )
        self.fc = MLP([1, 8])
        self.encoder = nn.ModuleList()
        self.encoder.append(SCF(8, 16))
        self.encoder.append(SCF(16, 64))
        self.encoder.append(SCF(64, 128))
        self.encoder.append(SCF(128, 256))
        self.encoder.append(SCF(256, 512))
        # self.last_encoder_layer = nn.Sequential(
        #     nn.Linear(512, 512),
        #     # nn.InstanceNorm1d(512),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(0.2)
        # )
        self.last_encoder_layer = MLP([512, 512])
        # self.npts = [2048, 1024, 512, 256, 128]
        #
        self.up1 = UpSample(768, 256)
        self.up2 = UpSample(384, 128)
        self.up3 = UpSample(192, 64)
        self.up4 = UpSample(80, 16)
        self.up5 = UpSample(32, 16)
        # #
        # self.clasify_layer = nn.Sequential(
        #     nn.Linear(16, 64),
        #     # nn.InstanceNorm1d(64),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(64, 32),
        #     # nn.InstanceNorm1d(32),
        #     nn.BatchNorm1d(32),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(0.5),
        #     nn.Linear(32, 1),
        #     nn.Sigmoid()
        # )
        self.pre_clasify_layer = MLP([16, 64, 32])
        self.clasify_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, get_features=False):
        # points:    [(n1+n2+n3) x 3, (n1'+n2'+n3') x 3, ...]
        # neighbors: [(n1+n2+n3) x k, (n1'+n2'+n3') x k, ...]
        # features:  (n1+n2+n3) x c
        # stack_lengths: [ndarr(n1, n2, n3), ndarr(n1', n2', n3'), ...]
        xyz = inputs["points"][0]
        features = self.fc(inputs["features"])
        lengths_list = [lens.cpu().numpy().tolist() for lens in inputs["stack_lengths"]]
        # bN0 x 16, bN1 x 16, bN2 x 64, bN3 x 128, bN4 x 256, bN5 x 512
        xyz_list, feature_list = [], []
        for i, en_layer in enumerate(self.encoder):
            # bn x k
            neigh_idx = inputs["neighbors"][i]
            # lengths = inputs["stack_lengths"][i]
            lengths = lengths_list[i]
            features = en_layer(xyz, features, neigh_idx, lengths)
            if i == 0:
                feature_list.append(features)
                xyz_list.append(xyz)
            xyz = inputs["points"][i+1]
            features = max_pool(features, inputs["pools"][i])
            feature_list.append(features)
            xyz_list.append(xyz)
        features = self.last_encoder_layer(features)
        features = self.up1(features, inputs["upsamples"][-2], feature_list[-2])
        features = self.up2(features, inputs["upsamples"][-3], feature_list[-3])
        features = self.up3(features, inputs["upsamples"][-4], feature_list[-4])
        features = self.up4(features, inputs["upsamples"][-5], feature_list[-5])
        features = self.up5(features, inputs["upsamples"][-6], feature_list[-6])
        # print(xyz.shape, features.shape)
        if get_features:
            all_features = features
        else:
            all_features = None
        features = self.pre_clasify_layer(features)
        features = self.clasify_layer(features)
        features = torch.clamp(features, min=0, max=1)
        features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
        result = []
        prefix, lengths = 0, inputs["stack_lengths"][0]
        for i in range(len(lengths)):
            result.append(features[prefix:prefix+lengths[i], :])
            prefix += lengths[i]
        if get_features:
            features = []
            prefix, lengths = 0, inputs["stack_lengths"][0]
            for i in range(len(lengths)):
                features.append(all_features[prefix:prefix + lengths[i], :])
                prefix += lengths[i]
            return result, features
        return result


class SCFMRFNet(nn.Module):
    def __init__(self, mini_rf_params_path):
        super(SCFMRFNet, self).__init__()
        self.mini_rf_params_path = mini_rf_params_path
        self.mini_rf = SCFNet()
        self.fc2 = MLP([17, 32])
        self.encoder = nn.ModuleList()
        self.encoder.append(SCF(32, 16))
        self.encoder.append(SCF(16, 64))
        self.encoder.append(SCF(64, 128))
        self.encoder.append(SCF(128, 256))
        self.encoder.append(SCF(256, 512))
        self.last_encoder_layer = MLP([512, 512])
        #
        self.up1 = UpSample(768, 256)
        self.up2 = UpSample(384, 128)
        self.up3 = UpSample(192, 64)
        self.up4 = UpSample(80, 16)
        self.up5 = UpSample(32, 16)

        self.pre_clasify_layer = MLP([16, 64, 32])
        self.clasify_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            # nn.Sigmoid()
        )

    def load(self):
        self.mini_rf.load_state_dict(torch.load(self.mini_rf_params_path))
        self.mini_rf.eval()

    def forward(self, inputs, sub_patch_inputs, sub_idx):
        # points:    [(n1+n2+n3) x 3, (n1'+n2'+n3') x 3, ...]
        # neighbors: [(n1+n2+n3) x k, (n1'+n2'+n3') x k, ...]
        # features:  (n1+n2+n3) x c
        # stack_lengths: [ndarr(n1, n2, n3), ndarr(n1', n2', n3'), ...]
        # 子块
        sub_feats = []
        if self.mini_rf.training:
            self.mini_rf.eval()
        with torch.no_grad():
            for sub_inp in sub_patch_inputs:
                _, features = self.mini_rf(sub_inp, True)
                sub_feats = sub_feats + features
        # print("two")
        ############################# two level ##################################
        xyz = inputs["points"][0]
        two_level_inp_feature, cnt = torch.zeros(xyz.shape[0], 16).to(xyz.device), torch.zeros(xyz.shape[0], 1).to(xyz.device)
        for i in range(len(sub_idx)):
            cnt[sub_idx[i]] += 1
            two_level_inp_feature[sub_idx[i]] += sub_feats[i]
        two_level_inp_feature /= cnt
        two_level_inp_feature = torch.where(torch.isnan(two_level_inp_feature), torch.zeros_like(two_level_inp_feature), two_level_inp_feature)
        two_level_inp_feature = torch.nn.functional.normalize(two_level_inp_feature, p=2, dim=1)

        two_level_inp_feature = torch.cat([two_level_inp_feature, torch.ones(xyz.shape[0], 1).to(xyz.device)], dim=1)
        features = self.fc2(two_level_inp_feature)
        lengths_list = [lens.cpu().numpy().tolist() for lens in inputs["stack_lengths"]]
        # bN0 x 16, bN1 x 16, bN2 x 64, bN3 x 128, bN4 x 256, bN5 x 512
        xyz_list, feature_list = [], []
        for i, en_layer in enumerate(self.encoder):
            # bn x k
            neigh_idx = inputs["neighbors"][i]
            # lengths = inputs["stack_lengths"][i]
            lengths = lengths_list[i]
            features = en_layer(xyz, features, neigh_idx, lengths)
            if i == 0:
                feature_list.append(features)
                xyz_list.append(xyz)
            xyz = inputs["points"][i + 1]
            features = max_pool(features, inputs["pools"][i])
            feature_list.append(features)
            xyz_list.append(xyz)
        features = self.last_encoder_layer(features)
        features = self.up1(features, inputs["upsamples"][-2], feature_list[-2])
        features = self.up2(features, inputs["upsamples"][-3], feature_list[-3])
        features = self.up3(features, inputs["upsamples"][-4], feature_list[-4])
        features = self.up4(features, inputs["upsamples"][-5], feature_list[-5])
        features = self.up5(features, inputs["upsamples"][-6], feature_list[-6])

        features = self.pre_clasify_layer(features)
        features = self.clasify_layer(features)
        # features = torch.clamp(features, min=0, max=1)
        features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
        result = []
        prefix, lengths = 0, inputs["stack_lengths"][0]
        for i in range(len(lengths)):
            result.append(features[prefix:prefix + lengths[i], :])
            prefix += lengths[i]
        return result


if __name__ == '__main__':
    # net = SCFNet()
    # device = torch.device("cuda:0")
    # net.to(device)
    #
    # xyz = torch.randn(3, 8192, 3).to(device)
    # features = torch.randn(3, 8192, 3).to(device)
    # y = net(xyz, features)
    # print(y.shape)
    pass