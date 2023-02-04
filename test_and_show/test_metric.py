import numpy as np
import laspy
import torch


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
    print("#################### t1 ####################")
    laz = laspy.read("../OpenGF_Test/T1.laz")
    xyz, clz = laz.xyz, np.asarray(laz.classification)-1
    print("\n50 receptive field result:")
    pred50 = np.load("../OpenGF_Test_Result/t1-50A.npy")
    evaluate(pred50, clz)
    print("\n150 receptive field result:")
    pred150 = np.load("../OpenGF_Test_Result/t1-150F.npy")
    evaluate(pred150, clz)
    print("\n250 receptive field result:")
    pred250 = np.load("../OpenGF_Test_Result/t1-250F.npy")
    evaluate(pred250, clz)
    print("\nadd 50 to 150 result:")
    evaluate(0.4*pred50+0.6*pred150, clz)
    print("\nadd 50 to 250 result:")
    evaluate(0.4*pred50+0.6*pred250, clz)
    print("\nadd 150 to 250 result:")
    evaluate(0.4 * pred150 + 0.6 * pred250, clz)
    print("\n150 -> 250  result:")
    pred150in250 = np.load("../OpenGF_Test_Result/t1-150in250G.npy")
    evaluate(pred150in250, clz)
    print("\nadd 50 to 150->250")
    evaluate(0.4*pred50+0.6*pred150in250, clz)
    print("\n#################### t2 ####################")
    laz = laspy.read("../OpenGF_Test/t2_withoutnoise.laz")
    xyz, clz = laz.xyz, np.asarray(laz.classification) - 1
    print("\n50 receptive field result:")
    pred50 = np.load("../OpenGF_Test_Result/t2-50A.npy")
    evaluate(pred50, clz)
    print("\n150 receptive field result:")
    pred150 = np.load("../OpenGF_Test_Result/t2-150F.npy")
    evaluate(pred150, clz)
    print("\n250 receptive field result:")
    pred250 = np.load("../OpenGF_Test_Result/t2-250F.npy")
    evaluate(pred250, clz)
    print("\nadd 50 to 150 result:")
    evaluate(0.4 * pred50 + 0.6 * pred150, clz)
    print("\nadd 50 to 250 result:")
    evaluate(0.4 * pred50 + 0.6 * pred250, clz)
    print("\nadd 150 to 250 result:")
    evaluate(0.4 * pred150 + 0.6 * pred250, clz)
    print("\n150 -> 250  result:")
    pred150in250 = np.load("../OpenGF_Test_Result/t2-150in250G.npy")
    evaluate(pred150in250, clz)
    print("\nadd 50 to 150->250")
    evaluate(0.4 * pred50 + 0.6 * pred150in250, clz)
