import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


def func_rpc(X, x2_num_20, x2_den_20, x3_num_20, x3_den_20):
    # x2_num_20, x2_den_20, x3_num_20, x3_den_20: [20]
    x1, x2, x3, x4 = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :], X[:, :, 3, :] # [B, N, H*W]
    batch, num_depth, num_pixel = X.shape[0], X.shape[1], X.shape[3]
    elements_add = torch.FloatTensor(batch, num_depth, 16, num_pixel).to(X.device) # [B, N, 16, H*W]

    elements_add[:, :, 0, :] = torch.mul(x2, x3)  # xy
    elements_add[:, :, 1, :] = torch.mul(x2, x4)  # xh
    elements_add[:, :, 2, :] = torch.mul(x3, x4)  # yh
    elements_add[:, :, 3, :] = torch.mul(x2, x2)  # xx
    elements_add[:, :, 4, :] = torch.mul(x3, x3)  # yy
    elements_add[:, :, 5, :] = torch.mul(x4, x4)  # hh
    elements_add[:, :, 6, :] = torch.mul(torch.mul(x2, x3), x4)  # xyh
    elements_add[:, :, 7, :] = torch.mul(torch.mul(x2, x2), x2)  # xxx
    elements_add[:, :, 8, :] = torch.mul(torch.mul(x2, x3), x3)  # xyy
    elements_add[:, :, 9, :] = torch.mul(torch.mul(x2, x4), x4)  # xhh
    elements_add[:, :, 10, :] = torch.mul(torch.mul(x2, x2), x3)  # xxy
    elements_add[:, :, 11, :] = torch.mul(torch.mul(x3, x3), x3)  # yyy
    elements_add[:, :, 12, :] = torch.mul(torch.mul(x3, x4), x4)  # yhh
    elements_add[:, :, 13, :] = torch.mul(torch.mul(x2, x2), x4)  # xxh
    elements_add[:, :, 14, :] = torch.mul(torch.mul(x3, x3), x4)  # yyh
    elements_add[:, :, 15, :] = torch.mul(torch.mul(x4, x4), x4)  # hhh

    # [B, N, 20, H*W]
    X_20 = torch.cat((X, elements_add), dim=2)
    x2_num_coef = x2_num_20.view(1, 1, 20, 1).repeat(batch, num_depth, 1, num_pixel)
    x2_den_coef = x2_den_20.view(1, 1, 20, 1).repeat(batch, num_depth, 1, num_pixel)
    x3_num_coef = x3_num_20.view(1, 1, 20, 1).repeat(batch, num_depth, 1, num_pixel)
    x3_den_coef = x3_den_20.view(1, 1, 20, 1).repeat(batch, num_depth, 1, num_pixel)

    # [B, N, H*W]
    x2_num, x2_den = torch.sum(torch.mul(x2_num_coef.double(), X_20.double()), dim=2), \
                     torch.sum(torch.mul(x2_den_coef.double(), X_20.double()), dim=2)
    x3_num, x3_den = torch.sum(torch.mul(x3_num_coef.double(), X_20.double()), dim=2), \
                     torch.sum(torch.mul(x3_den_coef.double(), X_20.double()), dim=2)
    x2_new, x3_new = torch.div(x2_num, x2_den), torch.div(x3_num, x3_den)

    # [B, N, 4, H*W]
    # ① (1, line_nom, samp_nom, heit_nom) -> (1, long_rpc, lati_rpc, heit_nom)
    # ② (1, long_rpc, lati_rpc, heit_nom) -> (1, line_ref, samp_ref, heit_nom)
    X_new = torch.ones_like(X).to(X.device)
    X_new[:, :, 0, :], X_new[:, :, 1, :], X_new[:, :, 2, :], X_new[:, :, 3, :] = x1, x2_new, x3_new, x4

    return X_new


def rpc_warping(src_fea, src_rpc, ref_rpc, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [1, 170]
    # ref_proj: [1, 170]
    # depth_values: [B, Ndepth]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1] # maybe are 1, 32
    num_depth = depth_values.shape[1]  # maybe is 192
    height, width = src_fea.shape[2], src_fea.shape[3]  # H, W
    height_ori, width_ori = height * 4, width * 4 # 384 768

    with torch.no_grad():
        src_rpc_170, ref_rpc_170 = src_rpc.squeeze(0), ref_rpc.squeeze(0)
        # get polynomial coefficients
        src_line_off, src_line_sca = src_rpc_170[0], src_rpc_170[5]
        src_samp_off, src_samp_sca = src_rpc_170[1], src_rpc_170[6]
        src_lati_off, src_lati_sca = src_rpc_170[2], src_rpc_170[7]
        src_long_off, src_long_sca = src_rpc_170[3], src_rpc_170[8]
        src_heit_off, src_heit_sca = src_rpc_170[4], src_rpc_170[9]

        ref_line_off, ref_line_sca = ref_rpc_170[0], ref_rpc_170[5]
        ref_samp_off, ref_samp_sca = ref_rpc_170[1], ref_rpc_170[6]
        ref_lati_off, ref_lati_sca = ref_rpc_170[2], ref_rpc_170[7]
        ref_long_off, ref_long_sca = ref_rpc_170[3], ref_rpc_170[8]
        ref_heit_off, ref_heit_sca = ref_rpc_170[4], ref_rpc_170[9]

        src_line_num, src_line_den = src_rpc_170[10: 30], src_rpc_170[30: 50]
        src_samp_num, src_samp_den = src_rpc_170[50: 70], src_rpc_170[70: 90]
        src_lati_num, src_lati_den = src_rpc_170[90: 110], src_rpc_170[110: 130]
        src_long_num, src_long_den = src_rpc_170[130:150], src_rpc_170[150: 170]

        ref_line_num, ref_line_den = ref_rpc_170[10: 30], ref_rpc_170[30: 50]
        ref_samp_num, ref_samp_den = ref_rpc_170[50: 70], ref_rpc_170[70: 90]
        ref_lati_num, ref_lati_den = ref_rpc_170[90: 110], ref_rpc_170[110: 130]
        ref_long_num, ref_long_den = ref_rpc_170[130:150], ref_rpc_170[150: 170]

        # generate y、x ∈ [96 192] and resize them to [384 768]
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        y, x = y * (height_ori / height), x * (width_ori / width)
        z = torch.ones_like(x)

        # do normalization for (samp_src, line_src, heit_hypo)
        # 像点坐标 (line, sample) 或 (r, c)
        # (row, col) -- (height, width) -- (y, x)
        # 所以 y -- line, x -- sample
        y_norm, x_norm = (y - ref_line_off) / ref_line_sca, (x - ref_samp_off) / ref_samp_sca # [H*W]
        h_norm = (depth_values - ref_heit_off) / ref_heit_sca # [B, N]

        # get Xs = (1, samp_n, line_n, heit_n)
        xyh_norm = torch.stack((z, y_norm, x_norm, z)) # [4, H*W]
        xyh_norm = xyh_norm.unsqueeze(0).repeat(num_depth, 1, 1) # [N, 4, H*W]
        xyh_norm[:, 3, :] = xyh_norm[:, 3, :] * h_norm.transpose(1, 0) # [N, H*W]
        Xs = xyh_norm.unsqueeze(0).repeat(batch, 1, 1, 1) # [B, N, 4, H*W]

        # Xs = (1, samp_n, line_n, heit_n) --func_rpc--> Xo = (1, long_rpc_n, lati_rpc_n, heit_n)
        # 从源影像某像点坐标出发，计算源影像上该点对应的地面坐标
        Xo = func_rpc(Xs, ref_long_num, ref_long_den, ref_lati_num, ref_lati_den) # [B, N, 4, H*W]

        # src de norm for Xo
        Xo[:, :, 1, :] = Xo[:, :, 1, :] * ref_long_sca + ref_long_off
        Xo[:, :, 2, :] = Xo[:, :, 2, :] * ref_lati_sca + ref_lati_off

        # ref norm for Xo
        Xo[:, :, 1, :] = (Xo[:, :, 1, :] - src_long_off) / src_long_sca
        Xo[:, :, 2, :] = (Xo[:, :, 2, :] - src_lati_off) / src_lati_sca

        # Xo = (1, lati_rpc, long_rpc, heit_n) --func_rpc--> Xref = (1, samp_ref, line_ref, heit_n)
        # 由源影像上真实的地面坐标，计算参考影像上对应的像点坐标
        X_ref = func_rpc(Xo, src_samp_num, src_samp_den, src_line_num, src_line_den) # [B, N, 4, H*W]

        # ref de norm for Xref to [384 768]
        X_ref[:, :, 1, :] = X_ref[:, :, 1, :] * src_samp_sca + src_samp_off
        X_ref[:, :, 2, :] = X_ref[:, :, 2, :] * src_line_sca + src_line_off

        # grid sample norm
        X_ref[:, :, 1, :] = X_ref[:, :, 1, :] / ((width_ori - 1) / 2) - 1
        X_ref[:, :, 2, :] = X_ref[:, :, 2, :] / ((height_ori - 1) / 2) - 1

        # oriwarp
        # xyh = torch.stack((z, x, y, z)).unsqueeze(0).unsqueeze(0).repeat(batch, num_depth, 1, 1)
        # xyh[:, :, 1, :] = xyh[:, :, 1, :] / ((width - 1) / 2) - 1
        # xyh[:, :, 2, :] = xyh[:, :, 2, :] / ((height - 1) / 2) - 1
        # proj_xy = xyh[:, :, 1:3, :].transpose(3, 2) # [B, N, H*W, 2]

        proj_xy = X_ref[:, :, 1:3, :].transpose(3, 2) # [B, N, H*W, 2]
        grid = proj_xy.contiguous()

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.contiguous().view(batch, channels, num_depth, height, width)

    return warped_src_fea


# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    return depth


if __name__ == "__main__":
    # some testing code, just IGNORE it
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2

    MVSDataset = find_dataset_def("dtu_yao")
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, 256)
    dataloader = DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4].cuda()
    proj_matrices = item["proj_matrices"].cuda()
    mask = item["mask"].cuda()
    depth = item["depth"].cuda()
    depth_values = item["depth_values"].cuda()

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    warped_imgs = homo_warping(src_imgs[0], src_projs[0], ref_proj, depth_values)

    cv2.imwrite('../tmp/ref.png', ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        cv2.imwrite('../tmp/tmp{}.png'.format(i), img_np[:, :, ::-1] * 255)


    # generate gt
    def tocpu(x):
        return x.detach().cpu().numpy().copy()


    ref_img = tocpu(ref_img)[0].transpose([1, 2, 0])
    src_imgs = [tocpu(x)[0].transpose([1, 2, 0]) for x in src_imgs]
    ref_proj_mat = tocpu(ref_proj)[0]
    src_proj_mats = [tocpu(x)[0] for x in src_projs]
    mask = tocpu(mask)[0]
    depth = tocpu(depth)[0]
    depth_values = tocpu(depth_values)[0]

    for i, D in enumerate(depth_values):
        height = ref_img.shape[0]
        width = ref_img.shape[1]
        xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
        print("yy", yy.max(), yy.min())
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])
        X = np.vstack((xx, yy, np.ones_like(xx)))
        # D = depth.reshape([-1])
        # print("X", "D", X.shape, D.shape)

        X = np.vstack((X * D, np.ones_like(xx)))
        X = np.matmul(np.linalg.inv(ref_proj_mat), X)
        X = np.matmul(src_proj_mats[0], X)
        X /= X[2]
        X = X[:2]

        yy = X[0].reshape([height, width]).astype(np.float32)
        xx = X[1].reshape([height, width]).astype(np.float32)

        warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
        # warped[mask[:, :] < 0.5] = 0

        cv2.imwrite('../tmp/tmp{}_gt.png'.format(i), warped[:, :, ::-1] * 255)
