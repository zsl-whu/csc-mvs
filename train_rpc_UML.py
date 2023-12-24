import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import cv2
import time
import progressbar
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from models.mvsnet_rpc import MVSNet, mvsnet_loss
from datasets import get_loader
from utils_rpc import *
from losses.unsup_rpc_UML import *
import argparse

parser = argparse.ArgumentParser(description="RPCMVSNet args")


# dataset
parser.add_argument("--datapath", default=r'D:\zsl\WHU_RPC\open_dataset\open_dataset', type=str)
parser.add_argument("--trainlist", type=str)
parser.add_argument("--testlist", type=str)
parser.add_argument("--dataset_name", type=str, default="rpc", choices=["rpc", "dtu_yao", "whu"])
parser.add_argument('--batch_size', type=int, default=2, help='train batch size')
parser.add_argument('--numdepth', type=int, default=64, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')
parser.add_argument("--nviews", type=int, default=3)
# only for train and eval
parser.add_argument("--img_size", type=int, nargs='+', default=[384, 768])
parser.add_argument("--inverse_depth", action="store_true")
parser.add_argument('--fext', type=str, default='.png', help='Type of images.')
parser.add_argument('--resize_scale', type=float, default=1, help='output scale for depth and image (W and H)')

# training and val
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=35, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--scheduler', type=str, default="steplr", choices=["steplr", "cosinelr"])
parser.add_argument('--warmup', type=float, default=0.2, help='warmup epochs')
parser.add_argument('--milestones', type=float, nargs='+', default=[10, 12, 14, 20, 30], help='lr schedule') # [10, 20, 30]
parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay at every milestone')
parser.add_argument('--resume', default=None, type=str, help='path to the resume model')
parser.add_argument('--log_dir', default=r'.\ckpts_rpc\k\6', type=str, help='path to the log dir')

parser.add_argument('--seg_clusters', type=int, default=6, help='cluster centers for unsupervised co-segmentation')
parser.add_argument('--seg_types', type=str, default='both', help='seg methods for unsupervised co-segmentation')
parser.add_argument('--w_seg', type=float, default=1.0, help='initial weight for segments reprojection loss') # when not both for only nmf/eig
parser.add_argument('--w_nmf', type=float, default=1.0, help='initial weight for NMF') # when both for nmf
parser.add_argument('--w_eig', type=float, default=1.0, help='initial weight for eigenvection') # when both for eig(dsd)

parser.add_argument('--eval_freq', type=int, default=1, help='eval freq')
parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument("--val", default=False, action="store_true")
parser.add_argument("--sync_bn", action="store_true")
parser.add_argument("--blendedmvs_finetune", action="store_true")

# testing
parser.add_argument("--test", action="store_true")
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--num_view', type=int, default=3, help='num of view')
parser.add_argument('--max_h', type=int, default=384, help='testing max h')
parser.add_argument('--max_w', type=int, default=768, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')
parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')
parser.add_argument('--filter_method', type=str, default='gipuma', choices=["gipuma", "pcd", "dypcd"], help="filter method")
parser.add_argument('--display', action='store_true', help='display depth images and masks')
# pcd or dypcd
parser.add_argument('--conf', type=float, nargs='+', default=[0.1, 0.15, 0.9], help='prob confidence, for pcd and dypcd')
parser.add_argument('--thres_view', type=int, default=5, help='threshold of num view, only for pcd')
# dypcd
parser.add_argument('--dist_base', type=float, default=1 / 4)
parser.add_argument('--rel_diff_base', type=float, default=1 / 1300)
# gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='../fusibile/fusibile')
parser.add_argument('--prob_threshold', type=float, default='0.3')
parser.add_argument('--disp_threshold', type=float, default='0.25')
parser.add_argument('--num_consistent', type=float, default='3')

# visualization
parser.add_argument("--vis", action="store_true")
parser.add_argument('--depth_path', type=str)
parser.add_argument('--depth_img_save_dir', type=str, default="./")

# device and distributed
parser.add_argument("--distributed", default=True, action="store_true")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--local_rank", type=int, default=[0,1])
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

args = parser.parse_args()


class Model:
    def __init__(self, args):

        if args.vis:
            self.args = args
            return

        cudnn.benchmark = True

        # init_distributed_mode(args)

        self.args = args
        self.device = torch.device("cpu" if self.args.no_cuda or not torch.cuda.is_available() else "cuda")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = MVSNet(refine=False)

        self.log_var_seg = torch.tensor(self.args.w_seg).cuda()
        self.log_var_seg.requires_grad = True
        self.log_var_nmf = torch.tensor(self.args.w_nmf).cuda()
        self.log_var_nmf.requires_grad = True
        self.log_var_eig = torch.tensor(self.args.w_eig).cuda()
        self.log_var_eig.requires_grad = True

        if self.args.distributed:
            self.network = torch.nn.DataParallel(self.network, device_ids=self.args.local_rank).cuda()
        else:
            self.network.to(self.device)

        if self.args.distributed and self.args.sync_bn:
            self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)

        if not (self.args.val or self.args.test):
            if self.args.seg_types != 'both':
                self.parameters = ([p for p in self.network.parameters()] + [self.log_var_seg])
            else:
                self.parameters = ([p for p in self.network.parameters()] + [self.log_var_nmf] + [self.log_var_eig])
            self.optimizer = torch.optim.RMSprop(self.parameters, lr=args.lr, alpha=0.9, weight_decay=args.wd)
            # optim.Adam(self.network.parameters(), lr = args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
            self.train_loader, self.train_sampler = get_loader(args, args.datapath, args.trainlist, args.nviews, "train")

        if not self.args.test:
            self.loss_func = mvsnet_loss
            self.criterion_seg = UnSupSegLoss_rpc(args).cuda()
            self.val_loader, self.val_sampler = get_loader(args, args.datapath, args.testlist, args.nviews, "test")
            if is_main_process():
                self.writer = SummaryWriter(log_dir=args.log_dir, comment="Record network info")

        self.network_without_ddp = self.network
        if self.args.distributed:
            # self.network = DistributedDataParallel(self.network, device_ids=[self.args.local_rank])
            self.network_without_ddp = self.network.module

        if self.args.resume:
            checkpoint = torch.load(self.args.resume) #, map_location="cpu")
            if not (self.args.val or self.args.test or self.args.blendedmvs_finetune):
                self.args.start_epoch = checkpoint["epoch"] + 1
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                # self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.network_without_ddp.load_state_dict(checkpoint["model"])
        if not (self.args.val or self.args.test):
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, args.milestones,
                                                                     gamma=args.lr_decay,
                                                                     last_epoch=args.start_epoch - 1)

    def main(self):
        # if self.args.vis:
        #     self.visualization()
        #     return
        if self.args.val:
            self.validate()
            return
        # if self.args.test:
        #     self.test()
        #     return
        self.train()

    def train(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            # if self.args.distributed:
            #     self.train_sampler.set_epoch(epoch)
            self.lr_scheduler.step()
            if self.args.seg_types == 'both':
                nmf, eig = self.train_epoch(epoch)
                print(nmf, eig)
            else:
                seg = self.train_epoch(epoch)
                print(seg)
            # self.train_epoch(epoch)

            if is_main_process():
                torch.save({
                    'epoch': epoch,
                    'model': self.network_without_ddp.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict()},
                    "{}/model_{:0>6}.ckpt".format(self.args.log_dir, epoch))

            if (epoch % self.args.eval_freq == 0) or (epoch == self.args.epochs - 1):
                avg_scalars = self.validate(epoch)
                os.rename("{}/model_{:0>6}.ckpt".format(self.args.log_dir, epoch),
                          "{}/model_{:0>6}_{:.4f}.ckpt".format(self.args.log_dir, epoch, avg_scalars.avg_data["abs_depth_error"]))
            torch.cuda.empty_cache()

    def train_epoch(self, epoch):
        self.network.train()

        if is_main_process():
            pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ",
                        progressbar.Bar(), " ", progressbar.Timer(), ",", progressbar.ETA(), ",",
                        progressbar.Variable('LR', width=1), ",", progressbar.Variable('Loss', width=1), ",",
                        progressbar.Variable('Loss_pc', width=1), ",", progressbar.Variable('Loss_seg', width=1), ",",
                        progressbar.Variable('MAE', width=1), ",", progressbar.Variable('Th75', width=1), ",",
                        progressbar.Variable('Th25', width=1), ",", progressbar.Variable('Th3i', width=1)]
            pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(self.train_loader),
                                           prefix="Epoch {}/{}: ".format(epoch, self.args.epochs)).start()

        avg_scalars = DictAverageMeter()

        for batch, data in enumerate(self.train_loader):
            data = tocuda(data)

            outputs = self.network(data["imgs"], data["proj_matrices"], data["depth_values"])

            oriH = data["imgs"][0].shape[2]
            oriW = data["imgs"][0].shape[3]
            depth_est = outputs["depth"]
            depth_est = F.interpolate(depth_est.unsqueeze(1), size=[oriH, oriW]).squeeze(1)  # [B, H, W]
            model_loss = self.loss_func(depth_est, data["depth"], data["mask"])

            if self.args.seg_types == 'both':
                w_nmf_precision, w_eig_precision = torch.exp(-self.log_var_nmf), torch.exp(-self.log_var_eig)
                segment_loss_nmf, segment_loss_eig, ref_seg, view_segs = self.criterion_seg(data["imgs_seg"], data["proj_matrices"], depth_est, oriH, oriW)
                segment_loss_nmf, segment_loss_eig = torch.mean(segment_loss_nmf), torch.mean(segment_loss_eig)
                segment_loss = segment_loss_nmf * w_nmf_precision + segment_loss_eig * w_eig_precision + self.log_var_nmf + self.log_var_eig
            else:
                w_seg_precision = torch.exp(-self.log_var_seg)
                segment_loss, ref_seg, view_segs = self.criterion_seg(data["imgs_seg"], data["proj_matrices"], depth_est, oriH, oriW)
                segment_loss = torch.mean(segment_loss) * w_seg_precision + self.log_var_seg

            loss = model_loss + segment_loss
            loss.requires_grad_(True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            depth_gt = data["depth"]
            mask = data["mask"]
            depth_interval = data["depth_interval"]
            thres1 = Thres_metrics(depth_est, depth_gt, mask > 0.5, 7.5)
            thres2 = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2.5)
            thres3i = Inter_metrics(depth_est, depth_gt, depth_interval, mask > 0.5, 3)
            abs_depth_error = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)

            scalar_outputs = {"loss": loss,
                              "model_loss": model_loss,
                              "segment_loss": segment_loss,
                              "abs_depth_error": abs_depth_error,
                              "thres2mm_error": thres1,
                              "thres4mm_error": thres2,
                              "thres8mm_error": thres3i}

            scalar_outputs = tensor2float(scalar_outputs)
            # image_outputs = tensor2numpy(image_outputs)

            if is_main_process():
                avg_scalars.update(scalar_outputs)
                if batch >= len(self.train_loader) - 1:
                    save_scalars(self.writer, 'train_avg', avg_scalars.avg_data, epoch)
                if (epoch * len(self.train_loader) + batch) % self.args.summary_freq == 0:
                    save_scalars(self.writer, 'train', scalar_outputs, epoch * len(self.train_loader) + batch)
                    # save_images(self.writer, 'train', image_outputs, epoch * len(self.train_loader) + batch)

                pbar.update(batch, LR=self.optimizer.param_groups[0]['lr'],
                            Loss="{:.3f}|{:.3f}".format(scalar_outputs["loss"], avg_scalars.avg_data["loss"]),
                            Loss_pc="{:.3f}|{:.3f}".format(scalar_outputs["model_loss"], avg_scalars.avg_data["model_loss"]),
                            Loss_seg="{:.3f}|{:.3f}".format(scalar_outputs["segment_loss"], avg_scalars.avg_data["segment_loss"]),
                            MAE="{:.3f}|{:.3f}".format(scalar_outputs["abs_depth_error"], avg_scalars.avg_data["abs_depth_error"]),
                            Th75="{:.3f}|{:.3f}".format(scalar_outputs["thres2mm_error"], avg_scalars.avg_data["thres2mm_error"]),
                            Th25="{:.3f}|{:.3f}".format(scalar_outputs["thres4mm_error"], avg_scalars.avg_data["thres4mm_error"]),
                            Th3i="{:.3f}|{:.3f}".format(scalar_outputs["thres8mm_error"], avg_scalars.avg_data["thres8mm_error"]))

        if is_main_process():
            pbar.finish()
            # print('\n')

        if self.args.seg_types == 'both':
            return self.log_var_nmf, self.log_var_eig
        else:
            return self.log_var_seg

    @torch.no_grad()
    def validate(self, epoch=0):
        self.network.eval()

        if is_main_process():
            pwidgets = [progressbar.Percentage(), " ", progressbar.Counter(format='%(value)02d/%(max_value)d'), " ",
                        progressbar.Bar(), " ", progressbar.Timer(), ",", progressbar.ETA(), ",",
                        progressbar.Variable('Loss', width=1), ",", progressbar.Variable('MAE', width=1), ",",
                        progressbar.Variable('Th75', width=1), ",", progressbar.Variable('Th25', width=1), ",",
                        progressbar.Variable('Th3i', width=1)]
            pbar = progressbar.ProgressBar(widgets=pwidgets, max_value=len(self.val_loader), prefix="Val:").start()

        avg_scalars = DictAverageMeter()

        for batch, data in enumerate(self.val_loader):
            data = tocuda(data)

            outputs = self.network(data["imgs"], data["proj_matrices"], data["depth_values"])

            oriH = data["imgs"][0].shape[2]
            oriW = data["imgs"][0].shape[3]
            depth_est = outputs["depth"]
            depth_est = F.interpolate(depth_est.unsqueeze(1), size=[oriH, oriW]).squeeze(1)  # [B, H, W]
            # loss = self.criterion(data["imgs"], data["proj_matrices"], depth_est, oriH, oriW)
            loss = self.loss_func(depth_est, data["depth"], data["mask"])

            depth_gt = data["depth"]
            mask = data["mask"]
            depth_interval = data["depth_interval"]
            thres1 = Thres_metrics(depth_est, depth_gt, mask > 0.5, 7.5)
            thres2 = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2.5)
            thres3i = Inter_metrics(depth_est, depth_gt, depth_interval, mask > 0.5, 3)
            abs_depth_error = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0,10])

            scalar_outputs = {"loss": loss,
                              "abs_depth_error": abs_depth_error,
                              "thres2mm_error": thres1,
                              "thres4mm_error": thres2,
                              "thres8mm_error": thres3i}


            scalar_outputs = tensor2float(scalar_outputs)
            # image_outputs = tensor2numpy(image_outputs)

            if is_main_process():
                avg_scalars.update(scalar_outputs)
                if batch >= len(self.val_loader) - 1:
                    save_scalars(self.writer, 'test_avg', avg_scalars.avg_data, epoch)
                if (epoch * len(self.val_loader) + batch) % self.args.summary_freq == 0:
                    save_scalars(self.writer, 'test', scalar_outputs, epoch * len(self.val_loader) + batch)
                    # save_images(self.writer, 'test', image_outputs, epoch * len(self.val_loader) + batch)

                pbar.update(batch,
                            Loss="{:.3f}|{:.3f}".format(scalar_outputs["loss"], avg_scalars.avg_data["loss"]),
                            MAE="{:.3f}|{:.3f}".format(scalar_outputs["abs_depth_error"], avg_scalars.avg_data["abs_depth_error"]),
                            Th75="{:.3f}|{:.3f}".format(scalar_outputs["thres2mm_error"], avg_scalars.avg_data["thres2mm_error"]),
                            Th25="{:.3f}|{:.3f}".format(scalar_outputs["thres4mm_error"], avg_scalars.avg_data["thres4mm_error"]),
                            Th3i="{:.3f}|{:.3f}".format(scalar_outputs["thres8mm_error"], avg_scalars.avg_data["thres8mm_error"]))

        if is_main_process():
            pbar.finish()
            print('\n')

        return avg_scalars



if __name__ == '__main__':
    model = Model(args)
    print(args)
    # torch.set_num_threads(1)
    model.main()
