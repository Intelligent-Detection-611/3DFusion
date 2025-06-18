import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader.nifti_fusion_data import NiftiFusionDataset
from models.common_3D import gradient3d as gradient, clamp
from models.fusion_model_3D import PIAFusion

def init_seeds(seed=0, cuda=False):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion 3D NIfTI')
    parser.add_argument('--dataset_path', metavar='DIR', default='datasets/nifti_train',
                        help='path to dataset folder. subfolders: vi/ ir/')
    parser.add_argument('--save_path', default='save_model')
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--loss_weight', default='[7, 50]', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cuda', default=True, type=bool)
    args = parser.parse_args()

    init_seeds(args.seed, args.cuda)

    # 1. 数据集加载
    train_dataset = NiftiFusionDataset(args.dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 2. 网络
    model = PIAFusion()
    if args.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    t2, t3 = eval(args.loss_weight)  # 损失权重

    for epoch in range(args.start_epoch, args.epochs):
        # 动态调整学习率
        if epoch < args.epochs // 2:
            lr = args.lr
        else:
            lr = args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        train_tqdm = tqdm(train_loader, total=len(train_loader))

        for vi_image, ir_image in train_tqdm:
            if args.cuda:
                vi_image = vi_image.cuda()
                ir_image = ir_image.cuda()
            optimizer.zero_grad()
            fused_image = model(vi_image, ir_image)
            fused_image = clamp(fused_image)

            # 损失函数
            loss_aux = F.l1_loss(fused_image, torch.max(vi_image, ir_image))
            gradinet_loss = F.l1_loss(gradient(fused_image), torch.max(gradient(ir_image), gradient(vi_image)))
            loss = t2 * loss_aux + t3 * gradinet_loss

            train_tqdm.set_postfix(epoch=epoch, loss_aux=(t2 * loss_aux).item(),
                                   gradinet_loss=(t3 * gradinet_loss).item(),
                                   loss_total=loss.item())
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f'{args.save_path}/fusion_model_epoch_{epoch}.pth')
