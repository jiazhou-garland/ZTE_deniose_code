import os, torch, datetime, time
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloader.dataloader import CustomImageDataset
import torch.backends.cudnn as cudnn
from Denoise.Img_denoise.utils.metics import metics
from Denoise.Img_denoise.models.UNetplusplus import UNetplusplus
from utils.losses import Weighted_L1_loss
import albumentations as A
from models.NAF_model.archs.NAFNet_arch import NAFNet

if __name__ == '__main__':

    # training parameters-----------------------------------------------------------------------------------------------
    batch_size = 2
    epochs = 400
    early_stopping = 20
    # optimizer
    lr = 1e-4
    min_lr =1e-7
    weight_decay = 2e-4
    # lr_scheduler
    lr_steps = [(x+2) * 40 for x in range(400//50)]
    gamma = 0.5
    # data
    num_workers = 20
    num_workers_val = 0
    # log
    experiment = 'NAFNet'
    InteLog = 10
    # data augumentation
    crop_width = 256
    crop_height = 256
    # model
    img_channel = 4
    width = 36
    enc_blks = [2, 4, 8, 8]
    middle_blk_num = 8
    dec_blks = [2, 4, 8, 8]
    # Prepare saving model--------------------------------------------------------------------------------------------------
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_path = os.path.join('runs', experiment + current_time)
    if not os.path.exists(current_path): os.mkdir(current_path)
    model_dir = os.path.join(current_path, 'models')
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    log_dir = os.path.join(current_path, 'logs')
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)

    # Prepare model-----------------------------------------------------------------------------------------------------------
    device_num = 2
    device = torch.device("cuda:" + str(device_num))
    model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    model.to(device)

    # Load model------------------------------------------------------------------------------------------------------
    model_ckpt = "/home/zhoujiazhou/PycharmProjects/code/Denoise/Img_denoise/runs/NAFNet2488_50.87/models/best_model.pth"
    model.load_state_dict(torch.load(model_ckpt))

    # prepare data --------------------------------------------------------------------------------------------------
    def train_augmentation():
        return A.Compose([
            A.RandomCrop(width=crop_height, height=crop_width),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
        ])
    transform = train_augmentation()
    trainTxtPath = r'/home/zhoujiazhou/PycharmProjects/code/Denoise/Img_denoise/dataloader/train.txt'
    datasets = CustomImageDataset(trainTxtPath, transform = transform)
    train_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True,
                        pin_memory=torch.cuda.is_available(),
                        drop_last=True,
                        num_workers=num_workers)

    valTxtPath = r'/home/zhoujiazhou/PycharmProjects/code/Denoise/Img_denoise/dataloader/val.txt'
    datasets_val = CustomImageDataset(valTxtPath, transform = transform)
    valid_loader = DataLoader(datasets_val, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available(),
                            drop_last=False, num_workers=num_workers_val)
    STEPS_val = len(valid_loader)

    # Optimizer and lr_scheduler----------------------------------------------------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_adam = optim.Adam(params,
                                 lr=lr,
                                 weight_decay=weight_decay,
                                 betas=(0.9, 0.999))
    optimizer_AdamW = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    optimizer_SGD = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=2e-4)
    optimizer_RMSprop = optim.RMSprop(params, lr=lr, weight_decay=0.9, momentum=0.9)

    # warmup_epochs = 3
    # scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, warmup_epochs+40, eta_min=min_lr)
    # lr_scheduler = GradualWarmupScheduler(optimizer_adam, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_adam, T_0=3, T_mult=2,
                                                                        eta_min=min_lr,last_epoch=-1)
    MultiStepLR_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_SGD, lr_steps, gamma)
    # Loss function-----------------------------------------------------------------------------------------------------------
    mse_criterion =  torch.nn.MSELoss(reduction='sum')
    l1_criterion = torch.nn.L1Loss(reduction='sum')
    weighted_l1_criterion = Weighted_L1_loss()
    # train the model --------------------------------------------------------------------------------------------------
    def train(dataloader, model, optimizer, epoch):
        model.train()  # use batch normalization and drop out

        size = len(dataloader.dataset)
        running_loss, running_loss_content, running_loss_fft =  0.0, 0.0, 0.0
        end_batch = time.time()

        for step, (batch_x, batch_y) in enumerate(dataloader):

            batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_x = batch_x.squeeze(dim = 1)
            batch_y = batch_y.squeeze(dim = 1)

            pred_img = model(batch_x)

            # loss_content = mse_criterion(pred_img, batch_y)
            # label_fft = torch.fft.fft2(batch_y, dim=(-2, -1))
            # pred_fft = torch.fft.fft2(pred_img, dim=(-2, -1))
            loss = weighted_l1_criterion(pred_img, batch_y)

            # loss = 0.9 * loss_content + 0.1 * loss_ff

            # compute gradient and do optimizing step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # running_loss_content += loss_content.item()
            # running_loss_fft += loss_fft.item()

            if step % InteLog == 0:

                writer.add_scalar('training loss', running_loss / 10, epoch * size + step)
                writer.add_scalar('training loss_content', running_loss_content / 10, epoch * size + step)
                writer.add_scalar('training loss_fft', running_loss_fft / 10, epoch * size + step)

                running_loss = 0.0
                running_loss_content = 0.0
                running_loss_fft = 0.0

            current = step * len(batch_x)
            batch_time = time.time() - end_batch
            # end_batch = time.time()
            # print({f"batch time: {batch_time:.3f} loss: {loss.item():>7f} loss_content: {loss_content.item():>7f} loss_fft:{loss_fft.item():>7f}"
            #        f"[current{current:>5d}/batch_size{size:>5d}]"})
            print({f"batch time: {batch_time:.3f} loss: {loss.item():>7f}"
                   f"[current{current:>5d}/batch_size{size:>5d}]"})

        # test the model ---------------------------------------------------------------------------------------------------

    def validate(dataloader, model, epoch):
        model.eval()  # switch to evaluate mode

        size = len(dataloader.dataset)
        mean_loss, mean_loss_fft, mean_loss_content, mean_psnr, mean_ssim = 0.0, 0.0, 0.0, 0.0, 0.0
        end = time.time()

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.FloatTensor)
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                batch_x = batch_x.squeeze(dim=1)
                batch_y = batch_y.squeeze(dim=1)

                pred_img = model(batch_x)
                # print(pred_img.size())

                # loss_content = mse_criterion(pred_img, batch_y)
                # label_fft = torch.fft.fft2(batch_y, dim=(-2, -1))
                # pred_fft = torch.fft.fft2(pred_img, dim=(-2, -1))
                loss = weighted_l1_criterion(pred_img, batch_y)

                psnr, ssim = metics(pred_img, batch_y, crop_height, crop_width)

                mean_loss += loss.item()
                # mean_loss_fft += loss_fft.item()
                # mean_loss_content = loss_content.item()
                mean_psnr += psnr.item()
                mean_ssim += ssim.item()

                print({f"current_loss: {loss.item():>7f} "
                       f"current_ssim: {ssim.item():>7f} current_mean_psnr: {psnr.item():>7f}"})

        mean_loss /= size
        mean_loss_fft /= size
        mean_loss_content /= size
        mean_psnr /= size
        mean_ssim /= size

        writer.add_scalar('val_loss', mean_loss, epoch)
        writer.add_scalar('val_loss_fft', mean_loss_fft, epoch)
        writer.add_scalar('val_loss_content', mean_loss_content, epoch)
        writer.add_scalar('val_psnr', mean_psnr, epoch)
        writer.add_scalar('val_ssim', mean_ssim, epoch)

        val_time = time.time() - end
        score = (0.8 * (mean_psnr-30) / 30 + 0.2 * (mean_ssim-0.8) / 0.2 ) * 100
        print(f"Test Error: \n Mean_Loss: {mean_loss:>8f} Total val time: {val_time:.3f}\n"
              f"mean_ssim: {mean_ssim:>7f} mean_psnr: {mean_psnr:>7f} score: {score:>3f}")

        return mean_psnr, mean_ssim


    # train and validate in epochs--------------------------------------------------------------------------------------

    cudnn.benchmark = True

    best_loss = np.Inf
    best_score = 0.0
    best_spnr = 0.0
    best_ssim = 0.0

    trigger = 0
    end_epoch = time.time()

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, optimizer_adam, epoch=t)
        mean_psnr, mean_ssim = validate(valid_loader, model, epoch=t)
        writer.add_scalar('lr', optimizer_adam.param_groups[0]['lr'], t)
        lr_scheduler.step()
        epoch_time = time.time() - end_epoch
        print(f"the {t + 1}th epoch_time: {epoch_time:.3f}")
        end_epoch = time.time()
        trigger += 1

        if mean_psnr >= best_spnr and mean_ssim >= best_ssim:
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            best_spnr = mean_psnr
            best_ssim = mean_ssim
            print('=> saved best model to ' + os.path.join(model_dir, 'best_model.pth'))
            trigger = 0

        # if trigger >= early_stopping:
        #     print("=> early stopping at the " + str(t+1) + "th epochs")
        #     break

        # torch.cuda.empty_cache()

    print("All is Done! The best model's score are " + str(best_spnr) + " and " + str(best_ssim) + ".")