import argparse
import os
import time
import torch
import torch.nn.parallel

import criteria
import helper

from model import PENet_C1
from model import PENet_C2
from model import PENet_C4
from dataloaders.kitti_loader import KittiDepth
from dataloaders.pp_loader import PPDataLoader, load_calib
from metrics import AverageMeter, Result
from vis_utils import pp_save_depth_as_uint16png
from inverse_warp import Intrinsics, homography_from

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-n',
                    '--network-model',
                    type=str,
                    default="pe",
                    choices=["e", "pe"],
                    help='choose a model: enet or penet'
                    )
#TODO check what workers are
parser.add_argument('--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start-epoch-bias',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number bias(useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) +
                    ' (default: l2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-3,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-6,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder',
                    default='/home/maciej/git/igdc/pp_implementation_testing',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
# parser.add_argument('--data-folder-rgb',
#                     default='/data/dataset/kitti_raw',
#                     type=str,
#                     metavar='PATH',
#                     help='data folder rgb (default: none)')
parser.add_argument('--data-folder-save',
                    default='/data/dataset/kitti_depth/submit_test/',
                    type=str,
                    metavar='PATH',
                    help='data folder test results(default: none)')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd',
#                     choices=input_options,
#                     help='input: | '.join(input_options)
                    )
parser.add_argument('--val',
                    type=str,
                    default="select",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
# parser.add_argument('--rank-metric',
#                     type=str,
#                     default='rmse',
#                     choices=[m for m in dir(Result()) if not m.startswith('_')],
#                     help='metrics for which best result is saved')

parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('-f', '--freeze-backbone', action="store_true", default=False,
                    help='freeze parameters in backbone')
parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

#random cropping
parser.add_argument('--random-crop', action="store_true", default=False,
                    help='prohibit random cropping')
parser.add_argument('-he', '--random-crop-height', default=320, type=int, metavar='N',
                    help='random crop height')
parser.add_argument('-w', '--random-crop-width', default=1216, type=int, metavar='N',
                    help='random crop height')

#geometric encoding
parser.add_argument('-co', '--convolutional-layer-encoding', default="xyz", type=str,
                    choices=["std", "z", "uv", "xyz"],
                    help='information concatenated in encoder convolutional layers')

#dilated rate of DA-CSPN++
parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                    choices=[1, 2, 4],
                    help='CSPN++ dilation rate')
# run on KITTI wit PENET's original options
parser.add_argument('--kitti', action="store_true", default=False,
                    help='run-KITTI-with-original-options for debugging and comparing purposes')

# the logic is if --test is True than no training, if false do the training using the mode from the arg below
parser.add_argument('--train', action="store_true", default=False,
                    help='save result kitti test dataset for submission')

parser.add_argument(
    '-m',
    '--train-mode',
    type=str,
    default="sparse+photo",
    choices=["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
    help='dense | sparse | photo | sparse+photo | dense+photo')

args = parser.parse_args()
# if I use photos to calulate relative poses and use them for training
#args.use_pose if "photo" in args.train_mode
args.use_pose = True if "photo" in args.train_mode else False

if args.kitti:
    args.data_folder = '/home/maciej/git/igdc/kitti_depth/depth'
    args.val_h = 352
    args.val_w = 1216
else:
    args.val_h = 1472
    args.val_w = 2048

args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input)
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input

print(args)


cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

# from https://github.com/fangchangma/self-supervised-depth-completion/blob/master/main.py#L93
if args.use_pose:
    # hard-coded KITTI camera intrinsics
    K = load_calib(args)
    fu, fv = float(K[0, 0]), float(K[1, 1])
    cu, cv = float(K[0, 2]), float(K[1, 2])
    pp_intrinsics = Intrinsics(args.val_w, args.val_h, fu, fv, cu, cv)
    if cuda:
        pp_intrinsics = pp_intrinsics.cuda()

# define loss functions
if args.train:
    depth_criterion = criteria.MaskedMSELoss() if (
        args.criterion == 'l2') else criteria.MaskedL1Loss()
    photometric_criterion = criteria.PhotometricLoss()
    smoothness_criterion = criteria.SmoothnessLoss()

# TODO check what multi_batch is
#multi batch
multi_batch_size = 1
def iterate(mode, args, loader, model, optimizer, logger, epoch):
    actual_epoch = epoch - args.start_epoch + args.start_epoch_bias

    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, actual_epoch, args)
    else:
        model.eval()
        lr = 0

    torch.cuda.empty_cache()
    for i, batch_data in enumerate(loader):
        # print("=> iterating over step " + str(i) + " ... ", end='')
        print("\t=> iterating over step " + str(i) + " ... ")
        dstart = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }

        if mode == 'train':
            gt = batch_data['gt']
        data_time = time.time() - dstart

        pred = None
        start = None
        gpu_time = 0


        start = time.time()

        if(args.network_model == 'e'):
            st1_pred, st2_pred, pred = model(batch_data)
        # if network_model is 'pe'
        else:
            pred = model(batch_data)

        if(args.evaluate):
            gpu_time = time.time() - start
        #

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None

        # inter loss_param
        st1_loss, st2_loss, loss = 0, 0, 0
        w_st1, w_st2 = 0, 0
        round1, round2, round3 = 1, 3, None
        if(actual_epoch <= round1):
            w_st1, w_st2 = 0.2, 0.2
        elif(actual_epoch <= round2):
            w_st1, w_st2 = 0.05, 0.05
        else:
            w_st1, w_st2 = 0, 0

        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            if 'sparse' in args.train_mode:
                depth_loss = depth_criterion(pred, batch_data['d'])
                #TODO fifute out what 1e-3 does here
                mask = (batch_data['d'] < 1e-3).float()
            elif 'dense' in args.train_mode:
                depth_loss = depth_criterion(pred, gt)
                mask = (gt < 1e-3).float()

            # Loss 2: the self-supervised photometric loss
            if args.use_pose:
                # create multi-scale pyramids
                pred_array = helper.multiscale(pred)
                rgb_curr_array = helper.multiscale(batch_data['rgb'])
                rgb_near_array = helper.multiscale(batch_data['rgb_near'])
                if mask is not None:
                    mask_array = helper.multiscale(mask)
                num_scales = len(pred_array)

                # compute photometric loss at multiple scales
                for scale in range(len(pred_array)):
                    pred_ = pred_array[scale]
                    rgb_curr_ = rgb_curr_array[scale]
                    rgb_near_ = rgb_near_array[scale]
                    mask_ = None
                    if mask is not None:
                        mask_ = mask_array[scale]

                    # compute the corresponding intrinsic parameters
                    height_, width_ = pred_.size(2), pred_.size(3)
                    intrinsics_ = pp_intrinsics.scale(height_, width_)

                    # inverse warp from a nearby frame to the current frame
                    # TODO how they compute R and t_vec...
                    warped_ = homography_from(rgb_near_, pred_,
                                              batch_data['r_mat'],
                                              batch_data['t_vec'], intrinsics_)
                    photometric_loss += photometric_criterion(
                        rgb_curr_, warped_, mask_) * (2 ** (scale - num_scales))

                # Loss 3: the depth smoothness loss
                smooth_loss = smoothness_criterion(pred) if args.w2 > 0 else 0


            # TODO these loses to be fixed: for self-supervised and supervised separately
            if 'dense' in args.train_mode:
                loss = depth_loss
            # so if self-supervised stuff needed
            else:
                if args.network_model == 'e':
                    # TODO check what these losses are
                    st1_loss = depth_criterion(st1_pred, gt)
                    st2_loss = depth_criterion(st2_pred, gt)
                    loss = w_st1 * st1_loss + w_st2 * st2_loss + \
                           (1 - w_st1 - w_st2) * \
                           depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss # I integrated the loss from Ma's paper
                else:
                    # loss = depth_loss
                    loss = depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss # I integrated the loss from Ma's paper


            if i % multi_batch_size == 0:
                optimizer.zero_grad()
            loss.backward()

            if i % multi_batch_size == (multi_batch_size-1) or i==(len(loader)-1):
                optimizer.step()
            print("loss:", loss, " epoch:", epoch, " ", i, "/", len(loader))
#/home/maciej/Downloads/210601_target/CC_registered/5m/210702_5m_target_wall

        # if mode == "test_completion":
        #     str_i = str(i)
        #     path_i = str_i.zfill(10) + '.png'
        #     path = os.path.join(args.data_folder_save, path_i)
        #     vis_utils.save_depth_as_uint16png_upload(pred, path)

        # if(not args.evaluate):
        #     gpu_time = time.time() - start
        # # measure accuracy and record loss
        # with torch.no_grad():
        #     mini_batch_size = next(iter(batch_data.values())).size(0)
        #     result = Result()
        #     if mode != 'test_prediction' and mode != 'test_completion':
        #         result.evaluate(pred.data, gt.data, photometric_loss)
        #         [
        #             m.update(result, gpu_time, data_time, mini_batch_size)
        #             for m in meters
        #         ]

    #             if mode != 'train':
    #                 logger.conditional_print(mode, i, epoch, lr, len(loader),
    #                                  block_average_meter, average_meter)
    #             logger.conditional_save_img_comparison(mode, i, batch_data, pred,
    #                                                epoch)
    #             logger.conditional_save_pred(mode, i, pred, epoch)
    #
    # avg = logger.conditional_save_info(mode, average_meter, epoch)
    # is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    # if is_best and not (mode == "train"):
    #     logger.save_img_comparison_as_best(mode, epoch)
    # logger.conditional_summarize(mode, avg, is_best)

    # return avg, is_best


def main():
    global args

    ## get the model
    if args.evaluate != '':
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate), end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            print("Completed.")
        else:
            print("No model found at '{}'".format(args.evaluate))
            return
    else:
        checkpoint = None
        print("Starting with clear model (without pretrained weights).")

    print("=> creating model ... ", end='')
    model = None
    penet_accelerated = True

    # architecture
    if args.dilation_rate == 1:
        model = PENet_C1(args).to(device)
    elif args.dilation_rate == 2:
        model = PENet_C2(args).to(device)
    elif args.dilation_rate == 4:
        model = PENet_C4(args).to(device)

    if penet_accelerated:
        model.encoder3.requires_grad = False
        model.encoder5.requires_grad = False
        model.encoder7.requires_grad = False

    # params
    model_named_params = None
    model_bone_params = None
    model_new_params = None
    optimizer = None
    if checkpoint is not None:
        #print(checkpoint.keys())
        if (args.freeze_backbone == True):
            model.backbone.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'], strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    #TODO see their logger and helper
    logger = None

    ## get the data loader
    print("=> creating data loaders ... ", end='')
    if args.kitti:
        val_dataset = KittiDepth('val', args)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True)  # set batch size to be 1 for validation
        print("completed: val_loader size:{}".format(len(val_loader)))
    elif not args.train:
        val_dataset = PPDataLoader('test', args)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True)  # set batch size to be 1 for validation
        print("completed: val_loader size:{}".format(len(val_loader)))
    else:
        train_dataset = PPDataLoader('train', args)
        # the same as in PENET's main.py
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            sampler=None)
        print("completed: train_loader size:{}".format(len(train_loader)))

    if args.train:
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False
        iterate("val", args, val_loader, model, None, logger, args.start_epoch - 1)
        return

    ### optimiser
    print("=> creating optimiser ... ", end='')
    if (args.freeze_backbone == True):
        for p in model.backbone.parameters():
            p.requires_grad = False
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    elif (args.network_model == 'pe'):
        model_bone_params = [
            p for _, p in model.backbone.named_parameters() if p.requires_grad
        ]
        model_new_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        model_new_params = list(set(model_new_params) - set(model_bone_params))
        optimizer = torch.optim.Adam([{'params': model_bone_params, 'lr': args.lr / 10}, {'params': model_new_params}],
                                     lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    else:
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    print("completed.")

    if not args.cpu:
        model = torch.nn.DataParallel(model)

    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]



    ## make a prediction
    # mode = "test_completion"
    model.eval()
    torch.cuda.empty_cache()


    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        # print("=> iterating over epoch " + str(epoch) + " ... ", end='')
        print("=> iterating over epoch " + str(epoch) + " ... ")
        iterate("train", args, train_loader, model, optimizer, logger, epoch)  # train for one epoch

        # dstart = time.time()
        # batch_data = {key: val.to(device) for key, val in batch_data.items() if val is not None}
        #
        # if args.kitti:
        #     gt = batch_data['gt'] #if mode != 'test_prediction' and mode != 'test_completion' else None
        # data_time = time.time() - dstart
        #
        # pred = None
        # start = None
        # gpu_time = 0
        #
        # start = time.time()
        # pred = model(batch_data)
        #
        # depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None
        # if args.train: # train
        #
        #     # Loss 1: the direct depth supervision from ground truth label
        #     # mask=1 indicates that a pixel does not ground truth labels
        #     if 'sparse' in args.train_mode:
        #         depth_loss = depth_criterion(pred, batch_data['d'])
        #         #TODO fifute out what 1e-3 does here
        #         mask = (batch_data['d'] < 1e-3).float()
        #     elif 'dense' in args.train_mode:
        #         depth_loss = depth_criterion(pred, gt)
        #         mask = (gt < 1e-3).float()
        #
        #     # Loss 2: the self-supervised photometric loss
        #     if args.use_pose:
        #         # create multi-scale pyramids
        #         pred_array = helper.multiscale(pred)
        #         rgb_curr_array = helper.multiscale(batch_data['rgb'])
        #         rgb_near_array = helper.multiscale(batch_data['rgb_near'])
        #         if mask is not None:
        #             mask_array = helper.multiscale(mask)
        #         num_scales = len(pred_array)
        #
        #         # compute photometric loss at multiple scales
        #         for scale in range(len(pred_array)):
        #             pred_ = pred_array[scale]
        #             rgb_curr_ = rgb_curr_array[scale]
        #             rgb_near_ = rgb_near_array[scale]
        #             mask_ = None
        #             if mask is not None:
        #                 mask_ = mask_array[scale]
        #
        #             # compute the corresponding intrinsic parameters
        #             height_, width_ = pred_.size(2), pred_.size(3)
        #             intrinsics_ = pp_intrinsics.scale(height_, width_)
        #
        #             # inverse warp from a nearby frame to the current frame
        #             warped_ = homography_from(rgb_near_, pred_,
        #                                       batch_data['r_mat'],
        #                                       batch_data['t_vec'], intrinsics_)
        #             photometric_loss += photometric_criterion(
        #                 rgb_curr_, warped_, mask_) * (2 ** (scale - num_scales))
        #
        #         # Loss 3: the depth smoothness loss
        #         smooth_loss = smoothness_criterion(pred) if args.w2 > 0 else 0
        #
        #     # backprop
        #     loss = depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #
        # gpu_time = time.time() - start
        #
        #
        # ## evaluate somehow
        # # TODO see what Ma did https://github.com/fangchangma/self-supervised-depth-completion/blob/master/main.py#L93
        # with torch.no_grad():
        #     mini_batch_size = next(iter(batch_data.values())).size(0)
        #     result = Result()
        #
        #     pp_save_depth_as_uint16png(pred.data, "/home/maciej/git/igdc/pp_implementation_testing/pred_penet/" + str(i) + ".png")
        #     # print(batch_data["index"])
        #
        #     if args.kitti:
        #         result.evaluate(pred.data, gt.data, photometric_loss)
        #         [
        #             m.update(result, gpu_time, data_time, mini_batch_size) for m in meters
        #         ]
        #
        #
        #         print(
        #             'RMSE={blk_avg.rmse:.2f}({average.rmse:.2f}) '
        #             'MAE={blk_avg.mae:.2f}({average.mae:.2f}) '
        #             'iRMSE={blk_avg.irmse:.2f}({average.irmse:.2f}) '
        #             'iMAE={blk_avg.imae:.2f}({average.imae:.2f})\n\t'
        #             .format(
        #                 blk_avg=block_average_meter.average(),
        #                 average=average_meter.average()
        #             )
        #         )
        #
        #         # cummullator += block_average_meter.average().rmse
        #
        # if i == 15:
        #     return
    print("=> iterating over the main loop completed.")


if __name__ == '__main__':
    main()
