import argparse
import os
import time
import torch
import torch.nn.parallel

from model import PENet_C1
from model import PENet_C2
from model import PENet_C4
from dataloaders.kitti_loader import KittiDepth
from dataloaders.pp_loader import PPDataLoader
from metrics import AverageMeter, Result
from vis_utils import pp_save_depth_as_uint16png

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
# parser.add_argument('-c',
#                     '--criterion',
#                     metavar='LOSS',
#                     default='l2',
#                     choices=criteria.loss_names,
#                     help='loss function: | '.join(criteria.loss_names) +
#                     ' (default: l2)')
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
parser.add_argument('--test', action="store_true", default=True,
                    help='save result kitti test dataset for submission')
parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

#random cropping
parser.add_argument('--not-random-crop', action="store_true", default=False,
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


args = parser.parse_args()
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



def main():
    global args

    ## get the model
    if args.evaluate == '':
        args.evaluate = "/home/maciej/git/igdc/PENet_ICRA2021/pretrained/pe.pth.tar"
    if os.path.isfile(args.evaluate):
        print("=> loading checkpoint '{}' ... ".format(args.evaluate), end='')
        checkpoint = torch.load(args.evaluate, map_location=device)
        print("Completed.")
    else:
        print("No model found at '{}'".format(args.evaluate))
        return
    is_eval = True

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
    if not args.cpu:
        model = torch.nn.DataParallel(model)

    #TODO see their logger and helper


    ## get the data loader
    if args.kitti:
        val_dataset = KittiDepth('val', args)
    elif args.test:
        val_dataset = PPDataLoader('test', args)
    else:
        val_dataset = PPDataLoader('val', args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    if is_eval:
        for p in model.parameters():
            p.requires_grad = False



    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]



    ## make a prediction
    mode = "test_completion"
    model.eval()
    lr = 0
    torch.cuda.empty_cache()

    # cummullator = 0

    for i, batch_data in enumerate(val_loader):
        dstart = time.time()
        batch_data = {key: val.to(device) for key, val in batch_data.items() if val is not None}

        if args.kitti:
            gt = batch_data['gt'] #if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - dstart

        pred = None
        start = None
        gpu_time = 0

        start = time.time()
        pred = model(batch_data)
        gpu_time = time.time() - start

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None




        ## evaluate somehow
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()

            pp_save_depth_as_uint16png(pred.data, "/home/maciej/git/igdc/pp_implementation_testing/pred_penet/" + str(i) + ".png")
            # print(batch_data["index"])

            if args.kitti:
                result.evaluate(pred.data, gt.data, photometric_loss)
                [
                    m.update(result, gpu_time, data_time, mini_batch_size) for m in meters
                ]


                print(
                    'RMSE={blk_avg.rmse:.2f}({average.rmse:.2f}) '
                    'MAE={blk_avg.mae:.2f}({average.mae:.2f}) '
                    'iRMSE={blk_avg.irmse:.2f}({average.irmse:.2f}) '
                    'iMAE={blk_avg.imae:.2f}({average.imae:.2f})\n\t'
                    .format(
                        blk_avg=block_average_meter.average(),
                        average=average_meter.average()
                    )
                )

                # cummullator += block_average_meter.average().rmse

        if i == 15:
            return


if __name__ == '__main__':
    main()