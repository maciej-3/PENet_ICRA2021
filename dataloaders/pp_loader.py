import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
from . import transforms
from .pp_pose_estimator import get_pose_pnp
import CoordConv


input_options = ['d', 'rgb', 'rgbd', 'g', 'gd'] # lidar depth, rgb_img, rgb+depth, gray, ground truth

# TODO implement dynamic (not hardcoded) calibration loading
def load_calib(args):

    if args.kitti:
        # EVEN THOUGH THERES PROJECTED VELODYNE DEPTH MAPS AND GT, PENET STILL USES THEIR K
        # in model.py -> class ENet -> forward(self, input) -> K = input['K']
        # and it affects convolutional layer encoding == "xyz" and sth they call geofeatures
        calib = open("dataloaders/calib_cam_to_cam.txt", "r")
        lines = calib.readlines()
        P_rect_line = lines[25]

        Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
        Proj = np.reshape(np.array([float(p) for p in Proj_str]),
                          (3, 4)).astype(np.float32)
        K = Proj[:3, :3]  # camera matrix

        # note: we will take the center crop of the images during augmentation
        # that changes the optical centers, but not focal lengths
        # K[0, 2] = K[0, 2] - 13  # from width = 1242 to 1216, with a 13-pixel cut on both sides
        # K[1, 2] = K[1, 2] - 11.5  # from width = 375 to 352, with a 11.5-pixel cut on both sides
        K[0, 2] = K[0, 2] - 13;
        K[1, 2] = K[1, 2] - 11.5;
    else:
        K = np.array([[2270.176025390625, 0.0, 977.3906152454074],
                [0.0, 2317.939208984375, 809.0955059066837],
                [0.0, 0.0, 1.0]])
        # H = np.array([[1.0, 0.0, 0.0, 0.05],
        #             [0.0, 1.0, 0.0, 0.05],
        #             [0.0, 0.0, 1.0, 0.05]])
        # the hard-coded numbers here are the size of the undistorted images
        # since I do center crop, I need to (undist_dim - desired_dim)/2
        K[0, 2] = K[0, 2] - (2060-args.val_w)/2;
        K[1, 2] = K[1, 2] - (1529-args.val_h)/2;

    return K



def get_paths_and_transform(split, args):
    assert (args.use_d or args.use_rgb or args.use_g), 'no proper input selected'

    paths_rgb = None
    paths_d = None
    glob_gt = None


    if args.kitti:
        # transform = no_transform
        transform = val_transform
        glob_d = os.path.join(
            args.data_folder,
            "data_depth_selection/val_selection_cropped/velodyne_raw/*.png")
        glob_gt = os.path.join(
            args.data_folder,
            "data_depth_selection/val_selection_cropped/groundtruth_depth/*.png"
        )

        def get_rgb_paths(p):
            return p.replace("groundtruth_depth", "image")
    elif split == 'test':
        # transform = no_transform
        transform = pp_val_transform
        glob_d = os.path.join(
            args.data_folder,
            "d/*.png")
        glob_rgb = os.path.join(
            args.data_folder,
            "rgb/*.png")
        # glob_gt = os.path.join(
        #     args.data_folder,
        #     "data_depth_selection/val_selection_cropped/groundtruth_depth/*.png"
        # )

        # def get_rgb_paths(p):
        #     return p.replace("d", "rgb")
    elif split == 'train':
        transform = pp_train_transform
        glob_d = os.path.join(args.data_folder,
                "d/*.png")
        glob_gt = os.path.join(args.data_folder,
                "gt/*.png")
        glob_rgb = os.path.join(
            args.data_folder,
            "rgb/*.png")

    if glob_gt is not None:
        paths_d = sorted(glob.glob(glob_d))
        paths_gt = sorted(glob.glob(glob_gt))
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]

    #TODO figure out how to handle d and gt here
    elif glob_d is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d))
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)

    #TODO write code that can handle only rgb and d data without gt so that it works later
    # else:
    #     # test only has d or rgb
    #     paths_rgb = sorted(glob.glob(glob_rgb))
    #     paths_gt = [None] * len(paths_rgb)
    #     if split == "test_prediction":
    #         paths_d = [None] * len(
    #             paths_rgb)  # test_prediction has no sparse depth
    #     else:
    #         paths_d = sorted(glob.glob(glob_d))


    if glob_gt is not None:
        if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
            raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
        # if len(paths_d) == 0 and len(paths_rgb) == 0:
        #     raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
        if len(paths_d) == 0 and args.use_d:
            raise (RuntimeError("Requested sparse depth but none was found"))
        if len(paths_rgb) == 0 and args.use_rgb:
            raise (RuntimeError("Requested rgb images but none was found"))
        if len(paths_rgb) == 0 and args.use_g:
            raise (RuntimeError("Requested gray images but no rgb was found"))
        if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
            print(len(paths_rgb), len(paths_d), len(paths_gt))
            # for i in range(999):
            #    print("#####")
            #    print(paths_rgb[i])
            #    print(paths_d[i])
            #    print(paths_gt[i])
            # raise (RuntimeError("Produced different sizes for datasets"))
    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    return paths, transform



def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth

def pp_depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 256, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename) # this can be, though if i wanna make sure
    # that nothing below 1m is stored than this number should be 1000
    pp_scale = 655.35
    depth = depth_png.astype(np.float) / pp_scale
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


def train_transform(rgb, d, gt, position, args):
    pass

# TODO check what this function does
def val_transform(rgb, d, gt, position, args):
    oheight = args.val_h
    owidth = args.val_w

    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if d is not None:
        sparse = transform(d)
    if gt is not None:
        target = transform(gt)
    if position is not None:
        position = transform(position)
    target = None
    return rgb, sparse, target, position

def pp_train_transform(rgb, d, gt, position, args):
    oheight = args.val_h
    owidth = args.val_w

    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    transforms_list = [
        # transforms.Rotate(angle),
        # transforms.Resize(s),
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)
    ]

    # if small_training == True:
    # transforms_list.append(transforms.RandomCrop((rheight, rwidth)))

    transform_geometric = transforms.Compose(transforms_list)

    if d is not None:
        d = transform_geometric(d)
    print(gt)
    gt = transform_geometric(gt)
    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter),
                                       1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)
    # d = drop_depth_measurements(d, 0.9)

    if position is not None:
        bottom_crop_only = transforms.Compose([transforms.BottomCrop((oheight, owidth))])
        position = bottom_crop_only(position)

    # random crop
    #if small_training == True:
    if args.random_crop == True:
        h = oheight
        w = owidth
        rheight = args.random_crop_height
        rwidth = args.random_crop_width
        # randomlize
        i = np.random.randint(0, h - rheight + 1)
        j = np.random.randint(0, w - rwidth + 1)

        if rgb is not None:
            if rgb.ndim == 3:
                rgb = rgb[i:i + rheight, j:j + rwidth, :]
            elif rgb.ndim == 2:
                rgb = rgb[i:i + rheight, j:j + rwidth]

        if d is not None:
            if d.ndim == 3:
                d = d[i:i + rheight, j:j + rwidth, :]
            elif d.ndim == 2:
                d = d[i:i + rheight, j:j + rwidth]

        if gt is not None:
            if gt.ndim == 3:
                gt = gt[i:i + rheight, j:j + rwidth, :]
            elif gt.ndim == 2:
                gt = gt[i:i + rheight, j:j + rwidth]

        if position is not None:
            if position.ndim == 3:
                position = position[i:i + rheight, j:j + rwidth, :]
            elif position.ndim == 2:
                position = position[i:i + rheight, j:j + rwidth]

    return rgb, d, gt, position



def pp_val_transform(rgb, d, gt, rgb_near, position, args):
    oheight = args.val_h
    owidth = args.val_w

    transform = transforms.Compose([
        transforms.CenterCrop((oheight, owidth)),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if d is not None:
        d = transform(d)
    if gt is not None:
        rgb_near = transform(rgb_near)
    if position is not None:
        position = transform(position)
    gt = None
    return rgb, d, gt, rgb_near, position


def no_transform(rgb, sparse, target, rgb_near, position, args):
    return rgb, sparse, target, rgb_near, position


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


def handle_gray(rgb, args):
    if rgb is None:
        return None, None
    if not args.use_g:
        return rgb, None
    else:
        img = np.array(Image.fromarray(rgb).convert('L'))
        img = np.expand_dims(img, -1)
        if not args.use_rgb:
            rgb_ret = None
        else:
            rgb_ret = rgb
        return rgb_ret, img

# TODO check this function, night be interesting for self-supervision
def get_rgb_near(path, args):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        # new_filename = os.path.join(head, '%010d.png' % new_id)
        new_filename = os.path.join(head, '%d.png' % new_id)
        return new_filename

    head, number = extract_frame_id(path)
    count = 0
    max_frame_diff = 5
    candidates = [
        i - max_frame_diff for i in range(max_frame_diff * 2 + 1)
        if i - max_frame_diff != 0
    ]
    while True:
        # TODO naming of my rgbs and d's must be 1, 2, 3, 4, ... not 3, 16, 22 ...
        # TODO from what range to randomly choose must be tuned cos my consecutive iages will be quite close and the depth can be better calulated from the further rgbs than from closer ones
        random_offset = choice(candidates)
        path_near = get_nearby_filename(path, number + random_offset)
        if os.path.exists(path_near):
            break
        assert count < 20, "cannot find a nearby frame in 20 trials for {}".format(path_near)
        count += 1

    return rgb_read(path_near)


class PPDataLoader(data.Dataset):
    """A data loader for Maciej's PhD project
    """

    def __init__(self, split, args):
        self.args = args
        self.split = split
        paths, transform = get_paths_and_transform(split, args)
        self.paths = paths
        self.transform = transform
        self.K = load_calib(args)
        self.threshold_translation = 0.1

    def __getraw__(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if \
            (self.paths['rgb'][index] is not None and (self.args.use_rgb or self.args.use_g)) else None
        if self.args.kitti:
            d = depth_read(self.paths['d'][index]) if \
                (self.paths['d'][index] is not None and self.args.use_d) else None
            gt = depth_read(self.paths['gt'][index]) if \
                self.paths['gt'][index] is not None else None
        else:
            d = pp_depth_read(self.paths['d'][index]) if \
                (self.paths['d'][index] is not None and self.args.use_d) else None
            gt = pp_depth_read(self.paths['gt'][index]) if \
                self.paths['gt'][index] is not None else None
        rgb_near = get_rgb_near(self.paths['rgb'][index], self.args) if \
            self.split == 'train' and self.args.use_pose else None

        return rgb, d, gt, rgb_near

    def __getitem__(self, index):
        rgb, d, gt, rgb_near = self.__getraw__(index)
        position = CoordConv.AddCoordsNp(self.args.val_h, self.args.val_w)
        position = position.call()
        #TODO incorporate the rgb_near someh
        #rgb, d, gt, rgb_near, position = self.transform(rgb, d, gt, position, self.args)
        rgb, d, gt, position = self.transform(rgb, d, gt, position, self.args)

        # [R|t] between two rgb's
        r_mat, t_vec = None, None
        if self.split == 'train' and self.args.use_pose:
            success, r_vec, t_vec = get_pose_pnp(rgb, rgb_near, d, self.K)
            # discard if translation is too small
            success = success and LA.norm(t_vec) > self.threshold_translation
            if success:
                r_mat, _ = cv2.Rodrigues(r_vec)
            else:
                # return the same image and no motion when PnP fails
                rgb_near = rgb
                t_vec = np.zeros((3, 1))
                r_mat = np.eye(3)

        rgb, gray = handle_gray(rgb, self.args)
        # candidates = {"rgb": rgb, "d": sparse, "gt": target, \
        #              "g": gray, "r_mat": r_mat, "t_vec": t_vec, "rgb_near": rgb_near}
        candidates = {'rgb': rgb, 'd': d, 'gt': gt, 'rgb_near': rgb_near,
                      "r_mat": r_mat,
                      "t_vec": t_vec,
                      'g': gray,
                      'position': position,
                      'K': self.K, #"index": index
                      }

        items = {
            key: to_float_tensor(val)
            # key: val # that does not work
        for key, val in candidates.items() if val is not None #or key != "index" # that does not work
        }

        return items

    def __len__(self):
        return len(self.paths['gt'])



# testing
if __name__ == '__main__':
    # depth_pred = pp_depth_read("/home/maciej/git/igdc/pp_implementation_testing/pred/0.png")
    # depth_velo = pp_depth_read("/home/maciej/ros1_wss/pp_collector/src/pp_img_undistorter/tmp/test_dmap.png")
    args = None
    get_rgb_near("/home/maciej/git/igdc/pp_implementation_testing/rgb/3.png", args)


    print("")
