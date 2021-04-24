import os
import glob

import numpy as np
import torch.utils.data as data
from dataloaders import transforms
from PIL import Image

import matplotlib.pyplot as plt
from dataloaders.kitti_loader import handle_gray

# input_options = ['depth', 'rgb', 'rgb_d']


def load_calib():
    # so far hardcoded!
    # proj matrix is the one with lens undistorted
    Proj_matrix = np.array([[3225.25977, 0.0, 996.62146, 0.0],
                            [0.0, 3224.54517, 772.72072, 0.0],
                            [0.0, 0.0, 1.0, 0.0]])
    K = Proj_matrix[:3, :3]

    # here changes and calulations!!

    return K


def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()

    return rgb_png


def depth_read(filename):
    """
    copied from https://github.com/fangchangma/self-supervised-depth-completion
    """
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
    # change shape from (height, width) to (height, width, 1)
    depth = np.expand_dims(depth, -1)

    return depth


def display_rgb_or_depth(img):
    # if img.shape[2] == 1:
    img = np.squeeze(img)
    imgplot = plt.imshow(img)
    plt.show()


def get_paths(args):
    if args.mode == "infer":
        glob_rgb = os.path.join(args.data_folder, "rgb/*.png")
        glob_lidar_depth = os.path.join(args.data_folder, "lidar_depth_map/*.png")

        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_lidar_depth = sorted(glob.glob(glob_lidar_depth))

    if len(paths_lidar_depth) == 0 and len(paths_rgb) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_rgb)))

    # just to comply with ma2019self input
    paths_gt = None

    return {"rgb": paths_rgb, "d": paths_lidar_depth, "gt": paths_gt}


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


class PointPixDataset(data.Dataset):
    """
    A data stream coming from synchronised camera and lidar
    """

    def __init__(self, args):
        self.args = args
        self.paths = get_paths(args)

    def get_item_from_index(self, index):
        rgb = rgb_read(self.paths['rgb'][index]) if self.paths['rgb'][index] is not None else None
        sparse = depth_read(self.paths['d'][index]) if self.paths['d'][index] is not None else None
        # target = depth_read(self.paths['gt'][index]) if self.paths['gt'][index] is not None else None
        target, rgb_near = None, None
        return rgb, sparse, target, rgb_near

    def __getitem__(self, index):
        rgb, sparse, target, rgb_near = self.get_item_from_index(index)
        r_mat, t_vec = None, None

        rgb, gray = handle_gray(rgb, self.args)
        candidates = {"rgb": rgb, "d": sparse, "gt": target, \
                      "g": gray, "r_mat": r_mat, "t_vec": t_vec, "rgb_near": rgb_near}

        items = {
            key: to_float_tensor(val)
            for key, val in candidates.items() if val is not None
        }
        return items


    def __len__(self):
        return len(self.paths['d'])




class Arguments:

    def __init__(self):
        self.data_folder = None


#######################################
### TESTING ### TESTING ### TESTING ###
if __name__ == '__main__':
    # load_calib()
    # rgb_read("/home/maciej/git/depth_compl/self-supervised-depth-completion/data/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png")
    # this = depth_read("/home/maciej/git/depth_compl/self-supervised-depth-completion/data/depth_selection/val_selection_cropped/velodyne_raw/2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.png")
    # display_rgb_or_depth(this)

    args = Arguments()
    args.data_folder = "/home/maciej/git/depth_compl/self-supervised-depth-completion/data_pp"
    args.mode = "infer"
    # get_paths(args)
    infer_dataset = PointPixDataset(args)
    test = infer_dataset.__getitem__(0)
    print("")
