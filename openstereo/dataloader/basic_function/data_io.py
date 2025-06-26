from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from PIL import Image
import sys
import cv2


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def pil_loader(filename):
    return np.array(Image.open(filename).convert('RGB'), dtype=np.float32)


def cv2_loader(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    # return cv2.imread(filename)


def pfm_disp_loader(filename, subset=False):
    # For finalpass and cleanpass, gt disparity is positive, subset is negative
    disp = np.ascontiguousarray(_read_pfm(filename)[0].astype(np.float32))
    if subset:
        disp = -disp
    return disp


def _read_kitti_disp(filename):
    depth = np.array(Image.open(filename))
    depth = depth.astype(np.float32) / 256.
    return depth


def _read_disp_png(filename, scale_factor=128.):
    assert isinstance(scale_factor, float)
    depth = np.array(Image.open(filename), dtype=np.float32)
    depth = depth.astype(np.float32) / scale_factor
    return depth


def png_disp_loader(path):
    return np.array(Image.open(path), dtype=np.float32)


def read_disp(filename, subset=False, scale_factor=None):
    # Scene Flow dataset
    if filename.endswith('pfm'):
        # For finalpass and cleanpass, gt disparity is positive, subset is negative
        disp = np.ascontiguousarray(_read_pfm(filename)[0])
        if subset:
            disp = -disp
    # KITTI
    elif filename.endswith('png') and scale_factor is None:
        disp = _read_kitti_disp(filename)
    elif filename.endswith('png') and scale_factor is not None:
        # Log.info("loading img with scale factor {}".format(scale_factor))
        disp = _read_disp_png(filename, scale_factor)
    elif filename.endswith('npy'):
        disp = np.load(filename)
    else:
        raise Exception('Invalid disparity file format!')
    return disp  # [H, W]


def _read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(
            image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)

    image.tofile(file)

