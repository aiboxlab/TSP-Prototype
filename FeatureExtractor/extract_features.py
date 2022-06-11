import os
import math
import pandas as pd
import numpy as np
import torch
import cv2
import datetime as dt
from datetime import datetime
from FeatureExtractor.models.pytorch_i3d import InceptionI3d

# ===== Squeeze =====
from squeeze import squeeze_net

# ===== Rescaling =====
def crop_slices(image, height, width, slice_size):
    return image[0:height, 0+slice_size:width-slice_size]

def resize_square(image, resolution_factor):
    return cv2.resize(image, (resolution_factor, resolution_factor)) 

def make_frame(image, to_resolution):
    height, width, _ = image.shape
    diff = max(height, width) - min(height, width)
    slice_size= int(math.floor(diff/2))
    crop_img = crop_slices(image, height, width, slice_size)
    resize_img = resize_square(crop_img, to_resolution)
    return resize_img
# ===== Rescaling =====


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def load_all_rgb_frames_from_video(video, desired_channel_order='rgb'):
    cap = cv2.VideoCapture(video)
    
    frames = []

    while(True):

        frame = np.zeros((224,224,3), np.uint8)

        try:
            ret, frame = cap.read()

            # Rescaling
            frame = make_frame(frame, 224)

            frame = cv2.resize(frame, dsize=(224, 224))

            frame_transformed = frame.copy()   
            
            if desired_channel_order == 'bgr':
                frame_transformed = frame_transformed[:, :, [2, 1, 0]]

            frame_transformed = (frame_transformed / 255.) * 2 - 1
            frames.append(frame_transformed)

        except:
            break

    nframes = np.asarray(frames, dtype=np.float32)
    
    return nframes


def extract_features_fullvideo(model, inp, framespan, stride):
    rv = []

    indices = list(range(len(inp)))
    groups = []
    for ind in indices:

        if ind % stride == 0:
            groups.append(list(range(ind, ind+framespan)))

    for g in groups:
        # numpy array indexing will deal out-of-index case and return only till last available element
        frames = inp[g[0]: min(g[-1]+1, inp.shape[0])]

        num_pad = 9 - len(frames)
        if num_pad > 0:
            pad = np.tile(np.expand_dims(frames[-1], axis=0), (num_pad, 1, 1, 1))
            frames = np.concatenate([frames, pad], axis=0)

        frames = frames.transpose([3, 0, 1, 2])

        ft = _extract_features(model, frames)

        rv.append(ft)

    return rv


def _extract_features(model, frames):
    inputs = torch.from_numpy(frames)

    inputs = inputs.unsqueeze(0)

    inputs = inputs.cuda()
    with torch.no_grad():
        ft = model.extract_features(inputs)
    ft = ft.squeeze(-1).squeeze(-1)[0].transpose(0, 1)

    ft = ft.cpu()

    return ft


def run(weight, video, outroot, inp_channels='rgb'):

    # ===== setup models ======
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(2000)
    i3d.load_state_dict(torch.load(weight)) # Network's Weight
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    name = os.path.basename(video)[:-4]
    text = ""

    print('extracting.')

    # ===== extract features ======
    for framespan, stride in [(16, 2), (12, 2), (8, 2)]:

        outdir = os.path.join(outroot, 'span={}_stride={}'.format(framespan, stride))

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        out_path = os.path.join(outdir, os.path.basename(video[:-4])) + '.pt'

        frames = load_all_rgb_frames_from_video(video, inp_channels)
            
        features = extract_features_fullvideo(i3d, frames, framespan, stride)

        text = "[{\"ident\": \""+ name +"\", \"size\": "+ str(len(features)) +"}]"

        #print(name, len(features))

        # ===== Squeeze Avgpooling =====
        features = squeeze_net(name, features, stride)

        torch.save(features, os.path.join(outdir, os.path.basename(video[:-4])) + '.pt')

    return text