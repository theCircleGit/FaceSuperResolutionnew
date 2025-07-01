import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

####### for super resolution on faces
#from dlib_alignment import dlib_detect_face, face_recover
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.SRGAN_model import SRGANModel
import numpy as np
import argparse
from models.SupeResolution import utils
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from skimage import transform as trans
import dlib
from decimal import Decimal, getcontext

# Set the desired precision (e.g. 50 decimal places)
getcontext().prec = 50 

import glob
import scipy
import scipy.ndimage
import pdb
import os
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.utils.registry import ARCH_REGISTRY
#from retinaface import RetinaFace
## Code Former pretraine model
pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

D_LMF=1; # Dlib landmarks flag

dlib_detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor('weights/dlib/shape_predictor_68_face_landmarks.dat')
#predictor = dlib.shape_predictor('weights/dlib/shape_predictor_68_face_landmarks_GTX.dat')

predictor = dlib.shape_predictor('weights/dlib/shape_predictor_68_face_landmarks-fbdc2cb8.dat')

_transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                      std=[0.5, 0.5, 0.5])])

def read_cv2_img(path):
    '''
    Read color images
    :param path: Path to image
    :return: Only returns color images
    '''
    img = cv2.imread(path,  cv2.IMREAD_COLOR)
   

    if img is not None:
        if len(img.shape) != 3:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        

    return img

# def savef(data):
#     myFile = filedialog.asksaveasfile(mode='w', defaultextension='.jpg',filetypes=[('jpg', '*.jpg'), ('png', '*.png')])
#     if myFile is None:
#         return
#
#     data.save(myFile.name)
#     myFile.close()
#
# def savefTwo(data1,data2):
#     myFile = filedialog.asksaveasfile(mode='w', defaultextension='.jpg',filetypes=[('jpg', '*.jpg'), ('png', '*.png')])
#     if myFile is None:
#         return
#     #pdb.set_trace()
#     ## saving both High Resloution Image
#     tempPath= myFile.name;
#     NameParts= tempPath.split('.')
#
#     data1.save(NameParts[0] + '_HR1.' + NameParts[1])
#     data2.save(NameParts[0] + '_HR2.' + NameParts[1])
#     myFile.close()
#
# def savefFive(data1,data2,data3,data4,data5):
#     myFile = filedialog.asksaveasfile(mode='w', defaultextension='.jpg',filetypes=[('jpg', '*.jpg'), ('png', '*.png')])
#     if myFile is None:
#         return
#     #pdb.set_trace()
#     ## saving both High Resloution Image
#     tempPath= myFile.name;
#     NameParts= tempPath.split('.')
#
#     data1.save(NameParts[0] + '_HR1.' + NameParts[1])
#     data2.save(NameParts[0] + '_HR2.' + NameParts[1])
#     data3.save(NameParts[0] + '_HR3.' + NameParts[1])
#     data4.save(NameParts[0] + '_HR4.' + NameParts[1])
#     data5.save(NameParts[0] + '_HR5.' + NameParts[1])
#
#     myFile.close()
#
# def savefSix(data1,data2,data3,data4,data5,data6):
#     myFile = filedialog.asksaveasfile(mode='w', defaultextension='.jpg',filetypes=[('jpg', '*.jpg'), ('png', '*.png')])
#     if myFile is None:
#         return
#     #pdb.set_trace()
#     ## saving both High Resloution Image
#     tempPath= myFile.name;
#     NameParts= tempPath.split('.')
#
#     data1.save(NameParts[0] + '_HR1.' + NameParts[1])
#     data2.save(NameParts[0] + '_HR2.' + NameParts[1])
#     data3.save(NameParts[0] + '_HR3.' + NameParts[1])
#     data4.save(NameParts[0] + '_HR4.' + NameParts[1])
#     data5.save(NameParts[0] + '_HR5.' + NameParts[1])
#     data5.save(NameParts[0] + '_HR6.' + NameParts[1])
#
#     myFile.close()



def convert_img(input_img):
    input_img = np.asarray(input_img)
    input_img = tf.clip_by_value(input_img, 0, 255)
    input_img = Image.fromarray(tf.cast(input_img, tf.uint8).numpy())
    return input_img

# def display(img1, img2):
#     # Create a figure with two subplots, arranged side by side
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#     fig.patch.set_visible(False)
#     ax1.axis('off')
#     ax2.axis('off')
#
#     # Display the first image on the left subplot
#     ax1.imshow(convert_img(img1))
#     ax1.set_title('Original Image')
#
#     # Display the second image on the right subplot
#     ax2.imshow(convert_img(img2))
#     ax2.set_title('High Resolution Image')
#
#     # Remove the axis ticks and labels for a cleaner look
#     ax1.axis('off')
#     ax2.axis('off')
#
#     # Adjust the layout for better spacing
#     plt.tight_layout()
#
#     # Show the figure with the two images and captions
#     plt.show()
#     ans = messagebox.askyesno("Save image", "Do want to save the output image?")
#     if ans:
#         savef(convert_img(img2))
#
#
# def displayFour(img1, img2,img3,img4, imgE1,imgE2, imgE3):
#     # Create a figure with two subplots, arranged side by side
#     fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(10, 5))
#     fig.patch.set_visible(False)
#     ax1.axis('off')
#     ax2.axis('off')
#     ax3.axis('off')
#     ax4.axis('off')
#
#     # Display the first image on the left subplot
#     #ax1.imshow(convert_img(img1))
#     ax1.imshow(img1)
#     ax1.set_title('Original Image')
#
#     ax2.imshow(img2)
#     ax2.set_title('High Resolution Image 1')
#
#     # Display the second image on the right subplot
#     #ax2.imshow(convert_img(img2))
#     # because of codeformer
#     #img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
#     ax3.imshow(img3)
#     ax3.set_title('High Resolution Image 2')
#
#     #img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
#     ax4.imshow(img4)
#     ax4.set_title('High Resolution Image 3')
#
#     # Remove the axis ticks and labels for a cleaner look
#     ax1.axis('off')
#     ax2.axis('off')
#     ax3.axis('off')
#     ax4.axis('off')
#
#     # Adjust the layout for better spacing
#     plt.tight_layout()
#
#     # Show the figure with the two images and captions
#     plt.show()
#     ans = messagebox.askyesno("Save image", "Do want to save the output image?")
#     if ans:
#         #pdb.set_trace()
#         #savefTwo(convert_img(img2),convert_img(img3))
#         savefSix(convert_img(img2),convert_img(img3),convert_img(img4),convert_img(imgE1),convert_img(imgE2),convert_img(imgE3))
#
#
#
# def displayThree(img1, img2,img3, imgE1,imgE2, imgE3):
#     # Create a figure with two subplots, arranged side by side
#     fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 5))
#     fig.patch.set_visible(False)
#     ax1.axis('off')
#     ax2.axis('off')
#     ax3.axis('off')
#
#     # Display the first image on the left subplot
#     #ax1.imshow(convert_img(img1))
#     ax1.imshow(img1)
#     ax1.set_title('Original Image')
#
#     # Display the second image on the right subplot
#     #ax2.imshow(convert_img(img2))
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#     ax2.imshow(img2)
#     ax2.set_title('High Resolution Image 1')
#
#     img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
#     ax3.imshow(img3)
#     ax3.set_title('High Resolution Image 2')
#
#     # Remove the axis ticks and labels for a cleaner look
#     ax1.axis('off')
#     ax2.axis('off')
#     ax3.axis('off')
#
#     # Adjust the layout for better spacing
#     plt.tight_layout()
#
#     # Show the figure with the two images and captions
#     plt.show()
#     ans = messagebox.askyesno("Save image", "Do want to save the output image?")
#     if ans:
#         #pdb.set_trace()
#         #savefTwo(convert_img(img2),convert_img(img3))
#         imgE1 = cv2.cvtColor(imgE1, cv2.COLOR_BGR2RGB)
#         imgE2 = cv2.cvtColor(imgE2, cv2.COLOR_BGR2RGB)
#         imgE3 = cv2.cvtColor(imgE3, cv2.COLOR_BGR2RGB)
#         savefFive(convert_img(img2),convert_img(img3),convert_img(imgE1),convert_img(imgE2),convert_img(imgE3))

def preprocess_image(image_path):

    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)


def downscale_image(image):

    if len(image.shape) == 3:
        image_size = [image.shape[1], image.shape[0]]
    else:
        raise ValueError("Dimension mismatch. Can work only on single image.")

    image = tf.squeeze(
        tf.cast(
            tf.clip_by_value(image, 0, 255), tf.uint8))

    lr_image = np.asarray(
        Image.fromarray(image.numpy()).resize([image_size[0] // 4, image_size[1] // 4], Image.BICUBIC))

    lr_image = tf.expand_dims(lr_image, 0)
    lr_image = tf.cast(lr_image, tf.float32)
    return lr_image

def get_FaceSR_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr_G', type=float, default=1e-4)
    parser.add_argument('--weight_decay_G', type=float, default=0)
    parser.add_argument('--beta1_G', type=float, default=0.9)
    parser.add_argument('--beta2_G', type=float, default=0.99)
    parser.add_argument('--lr_D', type=float, default=1e-4)
    parser.add_argument('--weight_decay_D', type=float, default=0)
    parser.add_argument('--beta1_D', type=float, default=0.9)
    parser.add_argument('--beta2_D', type=float, default=0.99)
    parser.add_argument('--lr_scheme', type=str, default='MultiStepLR')
    parser.add_argument('--niter', type=int, default=100000)
    parser.add_argument('--warmup_iter', type=int, default=-1)
    parser.add_argument('--lr_steps', type=list, default=[50000])
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--pixel_criterion', type=str, default='l1')
    parser.add_argument('--pixel_weight', type=float, default=1e-2)
    parser.add_argument('--feature_criterion', type=str, default='l1')
    parser.add_argument('--feature_weight', type=float, default=1)
    parser.add_argument('--gan_type', type=str, default='ragan')
    parser.add_argument('--gan_weight', type=float, default=5e-3)
    parser.add_argument('--D_update_ratio', type=int, default=1)
    parser.add_argument('--D_init_iters', type=int, default=0)

    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--crop_size', type=float, default=0.85)
    parser.add_argument('--lr_size', type=int, default=128)
    parser.add_argument('--hr_size', type=int, default=512)

    # network G
    parser.add_argument('--which_model_G', type=str, default='RRDBNet')
    parser.add_argument('--G_in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--G_nf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=16)

    # network D
    parser.add_argument('--which_model_D', type=str, default='discriminator_vgg_128')
    parser.add_argument('--D_in_nc', type=int, default=3)
    parser.add_argument('--D_nf', type=int, default=64)

    # data dir
    
  
    parser.add_argument('--pretrain_model_G', type=str, default='weights/FaceESRGAN/90000_G.pth')
    parser.add_argument('--pretrain_model_D', type=str, default=None)

    args, _ = parser.parse_known_args([])

    return args


sr_model = SRGANModel(get_FaceSR_opt(), is_train=False)
sr_model.load()

def face_recover(img, M, ori_img):
    # img:rgb, ori_img:bgr
    # dst:rgb
    dst = ori_img.copy()
    cv2.warpAffine(img, M, (dst.shape[1], dst.shape[0]), dst,
                   flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_TRANSPARENT)
    return dst


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def dlib_alignment(img, landmarks, padding=0.25, size=128, moving=0.0):
    x_src = np.array([0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
                      0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
                      0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
                      0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
                      0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
                      0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
                      0.553364, 0.490127, 0.42689])
    y_src = np.array([0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
                      0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
                      0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
                      0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
                      0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
                      0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
                      0.784792, 0.824182, 0.831803, 0.824182])
    x_src = ((padding) + (x_src)) / ((2*padding) + 1)
    y_src = ((padding) + (y_src)) / ((2*padding) + 1)
    y_src += moving
    x_src *= size
    y_src *= size

    src = np.concatenate([np.expand_dims(x_src, -1), np.expand_dims(y_src, -1)], -1)
    dst = landmarks.astype(np.float32)
    src = np.concatenate([src[10:38, :], src[43:48, :]], axis=0)
    dst = np.concatenate([dst[27:55, :], dst[60:65, :]], axis=0)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]

    warped = cv2.warpAffine(img, M, (size, size), borderValue=0.0)

    return warped, M




def dlib_detect_face(img, image_size=(128, 128), padding=0.25, moving=0.0):
    dets = dlib_detector(img, 0)
    if dets:
        if isinstance(dets, dlib.rectangles):
            det = max(dets, key=lambda d: d.area())
        else:
            det = max(dets, key=lambda d: d.rect.area())
            det = det.rect
        face = predictor(img, det)
        landmarks = shape_to_np(face)
        img_aligned, M = dlib_alignment(img, landmarks, size=image_size[0], padding=padding, moving=moving)

        return img_aligned, M
    else:
        return None
    
def sr_forward(img, padding=0.5, moving=0.1):
    img_aligned, M = dlib_detect_face(img, padding=padding, image_size=(128, 128), moving=moving)
    input_img = torch.unsqueeze(_transform(Image.fromarray(img_aligned)), 0)
    sr_model.var_L = input_img.to(sr_model.device)
    sr_model.test()
    output_img = sr_model.fake_H.squeeze(0).cpu().numpy()
    output_img = np.clip((np.transpose(output_img, (1, 2, 0)) / 2.0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    rec_img = face_recover(output_img, M * 4, img)
    return output_img, rec_img


############## Functions for  CodeFormer , face alignmnets and restoration
def get_landmark(filepath, only_keep_largest=True):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    # Shangchen modified
    print("\tNumber of faces detected: {}".format(len(dets)))
    if only_keep_largest:
        print('\tOnly keep the largest.')
        face_areas = []
        for k, d in enumerate(dets):
            face_area = (d.right() - d.left()) * (d.bottom() - d.top())
            face_areas.append(face_area)

        largest_idx = face_areas.index(max(face_areas))
        d = dets[largest_idx]
        shape = predictor(img, d)
        # print("Part 0: {}, Part 1: {} ...".format(
        #     shape.part(0), shape.part(1)))
    else:
        for k, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #     k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            # print("Part 0: {}, Part 1: {} ...".format(
            #     shape.part(0), shape.part(1)))

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    # lm is a shape=(68,2) np.array
    return lm

def align_face(filepath):
    """
    :param filepath: str
    :return: PIL Image
    """
    try:
        lm = get_landmark(filepath)
        D_LMF=1;
    except:
        print('No landmark ...')
        D_LMF=0;
        img=0
        return

    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(filepath)

    output_size = 512
    transform_size = 4096
    enable_padding = False

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
 
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border,
                img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border,
               0), max(-pad[1] + border,
                       0), max(pad[2] - img.size[0] + border,
                               0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
            'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 -
            np.minimum(np.float32(x) / pad[0],
                       np.float32(w - 1 - x) / pad[2]), 1.0 -
            np.minimum(np.float32(y) / pad[1],
                       np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) -
                img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(
            np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)

    if output_size < transform_size:
        #img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)# by shan, ANTIALIAS was removed in Pillow 10.0.0
        img = img.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS)

    # Save aligned image.
    # print('saveing: ', out_path)
    #img.save(out_path)
    #return img, np.max(quad[:, 0]) - np.min(quad[:, 0])
    return img

def set_realesrgan():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available(): # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )

    if not gpu_is_available():  # CPU
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                        'The unoptimized RealESRGAN is slow on CPU. '
                        'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                        category=RuntimeWarning)
    return upsampler

def CodeFormerMain(ImagePTH_Data, algF, FedW):
    device = get_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./inputs/whole_imgs', 
            help='Input image, video or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output_path', type=str, default=None, 
            help='Output folder. Default: results/<input_name>_<w>')
    parser.add_argument('-w', '--fidelity_weight', type=float, default=0.5, 
            help='Balance the quality and fidelity. Default: 0.5')
    parser.add_argument('-s', '--upscale', type=int, default=2, 
            help='The final upsampling scale of the image. Default: 2')
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
    parser.add_argument('--draw_box', action='store_true', help='Draw the bounding box for the detected faces. Default: False')
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50', 
            help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
                Default: retinaface_resnet50')
    parser.add_argument('--bg_upsampler', type=str, default='None', help='Background upsampler. Optional: realesrgan')
    parser.add_argument('--face_upsample', action='store_true', help='Face upsampler after enhancement. Default: False')
    parser.add_argument('--bg_tile', type=int, default=400, help='Tile size for background sampler. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces. Default: None')
    parser.add_argument('--save_video_fps', type=float, default=None, help='Frame rate for saving video. Default: None')

    args, _ = parser.parse_known_args([])
    ## setting our variables/configurations ####### Later needs to be in Gui, like combo box and check boxes
    #args.input_path = InputImgPath; # passing the input path from outside
    
    
    args.face_upsample=True; 
    args.bg_upsampler == 'realesrgan'
   
    # ------------------------ input & output ------------------------
    #w = args.fidelity_weight
    w = FedW  # fedility number
  
    args.has_aligned= algF ; # alignment flag
    if  algF == False:
        args.input_path= ImagePTH_Data ## will be single image path
        input_video = False
        if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
            input_img_list = [args.input_path]
            result_root = f'results/test_img_{w}'
        elif args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
            from basicsr.utils.video_util import VideoReader, VideoWriter
            input_img_list = []
            vidreader = VideoReader(args.input_path)
            image = vidreader.get_frame()
            while image is not None:
                input_img_list.append(image)
                image = vidreader.get_frame()
            audio = vidreader.get_audio()
            fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps   
            video_name = os.path.basename(args.input_path)[:-4]
            result_root = f'results/{video_name}_{w}'
            input_video = True
            vidreader.close()
        else: # input img folder
            if args.input_path.endswith('/'):  # solve when path ends with /
                args.input_path = args.input_path[:-1]
            # scan all the jpg and png images
            input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
            result_root = f'results/{os.path.basename(args.input_path)}_{w}'

        if not args.output_path is None: # set output path
            result_root = args.output_path

        test_img_num = len(input_img_list)
        if test_img_num == 0:
            raise FileNotFoundError('No input image/video is found...\n' 
                '\tNote that --input_path for video should end with .mp4|.mov|.avi')

        # ------------------ set up background upsampler ------------------
        if args.bg_upsampler == 'realesrgan':
            bg_upsampler = set_realesrgan()
        else:
            bg_upsampler = None

        # ------------------ set up face upsampler ------------------
        if args.face_upsample:
            if bg_upsampler is not None:
                face_upsampler = bg_upsampler
            else:
                face_upsampler = set_realesrgan()
        else:
            face_upsampler = None

        # ------------------ set up CodeFormer restorer -------------------
        net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                                connect_list=['32', '64', '128', '256']).to(device)
        
        # ckpt_path = 'weights/CodeFormer/codeformer.pth'
        ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                        model_dir='weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        net.load_state_dict(checkpoint)
        net.eval()

        # ------------------ set up FaceRestoreHelper -------------------
        # large det_model: 'YOLOv5l', 'retinaface_resnet50'
        # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
        if not args.has_aligned: 
            print(f'Face detection model: {args.detection_model}')
        if bg_upsampler is not None: 
            print(f'Background upsampling: True, Face upsampling: {args.face_upsample}')
        else:
            print(f'Background upsampling: False, Face upsampling: {args.face_upsample}')

        face_helper = FaceRestoreHelper(
            args.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model = args.detection_model,
            save_ext='png',
            use_parse=True,
            device=device)

        # -------------------- start to processing ---------------------
        for i, img_path in enumerate(input_img_list):
            # clean all the intermediate results to process the next image
            face_helper.clean_all()
            
            if isinstance(img_path, str):
                img_name = os.path.basename(img_path)
                basename, ext = os.path.splitext(img_name)
                print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            else: # for video processing
                basename = str(i).zfill(6)
                img_name = f'{video_name}_{basename}' if input_video else basename
                print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
                img = img_path

            if args.has_aligned: 
                # the input faces are already cropped and aligned
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_helper.is_gray = is_gray(img, threshold=10)
                if face_helper.is_gray:
                    print('Grayscale input: True')
                face_helper.cropped_faces = [img]
            else:
                face_helper.read_image(img)
                # get face landmarks for each face
                num_det_faces = face_helper.get_face_landmarks_5(
                    only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
                print(f'\tdetect {num_det_faces} faces')
                if num_det_faces==0:
                    return np.array([[0]]) ## to show no image is detected

                # align and warp each face
                face_helper.align_warp_face()

            # face restoration for each cropped face
            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                # prepare data
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        output = net(cropped_face_t, w=w, adain=True)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                    torch.cuda.empty_cache()
                except Exception as error:
                    print(f'\tFailed inference for CodeFormer: {error}')
                    restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                restored_face = restored_face.astype('uint8')
                face_helper.add_restored_face(restored_face, cropped_face)

            # paste_back
            if not args.has_aligned:
                # upsample the background
                if bg_upsampler is not None:
                    # Now only support RealESRGAN for upsampling background
                    bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
                else:
                    bg_img = None
                face_helper.get_inverse_affine(None)
                # paste each restored face to the input image
                if args.face_upsample and face_upsampler is not None: 
                    restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box, face_upsampler=face_upsampler)
                else:
                    restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box)

            # save faces
            for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
                # save cropped face
                if not args.has_aligned: 
                    save_crop_path = os.path.join(result_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
                    imwrite(cropped_face, save_crop_path)
                # save restored face
                if args.has_aligned:
                    save_face_name = f'{basename}.png'
                else:
                    save_face_name = f'{basename}_{idx:02d}.png'
                if args.suffix is not None:
                    save_face_name = f'{save_face_name[:-4]}_{args.suffix}.png'
                save_restore_path = os.path.join(result_root, 'restored_faces', save_face_name)
                imwrite(restored_face, save_restore_path)

            # save restored img
            if not args.has_aligned and restored_img is not None:
                if args.suffix is not None:
                    basename = f'{basename}_{args.suffix}'
                save_restore_path = os.path.join(result_root, 'final_results', f'{basename}.png')
                imwrite(restored_img, save_restore_path)

        # save enhanced video
        if input_video:
            print('Video Saving...')
            # load images
            video_frames = []
            img_list = sorted(glob.glob(os.path.join(result_root, 'final_results', '*.[jp][pn]g')))
            for img_path in img_list:
                img = cv2.imread(img_path)
                video_frames.append(img)
            # write images to video
            height, width = video_frames[0].shape[:2]
            if args.suffix is not None:
                video_name = f'{video_name}_{args.suffix}.png'
            save_restore_path = os.path.join(result_root, f'{video_name}.mp4')
            vidwriter = VideoWriter(save_restore_path, height, width, fps, audio)
            
            for f in video_frames:
                vidwriter.write_frame(f)
            vidwriter.close()

        print(f'\nAll results are saved in {result_root}')

        return restored_face
    
    elif algF == True:

        test_img_num = 1 # fo reading one image only 
        img= ImagePTH_Data # In this case, we need to pass image--> aligned 
        # ------------------ set up background upsampler ------------------
        if args.bg_upsampler == 'realesrgan':
            bg_upsampler = set_realesrgan()
        else:
            bg_upsampler = None

        # ------------------ set up face upsampler ------------------
        if args.face_upsample:
            if bg_upsampler is not None:
                face_upsampler = bg_upsampler
            else:
                face_upsampler = set_realesrgan()
        else:
            face_upsampler = None

        # ------------------ set up CodeFormer restorer -------------------
        net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                                connect_list=['32', '64', '128', '256']).to(device)
        
        #ckpt_path = 'weights/CodeFormer/codeformer.pth'
        ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                        model_dir='weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        net.load_state_dict(checkpoint)
        net.eval()

        # ------------------ set up FaceRestoreHelper -------------------
        # large det_model: 'YOLOv5l', 'retinaface_resnet50'
        # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
        if not args.has_aligned: 
            print(f'Face detection model: {args.detection_model}')
        if bg_upsampler is not None: 
            print(f'Background upsampling: True, Face upsampling: {args.face_upsample}')
        else:
            print(f'Background upsampling: False, Face upsampling: {args.face_upsample}')
        
    
        face_helper = FaceRestoreHelper(
            args.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model = args.detection_model,
            save_ext='png',
            use_parse=True,
            device=device)

        # -------------------- start to processing ---------------------
    # for i, img_path in enumerate(input_img_list):
            # clean all the intermediate results to process the next image
        face_helper.clean_all()
    
        
        if args.has_aligned: 
            # the input faces are already cropped and aligned
            #pdb.set_trace()
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=10)
            if face_helper.is_gray:
                print('Grayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            #pdb.set_trace()
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = net(cropped_face_t, w=w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        if not args.has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if args.face_upsample and face_upsampler is not None: 
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box, face_upsampler=face_upsampler)
            else:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box)

        return restored_face



def enhance_image(image_path, model):
    
    #sr_model = SRGANModel(get_FaceSR_opt(), is_train=False)
    #sr_model.load()
 
    ################################# Old ESR GAN results ######





    #hr_image = preprocess_image(image_path)
    #lr_image = downscale_image(tf.squeeze(hr_image))
    #img_path = 'Sample_face2.PNG'
    #orig_Image = utils.read_cv2_img(image_path)
    
    #######****** ESRGANs retrained on Faces #######
    orig_Image = read_cv2_img(image_path) # To be removed to make it consistent
    #output_img, HR_img = sr_forward(orig_Image)
    ###########################################################
    #utils.save_image(output_img, 'output_face.jpg') ## corrected image without transformations
    #utils.save_image(rec_img, 'output_img.jpg') # ## corrected image with transformations\
    #pdb.set_trace()
    
    PiLImage = align_face(image_path) ## Face cropped image
                    

    if PiLImage is not None:
        #pdb.set_trace()
        AlignedFace_img = np.array(PiLImage)# converting pil to numpy 
        output_img, HR_img = sr_forward(AlignedFace_img) ### ESRGAN ,retrained on Face Images
        
        HR_Img_CodeFormer1 = CodeFormerMain(AlignedFace_img,True, 0)  ##CodeFormer, alighmnet flag, fedility weight
        HR_Img_CodeFormer2 = CodeFormerMain(AlignedFace_img,True, 0.2)  ##CodeFormer, alighmnet flag, fedility weight
        HR_Img_CodeFormer3 = CodeFormerMain(AlignedFace_img,True, 0.5)  ##CodeFormer, alighmnet flag, fedility weight
        HR_Img_CodeFormer4 = CodeFormerMain(AlignedFace_img,True, 0.7)  ##CodeFormer, alighmnet flag, fedility weight
        HR_Img_CodeFormer5 = CodeFormerMain(AlignedFace_img,True, 1)  ##CodeFormer, alighmnet flag, fedility weight
        # displayFour(AlignedFace_img, HR_img, HR_Img_CodeFormer1, HR_Img_CodeFormer4,HR_Img_CodeFormer2,HR_Img_CodeFormer3,HR_Img_CodeFormer5)
    else: 

        ### testing code 
        HR_Img_CodeFormer1 = CodeFormerMain(image_path,False, 0)  ##CodeFormer
        #pdb.set_trace()
        if (len(HR_Img_CodeFormer1)==1):## means single  integer  not detected face by ratina face, so needs to be treated as generalized image 
            
            hr_image = preprocess_image(image_path)
            lr_image = downscale_image(tf.squeeze(hr_image))
            fake_image = model(lr_image)
            fake_image = tf.squeeze(fake_image)
  
            # display(tf.squeeze(hr_image), tf.squeeze(fake_image))
        else:
            HR_Img_CodeFormer2 = CodeFormerMain(image_path,False, 0.2)  ##CodeFormer
            HR_Img_CodeFormer3 = CodeFormerMain(image_path,False, 0.5)  ##CodeFormer
            HR_Img_CodeFormer4 = CodeFormerMain(image_path,False, 0.7)  ##CodeFormer
            HR_Img_CodeFormer5 = CodeFormerMain(image_path,False, 1)  ##CodeFormer
            #img=np.array(faces[0]);
            
            displayThree(orig_Image, HR_Img_CodeFormer1, HR_Img_CodeFormer4,HR_Img_CodeFormer2,HR_Img_CodeFormer3,HR_Img_CodeFormer5)
        #pdb.set_trace()
        '''
        faces = RetinaFace.extract_faces(img_path = image_path, align = True) 
        for face in faces:
        #Resizedface=cv2.resize(face, (512,512), interpolation = cv2.INTER_AREA)
            output_img, HR_img = sr_forward(face) ### ESRGAN ,retrained on Face Images
            #HR_Img_CodeFormerAL = CodeFormerMain(face)  ##CodeFormer
            
            displayThree(face, HR_img, HR_Img_CodeFormer)

           ''' 
    ######### Old implementation of ESR GANS using TF
    '''
    hr_image = preprocess_image(image_path)
    lr_image = downscale_image(tf.squeeze(hr_image))


    start = time.time()
    fake_image = model(lr_image)
    fake_image = tf.squeeze(fake_image)
    # print("Time Taken: %f" % (time.time() - start))
    display(tf.squeeze(hr_image), tf.squeeze(fake_image))
    '''


        #start = time.time()
        #fake_image = model(lr_image)
        #fake_image = tf.squeeze(fake_image)
        # print("Time Taken: %f" % (time.time() - start))
        #display(tf.squeeze(hr_image), tf.squeeze(fake_image))
        ##display(orig_Image, HR_img) # final for ESRGAN- Fine Tuned at face images
        #display(orig_Image, HR_Img_CodeFormer) ## for CoderFormer
        #pdb.set_trace()
    
        

    #######****** ESRGANs retrained on Faces #######
    #orig_Image = read_cv2_img(image_path) # To be removed to make it consistent
    #output_img, HR_img = sr_forward(orig_Image)
    ###########################################################
    #utils.save_image(output_img, 'output_face.jpg') ## corrected image without transformations
    #utils.save_image(rec_img, 'output_img.jpg') # ## corrected image with transformations\
    
    #PiLImage = align_face(image_path) ## Face cropped image
    #pdb.set_trace()
    #PiLImage.save('test_Face_Align_image.png')
    #FaceAlignedImage = read_cv2_img('test_Face_Align_image.png')
    #AlignedFace_img = np.array(PiLImage)# converting pil to numpy 
    #output_img, HR_img = sr_forward(AlignedFace_img) ### ESRGAN ,retrained on Face Images
    #HR_Img_CodeFormer = CodeFormerMain(FaceAlignedImage)  ##CodeFormer

    

    #start = time.time()
    #fake_image = model(lr_image)
    #fake_image = tf.squeeze(fake_image)
    # print("Time Taken: %f" % (time.time() - start))
    #display(tf.squeeze(hr_image), tf.squeeze(fake_image))
    ##display(orig_Image, HR_img) # final for ESRGAN- Fine Tuned at face images
    #display(orig_Image, HR_Img_CodeFormer) ## for CoderFormer
    #pdb.set_trace()
   
    #displayThree(AlignedFace_img, HR_Img_CodeFormer, HR_Img_CodeFormer)

def enhance_image_api_method(image_path, model):
    orig_Image = read_cv2_img(image_path) # To be removed to make it consistent
    PiLImage = align_face(image_path) ## Face cropped image

    if PiLImage is not None:
        AlignedFace_img = np.array(PiLImage)# converting pil to numpy 
        output_img, HR_img = sr_forward(AlignedFace_img) ### ESRGAN ,retrained on Face Images
        
        HR_Img_CodeFormer1 = CodeFormerMain(AlignedFace_img,True, 0)  ##CodeFormer, alighmnet flag, fedility weight
        HR_Img_CodeFormer2 = CodeFormerMain(AlignedFace_img,True, 0.2)  ##CodeFormer, alighmnet flag, fedility weight
        HR_Img_CodeFormer3 = CodeFormerMain(AlignedFace_img,True, 0.5)  ##CodeFormer, alighmnet flag, fedility weight
        HR_Img_CodeFormer4 = CodeFormerMain(AlignedFace_img,True, 0.7)  ##CodeFormer, alighmnet flag, fedility weight
        HR_Img_CodeFormer5 = CodeFormerMain(AlignedFace_img,True, 1)  ##CodeFormer, alighmnet flag, fedility weight
        return convert_img(HR_Img_CodeFormer1), convert_img(HR_Img_CodeFormer4),convert_img(HR_Img_CodeFormer2),convert_img(HR_Img_CodeFormer3),convert_img(HR_Img_CodeFormer5)
    else: 
        ### testing code 
        HR_Img_CodeFormer1 = CodeFormerMain(image_path,False, 0)  ##CodeFormer
        #pdb.set_trace()
        if (len(HR_Img_CodeFormer1)==1):## means single  integer  not detected face by ratina face, so needs to be treated as generalized image 
            return None            
           # hr_image = preprocess_image(image_path)
           # lr_image = downscale_image(tf.squeeze(hr_image))
           # fake_image = model(lr_image)
           # fake_image = tf.squeeze(fake_image)
            #display(tf.squeeze(hr_image), tf.squeeze(fake_image))
        else:
            HR_Img_CodeFormer2 = CodeFormerMain(image_path,False, 0.2)  ##CodeFormer
            HR_Img_CodeFormer3 = CodeFormerMain(image_path,False, 0.5)  ##CodeFormer
            HR_Img_CodeFormer4 = CodeFormerMain(image_path,False, 0.7)  ##CodeFormer
            HR_Img_CodeFormer5 = CodeFormerMain(image_path,False, 1)  ##CodeFormer
            #img=np.array(faces[0]);
            return convert_img(HR_Img_CodeFormer1),convert_img(HR_Img_CodeFormer4),convert_img(HR_Img_CodeFormer2),convert_img(HR_Img_CodeFormer3),convert_img(HR_Img_CodeFormer5) 

# example to run the code
# # image path
# IMAGE_PATH = "C:\\images\\2.JPG"
#
# SAVED_MODEL_PATH = "C:/esrgan-tf2/1"
#
# enhance_image(IMAGE_PATH, SAVED_MODEL_PATH)
