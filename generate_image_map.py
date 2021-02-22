import os, sys
import numpy as np
import scipy.io as sio
from skimage import io
from skimage import img_as_ubyte
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import dlib
import argparse

import face3d.face3d
from face3d.face3d import mesh_numpy
from face3d.face3d.morphable_model import MorphabelModel



def main(args):
    with open(args.img_list) as f:
        img_list = [x.strip() for x in f.readlines()]
    landmark_list = []
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.save_lmk_dir):
        os.mkdir(args.save_lmk_dir)

    for img_idx, img_fp in enumerate(tqdm(img_list)):
        im = cv2.imread(os.path.join(args.img_prefix, img_fp), 1)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        h, w, c = im.shape

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("face3d/examples/models/shape_predictor_68_face_landmarks.dat")

        rects = detector(gray, 1)
        shape = predictor(gray, rects[0])
        tl_corner_x = rects[0].center().x - rects[0].width()/2
        tl_corner_y = rects[0].center().y - rects[0].height()/2
        br_corner_x = rects[0].center().x + rects[0].width()/2
        br_corner_y = rects[0].center().y + rects[0].height()/2
        rects = [(tl_corner_x, tl_corner_y), (br_corner_x, br_corner_y)]
        landmarks = np.zeros((68, 2))

        for i, p in enumerate(shape.parts()):
            landmarks[i] = [p.x, p.y]
            im = cv2.circle(im, (p.x, p.y), radius=3, color=(0, 0, 255), thickness=5)

        bfm = MorphabelModel('face3d/examples/Data/BFM/Out/BFM.mat')
        x = mesh_numpy.transform.from_image(landmarks, h, w)
        X_ind = bfm.kpt_ind


        fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter=200, isShow=False)
        colors = bfm.generate_colors(np.random.rand(bfm.n_tex_para, 1))
        colors = np.minimum(np.maximum(colors, 0), 1)

        fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
        transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
        image_vertices = mesh_numpy.transform.to_image(transformed_vertices, h, w)

        triangles = bfm.triangles
        colors = colors/np.max(colors)


        attribute = colors
        color_image = mesh_numpy.render.render_colors(image_vertices, triangles, attribute, h, w, c=3)
        io.imsave(os.path.join(args.save_lmk_dir, str(img_idx) + ".png"), img_as_ubyte(color_image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')

    parser.add_argument('--bbox_init', default='two', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--save_dir', default='results_2', type=str, help='dir to save result')
    parser.add_argument('--img_list', default='results/param/file_list.txt', type=str, help='test image list file')
    parser.add_argument('--save_lmk_dir', default='results_2', type=str, help='dir to save landmark result')

    parser.add_argument('--img_prefix', default='results/param', type=str, help='test image prefix')

    args = parser.parse_args()
    main(args)

