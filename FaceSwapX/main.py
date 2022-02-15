#! /usr/bin/env python
import os
import cv2
import argparse
from FaceSwapX.face_detection import select_face, select_all_faces
from FaceSwapX.face_swap import face_swap


def swap_faces(src, dst, warp_2d=False, correct_color=True, no_debug_window=True):
  
  args = argparse.Namespace()
  args.src=src
  args.dst=dst
  args.out=''
  args.warp_2d=warp_2d
  args.correct_color=correct_color
  args.no_debug_window=no_debug_window
  
  src_img = cv2.imread(args.src)
  dst_img = cv2.imread(args.dst)
  # Select src face
  src_points, src_shape, src_face = select_face(src_img)
  # Select dst face
  dst_faceBoxes = select_all_faces(dst_img)

  if dst_faceBoxes is None:
    print('Detect 0 Face !!!')
    exit(-1)

  output = dst_img
  for k, dst_face in dst_faceBoxes.items():
    output = face_swap(src_face, dst_face["face"], src_points,
                        dst_face["points"], dst_face["shape"],
                        output, args)
  return output
  # dir_path = os.path.dirname(args.out)
  # if not os.path.isdir(dir_path):
  #   os.makedirs(dir_path)

  # cv2.imwrite(args.out, output)
