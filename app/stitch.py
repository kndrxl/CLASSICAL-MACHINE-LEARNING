
import os
import cv2
import numpy as np

from cv2 import DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS as features_flag
from cv2 import DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS as matches_flag
from app import logger
from app.utils import Utilities

class ImageStitcher:

  def __init__(self, folder_name, state):
    self.state=state
    self.folder_name = folder_name if self.state=='url' else 'sample_images'
    self.input_path = f"{os.getenv('SRC_PATH')}{self.folder_name}/input/"
    self.output_path = f"{os.getenv('SRC_PATH')}{self.folder_name}/output/"
    self.ratio = os.getenv('RATIO')
    self.utils = Utilities(self.folder_name, self.state)

  def stitch_images(self, path_r, path_l):
    rgb_l = cv2.cvtColor(cv2.imread(path_l), cv2.COLOR_BGR2RGB)
    rgb_r = cv2.cvtColor(cv2.imread(path_r), cv2.COLOR_BGR2RGB)
    rgb_l = cv2.cvtColor(rgb_l, cv2.COLOR_RGB2BGR)
    rgb_r = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR)
    feature_extractor = cv2.SIFT_create()
    kp_l, desc_l = feature_extractor.detectAndCompute(rgb_l, None)
    kp_r, desc_r = feature_extractor.detectAndCompute(rgb_r, None)
    features_l = cv2.drawKeypoints(rgb_l, kp_l, None, flags=features_flag)
    features_r = cv2.drawKeypoints(rgb_r, kp_r, None, flags=features_flag)
    self.utils.cv_writer(features_l, 'features')
    self.utils.cv_writer(features_r, 'features')
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_l, desc_r, k=2)
    matches_list = [m for m in matches if m[0].distance/m[1].distance < float(self.ratio)]
    im_matches = cv2.drawMatchesKnn(rgb_l, kp_l, rgb_r, kp_r, matches_list[0:30], None, flags=matches_flag)
    self.utils.cv_writer(im_matches, 'matches')
    good_match_arr = np.asarray(matches_list)[:,0]
    good_kp_l = np.array([kp_l[m.queryIdx].pt for m in good_match_arr])
    good_kp_r = np.array([kp_r[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_r, good_kp_l, cv2.RANSAC, 5.0)
    result = self.warp_images(rgb_l, rgb_r, H)
    self.utils.cv_writer(result, 'output')


  def warp_images(self, img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    result = cv2.warpPerspective(img2, Ht@H, (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    return result

  def main(self, images_path=[]):
    try:
      process = self.utils.download_images(images_path) if self.state == 'url' else None
      test_image = [file for file in os.listdir(self.input_path) if file.endswith(('.png','.jpg','.jpeg','.JPG','.JPEG'))]
      test_image.sort()
      for img in range(len(test_image)):
        if img != 1:
          path_l = f"{self.input_path}{test_image[img]}" if img==0 and img < 2 else f"{self.output_path}output.JPEG"
          path_r = f"{self.input_path}{test_image[img+1]}" if img==0 and img < 2 else f"{self.input_path}{test_image[img]}"
          print(f"Stitching images {path_l} and {path_r} ...")
          self.stitch_images(path_l, path_r)
      print(f"Done Stitching")
      return {"Message": f"Done stitching. Output is saved on this path: {os.getenv('SRC_PATH')}{self.folder_name}/output/ "}

      #### UNCOMMENT THESE OUT IF YOU WANT TO UPLOAD THE OUTPUT TO AN AWS S3 BUCKET"
      # upload = self.utils.upload_to_s3()
      # return upload
    except Exception as e:
      return {"Message": f"Error: {e}"}