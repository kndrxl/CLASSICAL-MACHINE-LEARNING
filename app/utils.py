import os
import shutil
import boto3
import cv2
import numpy as np
import urllib.request

from pathlib import Path
from datetime import datetime
from app import logger


class Utilities:
    def __init__(self, folder_name, state):
        self.state=state
        self.folder_name = folder_name if self.state=='url' else 'sample_images'
        self.s3 = boto3.client("s3")
        self.ratio = os.getenv('RATIO')
        self.input_path = f"{os.getenv('SRC_PATH')}{folder_name}/input/"
        self.output_path = f"{os.getenv('SRC_PATH')}{folder_name}/output/"
        self.features_path = f"{os.getenv('SRC_PATH')}{folder_name}/features/"
        self.matches_path = f"{os.getenv('SRC_PATH')}{folder_name}/matches/"


    def download_images(self, images_urls):
        if  not os.path.exists(self.input_path):
            Path(self.input_path).mkdir(parents=True, exist_ok=True)
        for i in range(len(images_urls)):
            print(f"Downloading {images_urls[i]}")
            req = urllib.request.urlopen(images_urls[i])
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            cv2.imwrite(f"{self.input_path}{i}.{images_urls[i].split('.')[-1]}", cv2.imdecode(arr, -1))
        print(os.listdir(self.input_path))


    def cv_writer(self, file, type):
        now = datetime.now()
        category = {
            'features': {
                "dir": f"{self.features_path}", 
                "name": f"{now.strftime('%H.%M.%S.%f')}.JPEG"
            },
            'matches': {
                "dir": f"{self.matches_path}",
                "name": f"{now.strftime('%H.%M.%S.%f')}.JPEG"
            },
            'output': {
                "dir": f"{self.output_path}",
                "name": "output.JPEG"
            }
        }
        if not os.path.exists(category[type]['dir']):
            os.mkdir(category[type]['dir'])
        file_name = f"{category[type]['dir']}{category[type]['name']}"
        cv2.imwrite(file_name, file)

    
################## FUNCTIONS FOR UPLOADING THE OUTPUT TO AN S3 BUCKET ###########################

    def clean_up(self):
        print('Cleaning Up Folders...')
        delete_output = shutil.rmtree(self.output_path) if os.path.exists(self.output_path) else 1
        delete_input = shutil.rmtree(self.input_path) if os.path.exists(self.input_path) else 1
        delete_features = shutil.rmtree(self.features_path) if os.path.exists(self.features_path) else 1
        delete_matches =  shutil.rmtree(self.matches_path) if os.path.exists(self.matches_path) else 1
        print('Done Clean Up')


    def upload_to_s3(self):
        now = datetime.now()
        os.rename(
            f"{self.output_path}output.JPEG", 
            f"{self.output_path}{now.strftime('%H.%M.%S')}.JPEG"
        )
        output_list = os.listdir(self.output_path)
        output_key = f"{os.getenv('OUTPUT_PREFIX')}/{now.strftime('%Y/%m/%d')}/{output_list[0]}"
        print('Uploading Ouput to S3...')
        self.put_object(f"{self.output_path}{output_list[0]}", output_key)
        url = f"{os.getenv('IMAGE_URL')}{output_key}"
        print('Uploaded Successfully.') 
        return {
            "output_url": url
        }


    def put_object(self, file, key):
        self.s3.put_object(
            Bucket = os.getenv("BUCKET_NAME"),
            Key = key,
            Body = open(file, "rb").read()
        )

