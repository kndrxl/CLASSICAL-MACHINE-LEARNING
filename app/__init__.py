
from dotenv import load_dotenv, find_dotenv
import logging

logger = logging.getLogger('Image Stitcher')
logging.getLogger('botocore').setLevel(logging.ERROR)

try:
    load_dotenv(find_dotenv())
except:
    ...