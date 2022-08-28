import os

from app import logger
from app.stitch import ImageStitcher
from app.service import Service
from typing import Literal
from fastapi import FastAPI, Depends
from mangum import Mangum
from pydantic import BaseModel

serve = Service()

class Instance(BaseModel):
    url_list: list
    state: Literal['url', 'local']

app = FastAPI(
    title="Image Stitching",
    root_path=os.environ.get("OPENAPI_PREFIX", "")
)


#################### STITCHING USING IMAGE URLS #################### 
@app.post("/stitch/url")
def URL_generated(instance: Instance=Depends()):
    folder_name = serve.generate_folder_name()
    stitch = ImageStitcher(folder_name, instance.state)
    result = stitch.main(instance.url_list)
    return result

############### STITCHING USING IMAGES SAVED LOCALLY ###############
@app.post("/stitch/local_path")
def local_generated(instance: Instance=Depends()):
    folder_name = serve.generate_folder_name()
    stitch = ImageStitcher(folder_name, instance.state)
    result = stitch.main(instance.url_list)
    return result


handler = Mangum(app)