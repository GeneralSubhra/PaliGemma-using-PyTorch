from typing import Dict,List,Optional,Union,Tuple,Iterable
import numpy as np
from PIL import Image 
import torch

IMAGENET_STD_MEAN = [0.5,0.5,0.5]
IMAGENET_STD_STD = [0.5,0.5,0.5]

class PaliGemmaProcessor:
    def __init__(self,tokenizer,num_img_tokens:int,img_size:int):
        super().__init__()
        self.img_seq_len = num_img_tokens
        self.img_size=img_size
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
            ] #token used for object detection bounding box
        