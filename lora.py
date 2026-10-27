import math
import torch
import torch.nn as nn
import torch.functional as F
from dataclasses import dataclass
from typing import Optional , Literal , Union
from safetensors.torch import save_file


class LoRALayerBase:
    # this is a kinda layer where we'll be reusing all 
    # these methods for different stuffs

    def __init__(self , rank = 8 , lora_alpha = 8 , lora_dropout = 0.0 , use_rslora = True):
        # use Huggingface lib more settings to tweak or configure with 

        self.rank = rank
        self.lora_alpha = self.lora_alpha
        # if we don't want dropout then takes the input only
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x : x
        self.use_rslora = use_rslora
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank

    # this method will get used at very end at the time of loading weights 
    def load_pretrained_weights(self , state_dict):
        # we don't want lora linear layer . we want to apply lora linear
        # to an already pretrained layer . this copies the weight from 
        # original linear layer of the pretrained model to our 
        # lora linear layer .

        # this weight class doesn't exists yet , but it will exist
        # when we write our layer class .
        self.weight.data = state_dict["weight"]
        if "bias" in state_dict.keys():
            self.bias.data = state_dict["bias"]
            
class LoRALinearLayer(nn.Linear , LoRALayerBase):
    # its just a regular linear layer , but we've just given a name to it
    # which is LoRALinearLayer . 
    # its basically a wrapper on nn.Linear module . and it also
    # inherits LoRALayerBase . there are multiple inheritance happening
    # here but it doesn't change anything . we can add few features here
    # which a regular linear layer has .

    def __init__(self, in_features, out_features, bias = True, rank = 8 , lora_alpha = 8 , lora_dropout = 0.0 , use_rslora = True , **kwargs):
        nn.Linear.__init__(self , in_features , out_features , bias = bias , **kwargs)

        # initialize this layer so that we can resue all the methods of this layer
        # from within lora linear layer also.
        LoRALayerBase(self , rank = rank , lora_alpha = lora_alpha , lora_dropout = lora_dropout , use_rslora=use_rslora)

    

if __name__ == "__main__":
    layer = LoRALinearLayer(4 , 4)
    print(layer)


        