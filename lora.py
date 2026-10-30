import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional , Literal , Union
from safetensors.torch import save_file


class LoRALayerBase:
    # this is a kinda layer where we'll be reusing all 
    # these methods for different stuffs

    def __init__(self , rank = 8 , lora_alpha = 8 , lora_dropout = 0.0 , use_rslora = True):
        # use Huggingface lib more settings to tweak or configure with 

        self.rank = rank
        self.lora_alpha = lora_alpha
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
        LoRALayerBase.__init__(self , rank = rank , lora_alpha = lora_alpha , lora_dropout = lora_dropout , use_rslora=use_rslora)

        # freeze weights of original layer and train new A and B matrices .
        # linear layer also has a bias , but training a bias/bias layer is optional.
        # we can toggle its implementation later .
        self.weight.requires_grad = False # ---> Turns off the weight layer and we don't do any gradient updates

        # we can train A & B matrices 2 ways :
        # 1. we can have 2 linear layers and no biases 
        # 2. we can have/create 2 matrices and pass them as paramters ans train them
        # we'll go the easy way!!!!!!!! 😂 -> way 2
        self.lora_A = nn.Parameter(torch.zeros(in_features , rank))
        self.lora_B = nn.Parameter(torch.zeros(rank , out_features))

        # in paper they initialized lora_A with gaussian distribution
        # but in implementation they used kaiming_uniform . idk why ?
        nn.init.kaiming_uniform_(self.lora_A , a = math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B , a = math.sqrt(5))

    def _merge_weights(self):
        # self.weight.data -> matrix(output , input) , because pytorch does the transpose
        # (self.lora_A @ self.lora_B) -> because we're doing it manually i just need to make sure   
        merged_weights = self.weight.data + self.scaling * (self.lora_A @ self.lora_B).T

        # create a new linear layer that uses W' the new matrix
        state_dict = {"weight" : merged_weights}
        if self.bias is not None:
            # if there's a bias in this layer then we can go ahead
            state_dict["bias"] = self.bias

            merged_linear = nn.Linear(self.in_features , self.out_features , bias=True if self.bias is not None else False)

            # merge our merged_weights and bias to this linear layer 
            merged_linear.load_state_dict(state_dict)

            return merged_linear

    def forward(self , x):
        # the actual logic 

        # the layer whose weights are being freezed 
        # we're using F.linear because it takes care of this multiplication . 
        # we used this just so that manual multiplication can be avoided
        org_layer_out = F.linear(x , self.weight , bias = self.bias)

        # the compressed layer 
        lora_mult = (self.lora_A @ self.lora_B) * self.scaling
        # we applied dropout just to reduce overfitting
        low_rank_out = self.lora_dropout(x) @ lora_mult

        output = org_layer_out + low_rank_out

        return output 
        
class LoRAEmbedding(nn.Embedding , LoRALayerBase):
    # its just a wrapper on nn.Embedding layer and LoRALayerBase

    # mostly embedding layers are frozen and not trained . but 
    # here we'll create a seperate lora embedding layer and 
    # potentially train it .

    def __init__(self , num_embeddings , embedding_dim , rank = 8 , lora_alpha = 8 , lora_dropout = 0.0 , use_rslora = True , **kwargs):
        # num_embeddings = vocab size , how many possible words do i've in my tokenizer
        # embedding_dim = how many numbers will i use to represent every single word 

        # initialize inherited classes
        nn.Embedding.__init__(self , num_embeddings , embedding_dim , **kwargs)
        LoRALayerBase.__init__(self , rank=rank , lora_alpha=lora_alpha , lora_dropout=lora_dropout , use_rslora=use_rslora)

        # we don't need to compute gradients on the original embedding 
        self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(num_embeddings , rank))
        self.lora_B = nn.Parameter(torch.zeros(rank , embedding_dim))

        nn.init.kaiming_uniform_(self.lora_A , a = math.sqrt(5))

    def forward(self , x):
        # linear layer has dropout here we don't have that because here we're using raw inputs 
        # i.e. x
        org_layer_out = F.embedding(input=x , weight=self.weight , padding_idx=self.padding_idx , max_norm=self.max_norm , norm_type=self.norm_type , scale_grad_by_freq=self.scale_grad_by_freq , sparse=self.sparse)
        low_rank_A_output = F.embedding(input=x , weight=self.lora_A , padding_idx=self.padding_idx , max_norm=self.max_norm , norm_type=self.norm_type , scale_grad_by_freq=self.scale_grad_by_freq , sparse=self.sparse)
        low_rank_output = (low_rank_A_output @ self.lora_B) * self.scaling
        output = org_layer_out + low_rank_output

        return output 
        
 
if __name__ == "__main__":
    # whether or not we do LoRA operation or we merge the weights 
    # and just do regular linear linear layer we get the same result .

    layer = LoRAEmbedding(200 , 64 , rank = 2)
    # rand = torch.rand(4,4)
    # print(layer)

    # output = layer(rand)
    # print(output)

    # when we do weight merging the new linear layer that we get from 
    # weight merging should have exactly same dim as the layer before
    # doing weight merging . 
    # merged = layer._merge_weights()
    # output2 = merged(rand)
    # print(output2)


        