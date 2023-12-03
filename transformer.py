# %%
import torch
from torch import nn
from torchvision.transforms import ToPILImage


# %%
class BuildPatches:
    """Build patches from images
        Accepts a tensor of shape (N, H, W) i.e. grayscale image and returns a tensor of shape (N, -1, patch_size**2)
    """
    def __init__(self, patch_size=4):
        self.patch_size = patch_size
        
    def __call__(self, img):
        N, H, W = img.shape
        assert (H % self.patch_size == 0) or (W % self.patch_size == 0), f"Height {H} or Width {W} not divisible by {self.patch_size}"
        return img.view(N, -1, self.patch_size**2)
# %%
class MiniVit(nn.Module):
    def __init__(self, input_dims, hidden_dims, num_heads, num_layers, output_classes, patch_size=4, img_dims=(28, 28)) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_classes = output_classes
        self.linear_emb = nn.Linear(input_dims, hidden_dims)
        self.patchify = BuildPatches()
        # self.pos_enc = nn.Parameter(torch.randn(1, hidden_dims))
        # classification token for classification task of image, 1 per hidden dim
        # reason for adding this is so that our transformer is not biased towards a particular token in the sequence
        self.class_token = nn.Parameter(torch.randn(1, hidden_dims))
        
        # positional encoding for every token of the image,
        # class token is at position 0
        # Why do we have to make it a parameter? 
        #   - afaik it is because althought we don't learn pos_enc but in future we might want to
        self.num_patches = (img_dims[0] // patch_size) * (img_dims[1] // patch_size)
        self.pos_enc = nn.Parameter(gen_pos_emb(self.num_patches+1, hidden_dims))
        self.pos_enc.requires_grad = False
        
        self.encoders = nn.ModuleList([VitEncoder(hidden_dims, 2) for _ in range(num_layers)])
        self.classfier = nn.Linear(hidden_dims, output_classes)
        
    def forward(self, x):
        tokens = self.linear_emb(self.patchify(x.squeeze())) # (N, L, D)
        # for every example in the batch, add the class token
        # the size of the token is same as the hidden dims
        x_cls = torch.stack([torch.cat((self.class_token, tokens[i]), dim=0) for i in range(tokens.shape[0])])
        x_enc = x_cls + self.pos_enc
        
        for encoder in self.encoders:
            x_enc = encoder(x_enc)
            
        return self.classfier(x_enc[:, 0, :])
        
        
class AttentionHead(nn.Module):
    def __init__(self, head_dims):
        super().__init__()
        self.keyw = nn.Linear(head_dims, head_dims)
        self.queryw = nn.Linear(head_dims, head_dims)
        self.valuew = nn.Linear(head_dims, head_dims)
    
    def forward(self, x):
        keys = self.keyw(x) # (N, L, D) @ (D, D) = (N, L, D)
        query = self.queryw(x)
        values = self.valuew(x)
        norm_factor = torch.sqrt(torch.tensor(keys.shape[-1])) # scaling factor
        weights = (1/norm_factor) * (query @ keys.permute(0, -1, -2)) # (N, L, D) @ (N, D, L) = (N, L, L)
        soft_weights = nn.functional.softmax(weights) 
        return x + (soft_weights @ values) # (N, L, L) @ (N, L, D) = (N, L, D)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dims, num_heads):
        super().__init__()
        self.emb_dims = emb_dims
        self.num_heads = num_heads
        self.head_dims = emb_dims // num_heads
        self.heads = nn.ModuleList([AttentionHead(self.head_dims) for _ in range(num_heads)])
    
    def forward(self, x): # x: (N, L, D)
        return torch.cat([head(x[:, :, i*self.head_dims: (i+1)*self.head_dims]) for i, head in enumerate(self.heads)], dim=-1)
    
class VitEncoder(nn.Module):
    def __init__(self, head_dims, num_heads, dims_to_mlp=4):
        super().__init__()
        self.msa = MultiHeadAttention(head_dims, num_heads)
        self.mlp = nn.Sequential(nn.Linear(head_dims, head_dims*dims_to_mlp), nn.GELU(), nn.Linear(head_dims*dims_to_mlp, head_dims))
    
    def forward(self, x): # x: (N, L, D)
        x = x + self.msa(nn.functional.layer_norm(x, x.shape[1:]))
        x = x + self.mlp(nn.functional.layer_norm(x, x.shape[1:]))
        return x
        

# %%
def gen_pos_emb(num_tokens, hidden_dims, n=10000):
    import math
    posemb = torch.zeros(num_tokens, hidden_dims)
    for t in range(num_tokens):
        for i in range(0, hidden_dims, 2):
            # calculate the internal value for sin and cos
            theta = t / (n ** ((i)/hidden_dims))
            posemb[t, i] = math.sin(theta)
            posemb[t, i+1] = math.cos(theta)
    return posemb
# %%