import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.ImageDraw as PImageDraw
from tqdm import tqdm
import glob
from PIL import Image
import sys
import time
import argparse
import jsonlines
from dataclasses import dataclass
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from autoregressive.modeling import ImgGen_HF,ImgGen_HF_CONFIG,ImgGenTextImageProcessor

@dataclass
class ARGS:
    exp_name: str = "AR-GRPO"
    ckpt_path: str = "YOUR CKPT PATH"
    sample_num: int = 1
    cfg_scale: float = 7.5
    top_k: int = 2000
    top_p: float = 1.0
    temperature: float = 1.0
    out_rt: str = "results/"
    delay_load_text_encoder: bool = True
    
args=ARGS()

print(args)

gen_cfg=dict(cfg_scale=7.5, top_k=2000, top_p=1.0, temperature=1.0)
for k in gen_cfg:
    if hasattr(args,k):
        gen_cfg[k]=getattr(args,k)
exp_name=args.exp_name
out_rt=args.out_rt

# download checkpoint
ckpt_path=args.ckpt_path

print("building models...")
model=ImgGen_HF.from_pretrained(ckpt_path,device="cuda",torch_dtype=torch.bfloat16,delay_load_text_encoder=args.delay_load_text_encoder)
if args.delay_load_text_encoder:
    model.load_text_encoder()
model.to(model.model_device)
print(next(model.vq_model.parameters()).device)
print(next(model.gpt_model.parameters()).device)
model.freeze_model()
processor=ImgGenTextImageProcessor(model.text_encoder,model.vq_model,model.config.codebook_embed_dim)
print(f'prepare finished.')
############################# 2. Sample with classifier-free guidance

# set args
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)

# seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# run faster
tf32 = True
torch.backends.cudnn.allow_tf32 = bool(tf32)
torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
torch.set_float32_matmul_precision('high' if tf32 else 'highest')

def batched_img(img_batch):
    for i in img_batch:
        img_np=i.permute(1,2,0).mul_(255).float().cpu().numpy()
        yield Image.fromarray(img_np.astype(np.uint8))

def save_png(img,img_p):
    img_dir=os.path.dirname(img_p)
    os.makedirs(img_dir,exist_ok=True)
    img.save(img_p)

# sample
exp_name=exp_name
exp_name=os.path.splitext(exp_name)[0]

all_data=[]
eval_path=os.path.join(args.eval_root,args.dataset,args.data_json)

texts = [
        "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
        "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grassin front of the Sydney Opera House holding a sign on the chest that says Welcome Friends!",
        "A blue Porsche 356 parked in front of a yellow brick wall.",
        "A photo of an astronaut riding a horse in the forest. There is a river in front of them with water lilies.",
        "A map of the United States made out of sushi. It is on a table next to a glass of red wine."
    ]

for idx,p in enumerate(tqdm(texts)): # for every prompt sample B images
    B = args.sample_num
    prompts = [p]*B
    data_dicts=[{"text":p} for p in prompts]
    input_data=processor(text=data_dicts)
    inputs,attn_mask=input_data["input_ids"],input_data["attention_mask"]
    with torch.inference_mode(),torch.device(model.model_device):
        with torch.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True):    # using bfloat16 can be faster
            x = model.generate(inputs.to("cuda"),attention_mask=attn_mask.to("cuda"), return_logits=False, **gen_cfg).output_ids
            x = processor.batch_decode(x)
            for jdx,j in enumerate(batched_img(x)):
                save_png(j,f"{out_rt}/{idx:>05}/samples/{jdx:>04}.png")
    with jsonlines.open(f"{out_rt}/{idx:>05}/metadata.jsonl","w") as f:
        f.write(i)
