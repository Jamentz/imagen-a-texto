#@title Setup
import os, subprocess

def setup():
    install_cmds = [
        ['pip', 'install', 'ftfy', 'gradio', 'regex', 'tqdm', 'transformers==4.21.2', 'timm', 'fairscale', 'requests'],
        ['pip', 'install', 'open_clip_torch'],
        ['pip', 'install', '-e', 'git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip'],
        ['git', 'clone', '-b', 'open-clip', 'https://github.com/pharmapsychotic/clip-interrogator.git']
    ]
    for cmd in install_cmds:
        print(subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8'))

setup()

# download cache files
print("Download preprocessed cache files...")
CACHE_URLS = [
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_artists.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_flavors.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_mediums.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_movements.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_trendings.pkl',
]
os.makedirs('cache', exist_ok=True)
for url in CACHE_URLS:
    print(subprocess.run(['wget', url, '-P', 'cache'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

import sys
sys.path.append('src/blip')
sys.path.append('clip-interrogator')

import gradio as gr
from clip_interrogator import Config, Interrogator

config = Config()
config.blip_offload = True
config.chunk_size = 2048
config.flavor_intermediate_count = 512
config.blip_num_beams = 64

ci = Interrogator(config)

def inference(image, mode, best_max_flavors):
    image = image.convert('RGB')
    if mode == 'best':
        return ci.interrogate(image, max_flavors=int(best_max_flavors))
    elif mode == 'classic':
        return ci.interrogate_classic(image)
    else:
        return ci.interrogate_fast(image)

inputs = [
    gr.inputs.Image(type='pil'),
    gr.Radio(['best', 'classic', 'fast'], label='', value='best'),
    gr.Number(value=4, label='best mode max flavors'),
]
outputs = [
    gr.outputs.Textbox(label="Output"),
]

io = gr.Interface(
    inference, 
    inputs, 
    outputs, 
    allow_flagging=False,
)
io.launch()