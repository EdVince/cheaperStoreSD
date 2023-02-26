import torch
from torch.nn.parameter import Parameter

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("../chilloutmix", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate(filename):
    image = pipe(
        prompt="best quality, ultra high res, (photorealistic:1.4), 1girl, thighhighs, (big chest), (upper body), (Kpop idol), (aegyo sal:1), (platinum blonde hair:1), ((puffy eyes)), looking at viewer, facing front, smiling",
        negative_prompt="paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glan",
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(1234),
        height=512,width=512,
    ).images[0].save(filename)


def compress(weight, group=32, b=4):
    b = 2**b - 1

    fp_weight = weight
    size = fp_weight.shape

    int_weight = fp_weight.view(-1,group)

    int_min = int_weight.min(dim=1)[0].unsqueeze(1)
    int_max = int_weight.max(dim=1)[0].unsqueeze(1)

    int_weight = torch.round((int_weight-int_min)/(int_max-int_min)*b)
    
    back_weight = int_min + (int_max-int_min)*int_weight/b

    back_weight = back_weight.view(*size)

    return Parameter(back_weight)




generate('test_reference.png')


num = 0
for op in pipe.unet.modules():
    if isinstance(op, torch.nn.Linear):
        op.weight = compress(op.weight,32,8)
        num += 1
print('compress Linear [weight]:',num)
generate('test_int_linear[weight].png')


num = 0
for op in pipe.unet.modules():
    if isinstance(op, torch.nn.Linear):
        if op.bias is not None:
            op.bias = compress(op.bias,32,8)
            num += 1
print('compress Linear [weight bias]:',num)
generate('test_int_linear[weight_bias].png')


num = 0
for op in pipe.unet.modules():
    if isinstance(op, torch.nn.Conv2d):
        if op.weight is not None:
            op.weight = compress(op.weight,32,8)
            num += 1
print('compress Linear [weight bias] Conv2d [weight]:',num)
generate('test_int_linear[weight_bias]_conv2d[weight].png')