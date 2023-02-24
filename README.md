# cheaperStoreSD

Considering the Stable Diffusion is huge for disk, e.g. 2G with fp16. We follow the idea of [FlexGen](https://github.com/FMInference/FlexGen), trying to only do quantization in the storage stage. 

We quantization the weight and bias when saving to disk, we dequantize them once loading it. We still calculate the model in float!

## Setp

- [x] test the idea with torch
- [ ] implement it with ncnn