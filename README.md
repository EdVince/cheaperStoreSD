# cheaperStoreSD

考虑到Stable Diffusion的模型很大，用fp16存也要接近2G。这里参考[FlexGen](https://github.com/FMInference/FlexGen)的做法，把一些权重量化到int来存储，在加载模型的时候反量化回去float，计算仍然保持fp16计算。

目前经过尝试，大约可以从1.6G压到1.1G左右。

## Setp

- [x] test the idea with torch
- [x] implement it with ncnn

## Method
1. 魔改了ncnnoptimize的代码，给Convolution和Multiheadattention的weight增加了分组量化的功能。
2. SD-NCNN的代码里实现了加载时的反量化

## Problem
1. 目前经过尝试，大约在组大小为32，使用int8量化时，效果不算崩的太厉害，但torch里压到int4还是能出图的，百思不得其解
2. 目前的量化就是单纯的计算组内的minmax然后四舍五入取整，有点暴力了，误差比较大