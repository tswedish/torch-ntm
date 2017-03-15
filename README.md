A Neural Turing Machine in Torch
================================

This is a fork of torch-ntm from kaishengtai/torch-ntm. It adds a batch mode, LRUA module layers, and some explicit CUDA support.

NOTE: Usage below has not yet been updated, instead try:

```
th tasks/one_shot.lua
```
This new example is a toy task that follows the Learning to Learn approach followed in this [paper](https://arxiv.org/abs/1605.06065) by Santoro et al.

A Torch implementation of the Neural Turing Machine model described in this 
[paper](http://arxiv.org/abs/1410.5401) by Alex Graves, Greg Wayne and Ivo Danihelka.

This implementation uses an LSTM controller. NTM models with multiple read/write heads are supported.

## Requirements

[Torch7](https://github.com/torch/torch7) (of course), as well as the following
libraries:

[penlight](https://github.com/stevedonovan/Penlight)

[nn](https://github.com/torch/nn)

[optim](https://github.com/torch/optim)

[nngraph](https://github.com/torch/nngraph)

All the above dependencies can be installed using [luarocks](http://luarocks.org). For example:

```
luarocks install nngraph
```

## Usage

For the copy task:

```
th tasks/copy.lua
```

For the associative recall task:

```
th tasks/recall.lua
```
