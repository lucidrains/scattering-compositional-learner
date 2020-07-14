<img src="./scattering.png" width="600px"></img>

## Scattering Compositional Learner

Implementation of <a href="https://arxiv.org/abs/2007.04212">Scattering Compositional Learner</a>, which reached superhuman levels on Raven's Progressive Matrices, a type of IQ test for analogical reasoning.

This repository is meant to be exploratory, so it may not follow the exact architecture of the paper down to the T. It is meant to find the underlying inductive bias that could be exported for use in attention networks. The paper suggests this to be the 'Scattering Transform', which is basically  grouped convolutions but where each group is tranformed by one shared neural network.

If you would like the exact architecture used in the paper, the <a href="https://github.com/dhh1995/SCL">official repository is here</a>.

## Use

```python
import torch
import torch.nn.functional as F
from scattering_transform import SCL

# data

questions = torch.randn(1, 8, 160, 160)   # 8 questions
answers   = torch.randn(1, 8, 160, 160)   # 8 possible answers
labels    = torch.tensor([2])             # correct answer is 2

answers   = answers[:, :, None, :, :]
questions = questions[:, None, :, :, :].expand(-1, 8, -1, -1, -1)

# the network looks at all permutations of questions to each answer

possibilities = torch.cat((questions, answers), dim=2)

# instantiate model

model = SCL(
    image_size = 160,
    set_size = 9,
    conv_channels = [1, 16, 16, 32, 32, 32],
    conv_output_dim = 80,
    attr_heads = 10,
    attr_net_hidden_dims = [128],
    rel_heads = 80,
    rel_net_hidden_dims = [64, 23, 5]
)

logits = model(possibilities)

# train

loss = F.cross_entropy(logits, labels)
loss.backward()
```

## Citation

```bibtex
@misc{wu2020scattering,
    title={The Scattering Compositional Learner: Discovering Objects, Attributes, Relationships in Analogical Reasoning},
    author={Yuhuai Wu and Honghua Dong and Roger Grosse and Jimmy Ba},
    year={2020},
    eprint={2007.04212},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
