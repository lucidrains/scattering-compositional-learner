## Scattering Compositional Learner

Implementation of <a href="https://arxiv.org/abs/2007.04212">Scattering Compositional Learner</a>, which reached superhuman levels on Raven's Progressive Matrices, a type of IQ test for analogical reasoning.

This repository is meant to be exploratory, so it may not follow the exact architecture of the paper down to the T. It is meant to find the underlying inductive bias that could be exported for use in attention networks. The paper suggests this to be the 'Scattering Transform', which is basically  grouped convolutions but where each group is tranformed by one shared neural network.

If you would like the exact architecture used in the paper, the <a href="https://github.com/dhh1995/SCL">official repository is here</a>.

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
