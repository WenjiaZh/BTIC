## BTIC
This repository configured for paper:
[**Supervised Contrastive Learning for Multimodal Unreliable News Detection in COVID-19 Pandemic**](https://arxiv.org/abs/2109.01850)

Wenjia Zhang, Lin Gui, Yulan He

In this work, we propose a BERT-based multimodal unreliable news detection framework, which captures both textual and visual information from unreliable articles utilising the contrastive learning strategy. The contrastive learner interacts with the unreliable news classifier to push similar credible news (or similar unreliable news) closer while moving news articles with similar content but opposite credibility labels away from each other in the multimodal embedding space. Experimental results on a COVID-19 related dataset, [ReCOVery](https://github.com/apurvamulay/ReCOVery), show that our model outperforms a number of competitive baseline in unreliable news detection.

<img src="https://github.com/WenjiaZh/BTIC/blob/main/ContrastiveFramework.png" width="500"><img src="https://github.com/WenjiaZh/BTIC/blob/main/case.png" width="500">

## Requirements
```
pandas
numpy
os
PIL
torch
space
sklearn
matplotlib
seaborn
```

## Citation
If you think our work is useful, please cite as:
```
@article{zhang2021supervised,
  title={Supervised Contrastive Learning for Multimodal Unreliable News Detection in COVID-19 Pandemic},
  author={Zhang, Wenjia and Gui, Lin and He, Yulan},
  journal={arXiv preprint arXiv:2109.01850},
  year={2021}
}
```
Thank you.

## Contact
If you have any question on the paper,or the code, please contact [wenjia.zhang@warwick.ac.uk](wenjia.zhang@warwick.ac.uk)
