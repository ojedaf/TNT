# TNT: Text-Conditioned Network with Transductive Inference for Few-Shot Video Classification

[[Blog]](https://ojedaf.github.io/tnt_site/) [[Paper]](https://www.bmvc2021-virtualconference.com/conference/papers/paper_1200.html)

Recently, few-shot video classification has received an increasing interest. Current approaches mostly focus on effectively exploiting the temporal dimension in videos to improve learning under low data regimes. However, most works have largely ignored that videos are often accompanied by rich textual descriptions that can also be an essential source of information to handle few-shot recognition cases. In this paper, we propose to leverage these human-provided textual descriptions as privileged information when training a few-shot video classification model. Specifically, we formulate a text-based task conditioner to adapt video features to the few-shot learning task. Furthermore, our model follows a transductive setting to improve the task-adaptation ability of the model by using the support textual descriptions and query instances to update a set of class prototypes. Our model achieves state-of-the-art performance on four challenging benchmarks commonly used to evaluate few-shot video action classification models.

![tnt-model](https://github.com/ojedaf/TNT/blob/main/img/full_model_img.png)


## Content

- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Usage](#usage)
- [Citation](#citation)

## Prerequisites

It is essential to install all the dependencies and libraries needed to run the project.

```
pip install -r requirements.txt
```

## Dataset

We provide the metadata of each dataset. This metadata contains the meta-train set, meta-val set, and meta-test set. However, you have to download the datasets that you will use. 

## Usage



## Citation

If you find this repository useful for your research, please consider citing our paper:

```
@inproceedings{tnt_villa,
  author    = {Andr{\'{e}}s Villa and
               Juan{-}Manuel Perez{-}Rua and
               Vladimir Araujo and
               Juan Carlos Niebles and
               Victor Escorcia and
               Alvaro Soto},
  title     = {{TNT:} Text-Conditioned Network with Transductive Inference for Few-Shot
               Video Classification},
  booktitle = {The British Machine Vision Conference (BMVC)},
  year={2021},
  month={November}
}
```

For any questions, welcome to create an issue or contact Andrés Villa (afvilla@uc.cl).
