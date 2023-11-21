# On the Reliability and Explainability of Language Models for Program Generation

## Overview
This repository contains all the necessary materials for replicating the research conducted in our study "[On the Reliability and Explainability of Language Models for Program Generation](https://arxiv.org/abs/2302.09587)." The study focuses on assessing the reliability and explainability of various language models in the context of program generation tasks. Our study reveals significant flaws in model performance and uncovers severe data duplication, leading to over-optimistic results. Our findings highlight the critical need for more rigorous evaluation methods and benchmarks to enhance the reliability and explainability of these models in practical applications.


## Datasets Used in the Study

| Dataset Name     | Task Types             | Venus             | Paper Link and Source Code             |
|------------------|------------------------|-------------------------------------|------------------------|
| Tufano et al.    | Code Review   | ICSE'19                  | [[data]](https://sites.google.com/view/learning-codechanges), [[Paper]](https://dl.acm.org/doi/10.1145/3340544)
| Bugs2Fix         | Code Repair    | TOSEM'19                   | [[data]](https://github.com/microsoft/CodeXGLUE), [[Paper]](https://dl.acm.org/doi/10.1109/ICSE.2019.00021)
| CodeReview       | Code Review   | FSE'22                   | [[data]](https://zenodo.org/records/6900648#.Y3TVtOzP30o), [[Paper]](https://dl.acm.org/doi/abs/10.1145/3540250.3549081)
| CodeTrans-Dataset| Code Translation    | NIPS'19                   | [[data]](https://github.com/microsoft/CodeXGLUE), [[Paper]](https://arxiv.org/abs/2102.04664)
| CONCODE          | Code Generation    | NIPS'19                   | [[data]](https://github.com/microsoft/CodeXGLUE), [[Paper]](https://arxiv.org/abs/2102.04664)



## Contents
- `Data`: Folder containing datasets used in the study.
- `Models`: Folder with the pre-trained models and fine-tuning scripts.
- `Analysis`: Jupyter notebooks or scripts for data analysis and results reproduction.
- `Figures`: Generated graphs and figures as seen in the paper.
- `Documentation`: Additional documentation and details about the project.

## Getting Started
To get started with replicating our study, please follow the steps below:

### Prerequisite and Setup

- Python 3.6 +
- Packages:

```shell
pip install -r requirements.txt
```

Choose a diretory and:

```shell
git clone https://github.com/yueyueL/ProgramGen-LMs-Reliability.git

cd ProgramGen-LMs-Reliability/
```

## Contribution
We welcome contributions and suggestions! Please open an issue or submit a pull request for any enhancements.

## Citation
If you use the resources provided in this repository, please cite our paper
```
@misc{liu2023reliability,
      title={On the Reliability and Explainability of Language Models for Program Generation}, 
      author={Yue Liu and Chakkrit Tantithamthavorn and Yonghui Liu and Li Li},
      year={2023},
      eprint={2302.09587},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```

