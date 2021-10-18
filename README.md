# Meta Two-Sample Testing

Hi, this is the core code of our rencent work "Meta Two-Sample Testing: Learning Kernels for Testing with Limited Data" (NeurIPS2021, https://arxiv.org/abs/2106.07636). This work is done by 

- Dr. Feng Liu (UTS), feng.liu@uts.edu.au
- Dr. Wenkai Xu (Gatsby Unit, UCL), wenkaix@gatsby.ucl.ac.uk
- Prof. Jie Lu (UTS), jie.lu@uts.edu.au
- Dr. Danica J. Sutherland (UBC), djs@djsutherland.ml.


# Software version
Torch version is 1.1.0. Python version is 3.8.5. CUDA version is 10.1.

For the implementation of most baselines, please refer to https://github.com/fengliu90/DK-for-TST.

These python files, of cause, require some basic scientific computing python packages, e.g., numpy. I recommend users to install python via Anaconda (python 3.8.5), which can be downloaded from https://www.anaconda.com/distribution/#download-section . If you have installed Anaconda, then you do not need to worry about these basic packages.

After you install anaconda and pytorch (gpu), you can run codes successfully.

# Meta kernel learning

Please feel free to test the Meta-KL testing method by running main_metaKL.py.

Specifically, please run

```
CUDA_VISIBLE_DEVICES=0 python main_metaKL.py
```

in your terminal (using the first GPU device).

It can be seen that, test power on the target task will be significantly improved after introducing related tasks.


# Citation
If you are using this code for your own researching, please consider citing
```
@article{liu2021meta,
  title={Meta Two-Sample Testing: Learning Kernels for Testing with Limited Data},
  author={Liu, Feng and Xu, Wenkai and Lu, Jie and Sutherland, Danica J.},
  journal={arXiv preprint arXiv:2106.07636},
  year={2021}
}
```

# Acknowledgment
FL and JL are supported by the Australian Research Council (ARC) under FL190100149. WX is supported by the Gatsby Charitable Foundation. FL would also like to thank Dr. Yanbin Liu for productive discussions.
