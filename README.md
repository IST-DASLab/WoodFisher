# WoodFisher 
This repository accompanies the NeurIPS 2020 paper on WoodFisher: Efficient Second-Order Approximation for Neural Network Compression
 (Singh & Alistarh, 2020). 
 
 <img src="https://github.com/IST-DASLab/WoodFisher/blob/main/woodfisher_camera_ready_uncompressed.png" width="500">

 The code is based on PyTorch and has been tested on version `1.2.0+cu92`.

#### Full-WoodFisher pruners

* `woodfisher`: Full WoodFisher, computes both off-diagonal and diagonal of the Hessian inverse.
* `woodtaylor`: Full WoodTaylor, computes both off-diagonal and diagonal of the Hessian inverse. In addition to WoodFisher (which is based on the second-order term of the Taylor series), it also considers the first-order term (i.e., the one containing the gradient) in Taylor series. 

#### Blockwise-WoodFisher pruners
* `woodfisherblock`: Same as `woodfisher` except that it only considers block of size `--fittable-params` along the diagonal.
* `woodtaylorblock`: Same as `woodtaylor` except that it only considers block of size `--fittable-params` along the diagonal. 

### Additional Blockwise-WoodFisher-based pruners
* `woodfisherblockdynamic`: The functionality is same as `woodfisherblock` but is more scalable. Instead of zeroing the inactive/pruned parameters while estimating the Hessian inverse during the gradual pruning, it only maintains the active parameters. This is done by storing their indices, can be thought of as doing computations based on the adjacency-graph representation instead of the Hessian matrix.

* `woodfisherblock_flops`: Takes into account the FLOP costs while deciding whether to prune a parameter or not. It comes into two modes: one where you can specify the `--flops-power` to reweigh the pruning statistic, other directly takes in the level of FLOPs you require for the final model via `--flops-target`. 
                                 
#### Baseline pruners

* `magni`: Magnitude-based automated gradual pruner
* `globalmagni`: Global-Magnitude-based automated gradual pruner
* `naivehess`: Naive Hessian-based automated gradual pruner (computes the diagonal of the hessian via double backprop)
* `diagfisher`: Diagonal Fisher-based automated gradual pruner
* `kfac`: K-FAC-based automated gradual pruner (blockwise with respect to layers)

## Structure of the repo

* `main.py` is the main file to run pruning from. You will need to provide data and config paths and specify dataset and architecture names.
* `configs/` contains yaml config files we use for specifying training and pruning schedules.
* `models/` directory contains currently available models. To add a new model, have it be loaded by `get_model` function in `models/__init__.py`.
* `policies/` contains `pruner`-specific policies, `trainer`-specific policies, and a `Manager` class which executes these policies as specified in a given config.
* `tests/` contain some tests for the above functionalities. 
* `utils/` contains utilities for loading datasets, masking layers for pruning, and performing helper computations.
* `scripts/` contains bash scripts for executing the corresponding `WoodFisher` results mentioned in the paper. Some examples for `Global Magnitude` are also provided, but you can get the `Global Magnitude` version of the other scripts by simply setting the `PRUNER` flag in the bash file: `PRUNER=globalmagni`.

## Args and config.yaml description

We use a yaml config file to specify all things related to training and pruning: choice of pruners and optimizers, scheduling, and so on. Writing this file is best explained with an example, which you can find in `configs/example_config.yaml`. As shown, the file should be split into two top-level sections: `pruners` and `trainers` (in any order). Second-level entries for both `pruners` and `trainers` list names of instances that are used to refer to them from the runner scripts. These names can be arbitrarily chosen by the user; however, if you are using only one model for training and pruning (which is most often the case), it is easiest to name your single trainer `default_trainer` to avoid having to specify a custom name for the runner script. For example, `example_config.yaml` has two pruners, `pruner_1` and `pruner_2`, and a `default_trainer`.

Also to allow an easier way to use, you can update the config file on the fly via command line arguments. In order to do so, `--update-config` must be passed, and besides that whatever argument that you might want to override, e.g., setting weight decay to zero `--prune-wdecay 0`, etc.
 
## Pruning Schedule
For the results, in the paper, we fix a budget of 100 epochs for the entire gradual pruning process (i.e., including finetuning). However, you might want to play around with this schedule depending on your needs, to either compress or lengthen the schedule. 
This can be achieved by using the flags `--prune-start`, `--prune-freq`, `--prune-end`, `--epochs`.

More details on these flags can be found in the argument descriptions contained in the `options.py` file. 

## Other useful flags

* `--disable-wdecay-after-prune`, bool: It disable weight decay after the pruning part has finished and retraining remains to be done. Empirically, this seems to further improve the performance, beyond the numbers reported in the paper as well. 
*  `--scale-prune-update`, float: This scales the readjustment of the other weights in the OBS (optimal brain surgeon) formulation. Default value is `1.0`. Setting to `0` should enable OBD (optimal brain damage) formulation instead of OBS. You can also play around with some other values. 
* `--label-smoothing`, float: This enables label smoothing in the loss. Default value is `0`, so label smoothing is disabled. Our reported results don't use label smoothing, but one can also try this to get slightly more improvements. 
* Recomputation of Hessian: This is again an additional feature to break down a single pruning step into multiple parts, whereupon after each part the inverse Hessian is recomputed. The motivation behind this is that the second order Taylor series approximation holds only in a small neighborhood of the current parameters. To use this, you have to append some additional flags such as the ones mentioned below, besides the usual command. 
```
--recompute-num 2 --recompute-schedule poly --recompute-degree 2

```

## Memory-management
At some point if you are pruning moderate to large-sized networks, there are some additional things you might want to take care off, in order to be able to run `WoodFisher`-based pruning variants. The below-mentioned flags will likely be useful in this scenario:

`--fittable-params`: This is one of the key flags that controls the block/chunk size to consider along the diagonal of the hessian. If this chunk size is `c`, then the memory consumed is of the order `O(cd)`, where `d` denotes the total number of parameters in the model. Hence choose `c` based on the model size and the amount of memory available at hand. E.g., for `ResNet50` which has `~25M` params, a chunk size of 1000-2000` should be good, while for `MobileNetV1` which has `~4.5M` params, a decent chunk size would be between `5000-20000`.

`--offload-inv`: This offloads the hessian inverse, computed during an intermediate step during pruning, on to the CPU.

`--offload-grads`: This offloads the loss gradients, computed over the various samples, on to the CPU.

`--fisher-cpu`: Shifts even the computation of some parts onto the CPU.

`--fisher-split-grads`: Tradeoff speed and memory while doing the Woodburry updates on the GPU. This is not required if you are using the CPU anyways. The flag expects an integer argument via an additional flag `--fisher-parts, and reduces the memory requirement from `2 * Hessian size` to `(1+ 1/k)* Hesian size` where `k` stands for the value of this *fisher-parts* flag (default value is 5).

### 


 
## Setup

First, clone the repository by running the following command:
```bash
$ git clone https://github.com/IST-DASLab/WoodFisher
```
After that, do not forget to run `source setup.sh`.

### Tensorboard Support

First of all ensure that your `torch` package has version 1.1.0 or above. Then install the nightly release of `tensorboard`:

```
$ pip install tb-nightly
```

After that ensure that `future` package is installed or invoke installation process by typing the following command in terminal:

```
$ pip install future
```

## Extensions and Pull Requests

We welcome the contributions of community in further enriching the current codebase, from the standpoint of improving efficiency to adding support for additional network types as well as to matters concerning the aesthetics. Feel free to send a pull request in such a scenario, possibly alongside the csv file generated by running the tests contained in the `tests/` folder. Also, an example `results.csv` with which you can match your results to see if everything is still alright (these results should be rough match as the exact numbers depend on the PyTorch versions, the inherent randomness across platforms, etc.).

Some example's of pull requests are labelled in the code as *PR*. 
## Acknowledgements

We thank Alex Shevchenko, Ksenia Korovina for providing an initial framework that we could re-purpose for the implementation of our work.

## Reference
This codebase corresponds to the paper: *WoodFisher: Efficient Second-Order Approximation for Neural Network Compression*. If you use any of the code or provided models for your research, please consider citing the paper as.
```
@inproceedings{NEURIPS2020_d1ff1ec8,
 author = {Singh, Sidak Pal and Alistarh, Dan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {18098--18109},
 publisher = {Curran Associates, Inc.},
 title = {WoodFisher: Efficient Second-Order Approximation for Neural Network Compression},
 url = {https://proceedings.neurips.cc/paper/2020/file/d1ff1ec86b62cd5f3903ff19c3a326b2-Paper.pdf},
 volume = {33},
 year = {2020}
}

```
