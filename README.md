# cfp

#### [[Project Website]](https://dittogym.github.io/) [[Paper]](https://arxiv.org/pdf/2401.13231.pdf)

[Suning Huang<sup>1</sup>](https://suninghuang19.github.io/), [Boyuan Chen<sup>2</sup>](https://boyuan.space/), [Huazhe Xu<sup>1</sup>](http://hxu.rocks//), [Vincent Sitzmann<sup>2</sup>](https://www.vincentsitzmann.com/) <br/>
<sup>1</sup>Tsinghua <sup>2</sup>MIT </br>

<p align="center">
  <img src="./teaser/M.gif" alt="M" style="display: inline-block;" />
  <img src="./teaser/I.gif" alt="I" style="display: inline-block;" />
  <img src="./teaser/T.gif" alt="T" style="display: inline-block;" />
</p>


This repo is the official algorithm implementation of ***[DittoGym: Learning to Control Soft Shape-Shifting Robots](https://dittogym.github.io/)***. We set up a fully-convolutional policy framework with **coarse-to-fine curriculum** to efficiently train control policies for highly reconfigurable robots to support locomotion and fine-grained morphology changes. You can also check the implementation of the benchmark **DittoGym** in ***[here](https://github.com/suninghuang19/dittogym)***.

If you find this work helpful to your research, please cite us as:

```
  @misc{huang2024dittogym,
    title={DittoGym: Learning to Control Soft Shape-Shifting Robots}, 
    author={Suning Huang and Boyuan Chen and Huazhe Xu and Vincent Sitzmann},
    year={2024},
    eprint={2401.13231},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

For questions about code, please create an issue on github. For questions about paper, please reach me at hsn19@mails.tsinghua.edu.cn

# setup conda environment

#### You can easily install the environment by:

```shell
conda env create -f conda_env.yaml
conda activate dittogym
# install DittoGym benchmark:
pip install git+https://github.com/suninghuang19/dittogym.git
```

Please check the official implementation of [DittoGym](https://github.com/suninghuang19/dittogym) to learn more about the benchmark.

## Run Experiments

#### To train the coarse policy, simply use command:

```shell
python train.py --env_name envname-coarse-v0 --config_file_path  ./models/envname/coarse.json
```

You can also sepcify to your own config file path.

#### To train the coarse-to-fine policy, simply use command:

```shell
python train.py --env_name envname-fine-v0 --residual True --coarse_model_path /path/to/your/coarse/model.pth --config_file_path  ./models/envname/fine.json
```

#### To run the ckpts, simply use command:

```shell
python test.py --env_name envname-fine-v0 --residual True --coarse_model_path /path/to/your/coarse/model.pth --fine_model_path /path/to/your/fine/model.pth --config_file_path  ./models/envname/fine.json
```

#### You can also test your coarse model by set :

```shell
python test.py --env_name envname-coarse-v0 --residual False --coarse_model_path /path/to/your/coarse/model.pth --config_file_path  ./models/envname/coarse.json
```

#### If you want to visualize the result, simply set (this is the default setting):

```shell
--visualize True
```

#### If you have a GUI, then you can visualize the real time results by setting:

```shell
--visualize True --gui True
```

#### You can also set the visualize interval to avoid rendering in every episode (e.g. render per 10 episodes):

```shell
--visualize_interval 10
```

#### If you set the render mode to be “rgb_array”, then you can also get the array from render function:

```python
# img (512, 512, 3)
img = env.render(mode="rgb_array")
```
