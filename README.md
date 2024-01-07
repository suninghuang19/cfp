# cfp

#### [[Project Website]](https://morphologicalmaze.github.io/) [[Paper]](./paper/Morphological_Maze_Control_Reconfigurable_Soft_Robots_with_Fine_grained_Morphology_Change.pdf)

[Suning Huang<sup>1</sup>](https://suninghuang19.github.io/), [Boyuan Chen<sup>2</sup>](https://boyuan.space/), [Huazhe Xu<sup>1</sup>](http://hxu.rocks//), [Vincent Sitzmann<sup>2</sup>](https://www.vincentsitzmann.com/) <br/>
<sup>1</sup>Tsinghua <sup>2</sup>MIT </br>

<center>

![](./teaser/M.gif)![](./teaser/I.gif)![](./teaser/T.gif)
</center>

This repo is the official algorithm implementation of ***[Morphological Maze: Control Reconfigurable Soft Robots with Fine-grained Morphology Change](https://morphologicalmaze.github.io/)***. We set up a fully-convolutional policy framework with **coarse-to-fine curriculum** to efficiently train control policies for highly reconfigurable robots to support locomotion and fine-grained morphology changes. If you find this work helpful to your research, please cite us as:

```
@inproceedings{
}
```

For questions about code, please create an issue on github. For questions about paper, please reach me at hsn19@mails.tsinghua.edu.cn

# setup conda environment

#### You can easily install the environment by:

```shell
conda create -n morphmaze python=3.9
conda activate morphmaze
pip install -r requirements.txt
# install morphmaze benchmark:
pip install git+https://github.com/suninghuang19/morphmaze.git
```

Please check the official implementation of [Morphological Maze](https://github.com/suninghuang19/morphmaze) to learn more about the benchmark.

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
