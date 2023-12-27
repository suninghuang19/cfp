# cfp

This repo is the official implementation of CFP (Coarse-to-Fine Policy) of ***[Morphological Maze: Control Reconfigurable Soft Robots with Fine-grained Morphology Change](https://morphologicalmaze.github.io/)***



You can easily install the environment by:

```python
pip install -r requirements.txt
```



Please check the official implementation of [Morphological Maze](https://github.com/suninghuang19/morphmaze) to learn more about the benchmark.



* To train the coarse policy, simply use command:

```shell
python train.py --env_name taskname_Coarse-v0 --config_file_path  ./models/taskname/coarse.json
```

The models and videos will be auto-saved.



* To train the coarse-to-fine policy, simply use command:

```shell
python train.py --env_name task_Fine-v0 --residual True --coarse_model_path /path/to/your/coarse/model.pth --config_file_path  ./models/taskname/fine.json
```



* To run the ckpts, simply use command: (you can also test your coarse model)

```
python test.py --env_name task_Fine-v0 --residual True --coarse_model_path /path/to/your/coarse/model.pth --fine_model_path /path/to/your/fine/model.pth --config_file_path  ./models/taskname/fine.json
```

You can also test your coarse model by set :

```shell
--residual False
```

