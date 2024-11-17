<!--
 * @Description:
 * @Author: Jianping Zhou
 * @Email: jianpingzhou0927@gmail.com
 * @Date: 2024-09-19 15:44:40
-->

# Pollutants_Molecular

It's a pytorch implementation of pollutant molecular network predictions.

```
.
└─Pollutants_Molecular
  ├─README.md
  ├─LICENSE
  ├─.gitignore
  ├─requirements.txt                    项目环境依赖包
  ├─datasets                            存放数据集数据文件
  ├─checkpoints                         存放保存的模型参数文件
  ├─dataloader                          存放读取 dataset 的 py 文件
  ├─models                              baselines的模型架构
  ├─logs                                存放运行结果的输出日志文件
  ├─results                             存放模型的测试结果
  ├─scripts                             运行脚本文件
  ├─src                                 核心代码源文件
  ├─utils                               工具函数源文件
```

## How to Run

1. Prepare environment

```
pip install -r requirements.txt
```

2. Data preprocessing

```
cd scripts
bash dataprocess.sh
bash tanimotoGenerator.sh
```

3. Training and testing

```
cd scripts
bash run_changjiang.sh
bash run_huangpujiang.sh
bash run_Sediment.sh
bash run_honeyNeg.sh
bash run_honeyPos.sh
bash run_synthtic.sh
```

