# go2arm-WBC

Go2 与机械臂的双策略全身控制。狗策略负责移动，臂策略负责末端操作，并给出机身 plan 指令。

## 安装

需要 Isaac Gym Preview 4 和 Python 3.8。

```bash
conda create -n go2arm-wbc python=3.8
conda activate go2arm-wbc

# 解压 Isaac Gym 后
cd isaacgym/python && pip install -e .

cd /path/to/go2arm-WBC
```

在仓库根目录运行脚本。训练还需要本机已有的 PyTorch、wandb 和 params-proto。

## 训练

Go2 + Airbot：

```bash
python scripts/auto_train.py \
  --num_envs 4096 \
  --run_name go2_airbot \
  --sim_device cuda:0 \
  --robot go2_airbot \
  --headless
```

日志写在 `runs/<run_name>/<日期>/auto_train/<时间>_seed<seed>/`。狗和臂的权重分别在 `checkpoints_dog/`、`checkpoints_arm/`。

`scripts/unified_train.py` 用同一个网络同时输出狗和臂的动作，机器人选项是 `go1` 和 `go2`。

常用开关：

| 开关 | 作用 |
| --- | --- |
| `--robot go2_airbot` | Go2 + Airbot。也可选 `go1`、`go2` |
| `--headless` | 不打开仿真窗口 |
| `--num_envs` | 并行环境数，默认 2048 |
| `--num_learning_iterations` | 训练迭代数，默认 100000 |
| `--wo_two_stage` | 跳过预训练到混合奖励的切换，从头训练 |
| `--use_rot6d` | 臂的姿态命令用 6D 旋转表示 |

## Play

Play 从 `logdir/parameters.pkl` 读取机器人模型。`--ckptid` 对应 `checkpoints_*/ac_weights_XXXXXX.pt`。

```bash
python scripts/play_by_key.py \
  --logdir runs/go2_airbot/<日期>/auto_train/<时间>_seed<seed> \
  --ckptid 60000 \
  --sim_device cuda:0
```

仓库里带了一份 Go2 权重，可以直接看：

```bash
python scripts/play_by_key.py \
  --logdir runs/test_go212.11/181025.391800_seed1133 \
  --ckptid 99999 \
  --sim_device cuda:0
```

`--headless` 关闭窗口。`--dog_only` 只跑狗策略。`--unified` 加载 `checkpoints_unified` 里的单网络权重。

键盘：

| 键 | 动作 | 键 | 动作 |
| --- | --- | --- | --- |
| 上 / 下 | 前进 / 后退 | U / O | 臂升高 / 降低 |
| 左 / 右 | 左移 / 右移 | I / K | 臂向前 / 向后 |
| Z / X | 左转 / 右转 | J / L | 臂向左 / 向右 |
| W / S | 臂 pitch | A / D | 臂 roll |
| Q / E | 臂 yaw | R | 复位 |
