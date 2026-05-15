# NLP Lab 4: Llama 7B Inference and LoRA Fine-tuning on MindSpore

本仓库用于完成“基于 MindSpore-Ascend 的 llama7b 模型推理与微调实验”。代码按华为云 ModelArts Notebook 环境组织，默认工作目录为 `/home/ma-user/work`。

## 1. ModelArts 环境

按实验指导创建 Notebook：

- 区域：乌兰察布一
- 镜像：`mindspore_2.1.0-cann_6.3.2-py_3.7-euler_2.10.7-aarch64-d910`
- 磁盘：128 GB
- 自动停止：建议 1 小时，可按训练时间延长

进入 Notebook 后，在 `/home/ma-user/work` 下拉取本仓库：

```bash
cd /home/ma-user/work
git clone https://github.com/Townzc/NLP_Lab4.git
cd NLP_Lab4
```

## 2. 一键流程

如果希望按实验流程完整执行：

```bash
bash scripts/run_all.sh
```

该脚本会依次完成：

1. 下载并解压实验包 `llama_lab.zip`
2. 下载 `llama_7b.ckpt`、`tokenizer.model` 及 lock 文件
3. 复制本仓库的推理脚本到 `llama_lab/mindformers`
4. 运行基础模型推理
5. 清洗智慧校园 Alpaca 数据并转换为对话格式
6. 转换 MindRecord 数据
7. 生成 LoRA 微调 YAML 配置
8. 执行单卡 LoRA 微调
9. 自动查找最新 LoRA checkpoint 并推理

日志会保存到本仓库 `results/` 目录，后续写实验报告时可直接复制关键输出。

如果下载 OBS 文件时遇到 `Connection timed out`，先停止脚本，然后拉取最新代码后重试：

```bash
git pull
bash scripts/00_prepare_lab.sh
```

脚本会自动续传已下载的部分文件。若仍然超时，可以尝试强制直连下载：

```bash
DISABLE_PROXY_DOWNLOAD=1 bash scripts/00_prepare_lab.sh
```

如果 ModelArts 到实验 OBS 桶一直不通，就在本地或其他网络下载文件后上传到脚本提示的精确路径，再重新运行 `scripts/00_prepare_lab.sh`。其中 `llama_7b.ckpt` 约 12.6 GiB，实验包 `llama_lab.zip` 约 45 MiB。

## 3. 分步运行

也可以逐步执行，便于截图和记录结果：

```bash
bash scripts/00_prepare_lab.sh
bash scripts/01_base_infer.sh
bash scripts/02_prepare_data.sh
bash scripts/03_train_lora.sh
bash scripts/04_lora_infer.sh
```

训练输出目录默认是：

```text
/home/ma-user/work/llama_lab/mindformers/scripts/mf_standalone/output/checkpoint/rank_0
```

LoRA 推理脚本会在该目录中自动选择最新的 `llama_7b_lora_rank_0-*.ckpt`。

## 4. 扩展实验调参

`scripts/03_train_lora.sh` 支持用环境变量生成不同 YAML 配置。例如：

```bash
CONFIG_NAME=run_llama_7b_lora_rank8_dropout10.yaml \
EPOCHS=10 \
BATCH_SIZE=2 \
LEARNING_RATE=5.e-5 \
LORA_RANK=8 \
LORA_ALPHA=16 \
LORA_DROPOUT=0.10 \
bash scripts/03_train_lora.sh
```

常用变量：

- `EPOCHS`：训练轮数，默认 `20`
- `BATCH_SIZE`：训练 batch size，默认 `2`
- `SINK_SIZE`：sink size，默认 `5`
- `LEARNING_RATE`：优化器和学习率调度的学习率，默认 `1.e-4`
- `OPTIMIZER_TYPE`：默认 `FP32StateAdamWeightDecay`
- `LR_SCHEDULE_TYPE`：默认 `CosineWithWarmUpLR`
- `LORA_RANK`：默认 `16`
- `LORA_ALPHA`：默认 `16`
- `LORA_DROPOUT`：默认 `0.05`

调参后再运行：

```bash
bash scripts/04_lora_infer.sh
```

推理脚本仍会自动选择最新 checkpoint。若要指定某个 checkpoint：

```bash
LORA_CKPT=/home/ma-user/work/llama_lab/mindformers/scripts/mf_standalone/output/checkpoint/rank_0/xxx.ckpt \
bash scripts/04_lora_infer.sh
```

## 5. 实验报告素材

报告初稿结构见 `docs/report_notes.md`。你在 ModelArts 跑完后，把下面几类内容复制回来即可继续补报告：

- `results/base_infer.log`
- `results/preprocess.log`
- `results/train_*.log`
- `results/lora_infer.log`
- 训练 loss、checkpoint 路径、推理问答结果截图
