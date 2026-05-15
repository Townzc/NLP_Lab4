# 实验报告记录稿

## 四、实验过程

1. 在华为云 ModelArts 控制台创建 Notebook，区域选择乌兰察布一，镜像选择 `mindspore_2.1.0-cann_6.3.2-py_3.7-euler_2.10.7-aarch64-d910`，磁盘容量设置为 128 GB。
2. 在 `/home/ma-user/work` 下拉取本仓库，执行 `bash scripts/00_prepare_lab.sh` 下载并解压实验包，同时下载 `llama_7b.ckpt` 与 `tokenizer.model`。
3. 执行 `bash scripts/01_base_infer.sh`，调用 MindFormers pipeline 对 Llama 7B 基础模型进行文本生成推理。
4. 将智慧校园问答数据组织为 Stanford Alpaca 格式，执行 `bash scripts/02_prepare_data.sh`。脚本先对 JSON 数据进行空白字符、控制字符和重复样本清理，再调用实验包中的 `alpaca_converter.py` 转换为对话格式，最后调用 `llama_preprocess_no_fschat.py` 生成 MindRecord 文件。
5. 执行 `bash scripts/03_train_lora.sh`，脚本基于原始 `run_llama_7b_lora.yaml` 生成 `run_llama_7b_lora_campus.yaml`，设置预训练权重路径、关闭并行训练、设置 epoch、batch size、学习率、LoRA rank/dropout 和训练集路径，然后调用 `run_standalone.sh` 进行单卡 LoRA 微调。
6. 训练完成后执行 `bash scripts/04_lora_infer.sh`，自动选择最新 LoRA checkpoint，使用智慧校园问题进行推理，记录微调后的问答效果。

## 六、实验结果与分析

### 1. 实验结果

待补充：

- 基础模型推理输出：见 `results/base_infer.log`
- 数据预处理输出：见 `results/preprocess.log`
- LoRA 训练 loss 和 checkpoint：见 `results/train_*.log`
- 微调后推理输出：见 `results/lora_infer.log`

### 2. 实验中遇到的问题及解决方法

可记录的问题方向：

- 7B 权重文件较大，下载时间长或中断。解决方法：脚本使用 `wget -c` 支持断点续传，重新运行准备脚本即可继续下载。
- 数据预处理对 JSON 格式敏感。解决方法：先用 `src/clean_alpaca_data.py` 清理和校验 Alpaca 数据，去除空样本、控制字符和重复问题，再进入 MindRecord 转换。
- LoRA checkpoint 文件名会随 epoch 和 step 变化。解决方法：推理脚本不写死 checkpoint 名称，而是在训练输出目录自动选择最新的 `llama_7b_lora_rank_0-*.ckpt`。

### 3. 扩展实验结果与分析

可对比以下配置：

| 实验 | Epoch | LR | LoRA Rank | Dropout | 观察点 |
| --- | --- | --- | --- | --- | --- |
| baseline | 20 | 1.e-4 | 16 | 0.05 | 默认配置 |
| rank8 | 10 | 5.e-5 | 8 | 0.10 | 参数更少，观察收敛速度和回答完整度 |
| rank32 | 20 | 1.e-4 | 32 | 0.05 | 参数更多，观察是否更贴合训练问答 |

分析时可从训练 loss 下降速度、回答是否覆盖关键步骤、是否出现幻觉或遗忘通用能力三个角度展开。
