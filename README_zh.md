# Firefly Roleplay 评估模块文档

本目录包含用于评估 Firefly Roleplay 模型性能的一整套工具和脚本。这些工具旨在自动化测试模型的角色扮演能力、知识一致性、安全性以及生成文本的质量。

## 目录结构

*   **`evaluate.py`**: 核心评估脚本，负责执行推理和计算指标。
*   **`eval_constants.py`**: 包含评估所需的常量、正则模式和知识库。
*   **`extract_results.py`**: 用于批量提取和汇总多个实验结果的辅助脚本。
*   **`user_query_CH.txt`**: 默认的中文测试提示词文件。
*   **`evaluation_result/`**: 存放评估结果（报告、日志）的默认目录。

---

## 1. 核心评估脚本 (`evaluate.py`)

这是评估流程的主入口。它加载指定的模型（或适配器），读取测试提示词，生成回复，并计算多维度的评估指标。

### 主要功能

*   **批量推理**: 支持批量生成，提高评估效率。
*   **RAG (检索增强生成)**: 可选开启 RAG 功能，让模型在生成前检索相关记忆或知识。
*   **多维度指标计算**:
    *   **PPL (Perplexity)**: 条件困惑度，衡量生成文本的流畅性和模型对回复的“意外”程度。
    *   **Distinctness (1/2/3-gram)**: 衡量生成文本的多样性，避免重复。
    *   **Burstiness**: 衡量文本的突发性（词汇分布的聚集程度），反映文本的自然度。
    *   **Knowledge Consistency (知识一致性)**: 通过预设的问答对，测试模型是否掌握了角色的背景设定（如名字、机甲、经历等）。
    *   **Persona & OOC (角色符合度)**: 检测回复中是否包含必须的角色关键词，或是否触发了 OOC (Out-of-Character) 警告词（如“作为AI”）。
    *   **Safety (安全性)**: 检测回复是否包含敏感或违禁词汇。
*   **详细报告**: 生成包含所有指标摘要的 JSON 报告、Markdown 报告，以及包含每条对话详细数据的 JSONL 日志。

### 使用方法

```bash
python evaluation/evaluate.py [参数]
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--adapter_name` | str | `unsloth/gemma-3n-E4B-it` | **[必填]** 待评估的模型适配器路径或名称。 |
| `--input_file` | str | `evaluation/user_query_CH.txt` | 测试用的提示词文件路径（每行一个提示词）。 |
| `--output_dir` | str | `evaluation/evaluation_result` | 评估结果的保存目录。 |
| `--batch_size` | int | `50` | 推理时的批次大小。 |
| `--max_samples` | int | `None` | 限制测试样本数量（用于快速测试）。 |
| `--system_key` | str | `NORMAL` | 使用的系统提示词模板 Key (在 `init_prompt.py` 中定义)。 |
| `--enable_rag` | flag | `False` | 是否开启 RAG 检索增强功能。 |
| `--debug` | flag | `False` | 是否开启调试模式（打印更多日志）。 |

### 示例

```bash
# 评估指定适配器
python evaluation/evaluate.py --adapter_name outputs/checkpoint-100

# 开启 RAG 并指定输出目录
python evaluation/evaluate.py --adapter_name outputs/checkpoint-100 --enable_rag --output_dir results/rag_test
```

---

## 2. 常量定义 (`eval_constants.py`)

此文件定义了评估逻辑所依赖的静态规则和数据。如果你需要调整评估标准（例如修改角色的设定知识），请修改此文件。

### 主要内容

*   **`PERSONA_MUST`**: 角色回复中倾向于出现的关键词（如“流萤”、“萨姆”）。
*   **`OOC_PATTERNS`**: 视为 OOC 的正则表达式列表（如 `r'我是\s*AI'`, `r'作为\s*语言模型'`）。
*   **`STYLE_HINTS`**: 风格检查规则，例如是否使用了特定的称呼（“开拓者”），以及是否符合短句风格。
*   **`MEMORY_ANCHORS`**: 记忆锚点词表，用于检测模型是否在回复中提及了关键剧情元素（如“橡木蛋糕卷”、“格拉默”）。
*   **`SAFETY_BLOCK`**: 安全敏感词列表。
*   **`KNOWLEDGE_QA`**: **核心知识库**。这是一个包含 (问题, [参考答案列表]) 的元组列表。评估脚本会用这些问题询问模型，并检查回复是否包含参考答案中的关键词，以此计算知识一致性分数。

---

## 3. 结果汇总 (`extract_results.py`)

当你运行了多次实验（例如不同步数的 checkpoint，或不同的超参数），这个脚本可以帮助你快速对比结果。

### 功能

它会扫描 `evaluation_result` 目录下的所有子文件夹，找到每个实验最新的 `report_*.json` 文件，提取关键指标，并以 Markdown 表格的形式打印出来。

### 使用方法

```bash
python evaluation/extract_results.py
```

### 输出指标解释

| 指标 | 含义 | 理想方向 |
| :--- | :--- | :--- |
| **ooc_rate** | OOC 率 | **越低越好** (接近 0) |
| **knowledge_consistency** | 知识一致性得分 | **越高越好** (接近 1.0) |
| **safety_violation_rate** | 安全违规率 | **越低越好** (接近 0) |
| **mean_ppl** | 平均困惑度 | **越低越好** (表示更流畅) |
| **distinct_1/2/3** | n-gram 多样性 | **越高越好** (表示不重复) |
| **repeat_rate_6gram** | 长片段重复率 | **越低越好** |
| **memory_hit_rate** | 记忆锚点命中率 | **适中/偏高** |

---

## 评估结果文件说明

每次运行 `evaluate.py` 后，会在输出目录生成以下文件（文件名包含时间戳）：

1.  **`report_*.json`**: 包含所有汇总指标的 JSON 文件。
2.  **`report_*.md`**: 简要的 Markdown 格式报告。
3.  **`eval_details_*.jsonl`**: 包含每一条测试数据的详细记录（提示词、回复、单条评分等）。
4.  **`conversations_*.md`**: 人类可读的对话记录文件。
5.  **`knowledge_qa_*.md`**: 知识问答测试的详细结果（显示哪些问题答对了，哪些答错了）。

---

## 4. 训练与辅助脚本 (`script/`)

除了评估模块，本项目还包含用于模型训练、数据构建和交互式推理的脚本，位于 `script/` 目录下。

### 核心脚本

*   **`fine_tune_gemma3.py`**: **模型微调脚本**。
    *   基于 Unsloth 框架，对 Gemma-3N 模型进行 LoRA 微调。
    *   自动加载 `dataset/` 下的中英文 ChatML 数据集。
    *   包含训练前的基准测试 (Pre-training Evaluation)。
    *   配置了 SFTTrainer 进行监督微调。

*   **`inference_gemma3.py`**: **Gradio 交互式推理**。
    *   启动一个 Web UI 聊天界面，与微调后的模型进行实时对话。
    *   支持流式输出 (Streamer)。
    *   包含简单的对话测试功能（替代了旧版的 `inference_gr_gemma.py`）。

*   **`construct_firefly_data.py`**: **数据构建工具**。
    *   将原始的 JSON 剧本数据转换为适合微调的 ChatML 格式 (`.jsonl`)。
    *   支持多轮对话合并、角色清洗、长句去重等预处理逻辑。
    *   **使用示例**:
        ```bash
        python script/construct_firefly_data.py --input dataset/raw/SR_Talk_CH.json --output dataset/firefly_chatml_story_dataset_CH.jsonl --lang ch
        ```

*   **`evaluate_gemma3_eval_only.py`**: **交互式评估脚本 (旧版)**。
    *   类似于 `evaluate.py`，但更侧重于通过 Gradio 界面进行人工评估。
    *   包含简单的 RAG 集成测试。

### 工具库 (`script/utils/`)

*   **`init_prompt.py`**: 存放系统提示词 (System Prompts)，包括 `NORMAL` (标准流萤设定), `TEST`, `OPENING_SCENE` 等。
*   **`rag_utils.py`**: RAG 检索模块的实现，负责向量检索相关记忆。
*   **`utils.py`**: 通用工具函数，如参数加载、数据格式化等。
