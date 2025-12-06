# Firefly Roleplay Evaluation Module Documentation

This directory contains a comprehensive suite of tools and scripts for evaluating the performance of the Firefly Roleplay model. These tools are designed to automate the testing of the model's roleplay capabilities, knowledge consistency, safety, and text generation quality.

## Directory Structure

*   **`evaluate.py`**: The core evaluation script responsible for executing inference and calculating metrics.
*   **`eval_constants.py`**: Defines constants, regex patterns, and the knowledge base used for evaluation.
*   **`extract_results.py`**: A helper script to batch extract and summarize results from multiple experiments.
*   **`user_query_CH.txt`**: The default Chinese prompt file for testing.
*   **`evaluation_result/`**: The default directory for storing evaluation results (reports, logs).

---

## 1. Core Evaluation Script (`evaluate.py`)

This is the main entry point for the evaluation pipeline. It loads a specified model (or adapter), reads test prompts, generates responses, and calculates multi-dimensional evaluation metrics.

### Key Features

*   **Batch Inference**: Supports batch generation to improve evaluation efficiency.
*   **RAG (Retrieval-Augmented Generation)**: Optionally enables RAG to allow the model to retrieve relevant memories or knowledge before generating.
*   **Multi-dimensional Metrics**:
    *   **PPL (Perplexity)**: Conditional perplexity, measuring the fluency of the generated text and how "surprised" the model is by the response.
    *   **Distinctness (1/2/3-gram)**: Measures the diversity of the generated text to avoid repetition.
    *   **Burstiness**: Measures the burstiness of text (clustering of vocabulary distribution), reflecting the naturalness of the text.
    *   **Knowledge Consistency**: Tests whether the model has mastered the character's background settings (e.g., name, mecha, experiences) through a preset list of QA pairs.
    *   **Persona & OOC (Out-of-Character)**: Detects if the response contains mandatory character keywords or triggers OOC warning words (e.g., "As an AI").
    *   **Safety**: Detects if the response contains sensitive or prohibited vocabulary.
*   **Detailed Reporting**: Generates a JSON report with metric summaries, a Markdown report, and a JSONL log containing detailed data for each conversation.

### Usage

```bash
python evaluation/evaluate.py [arguments]
```

### Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--adapter_name` | str | `unsloth/gemma-3n-E4B-it` | **[Required]** Path or name of the model adapter to evaluate. |
| `--input_file` | str | `evaluation/user_query_CH.txt` | Path to the test prompt file (one prompt per line). |
| `--output_dir` | str | `evaluation/evaluation_result` | Directory to save evaluation results. |
| `--batch_size` | int | `50` | Batch size for inference. |
| `--max_samples` | int | `None` | Limit the number of test samples (for quick testing). |
| `--system_key` | str | `NORMAL` | System prompt key to use (defined in `init_prompt.py`). |
| `--enable_rag` | flag | `False` | Whether to enable RAG (Retrieval-Augmented Generation). |
| `--debug` | flag | `False` | Whether to enable debug mode (prints more logs). |

### Examples

```bash
# Evaluate a specific adapter
python evaluation/evaluate.py --adapter_name outputs/checkpoint-100

# Enable RAG and specify output directory
python evaluation/evaluate.py --adapter_name outputs/checkpoint-100 --enable_rag --output_dir results/rag_test
```

---

## 2. Constants Definition (`eval_constants.py`)

This file defines the static rules and data relied upon by the evaluation logic. If you need to adjust evaluation standards (e.g., modifying character knowledge settings), please modify this file.

### Main Contents

*   **`PERSONA_MUST`**: Keywords that are preferred to appear in character responses (e.g., "Firefly", "Sam").
*   **`OOC_PATTERNS`**: List of regex patterns considered OOC (e.g., `r'I am\s*AI'`, `r'As a\s*language model'`).
*   **`STYLE_HINTS`**: Style check rules, such as whether specific forms of address ("Trailblazer") are used, and whether the short sentence style is met.
*   **`MEMORY_ANCHORS`**: Memory anchor word list, used to detect if the model mentions key plot elements in its response (e.g., "Oak Cake Roll", "Glamoth").
*   **`SAFETY_BLOCK`**: List of safety-sensitive words.
*   **`KNOWLEDGE_QA`**: **Core Knowledge Base**. This is a list of tuples containing (Question, [List of Reference Answers]). The evaluation script asks the model these questions and checks if the response contains keywords from the reference answers to calculate the knowledge consistency score.

---

## 3. Result Extraction (`extract_results.py`)

When you have run multiple experiments (e.g., checkpoints at different steps, or different hyperparameters), this script helps you quickly compare results.

### Functionality

It scans all subdirectories under the `evaluation_result` directory, finds the latest `report_*.json` file for each experiment, extracts key metrics, and prints them as a Markdown table.

### Usage

```bash
python evaluation/extract_results.py
```

### Metric Explanation

| Metric | Meaning | Ideal Direction |
| :--- | :--- | :--- |
| **ooc_rate** | OOC Rate | **Lower is better** (close to 0) |
| **knowledge_consistency** | Knowledge Consistency Score | **Higher is better** (close to 1.0) |
| **safety_violation_rate** | Safety Violation Rate | **Lower is better** (close to 0) |
| **mean_ppl** | Mean Perplexity | **Lower is better** (indicates better fluency) |
| **distinct_1/2/3** | n-gram Distinctness | **Higher is better** (indicates less repetition) |
| **repeat_rate_6gram** | Long Segment Repeat Rate | **Lower is better** |
| **memory_hit_rate** | Memory Anchor Hit Rate | **Moderate/Higher is better** |

---

## Evaluation Result Files

After running `evaluate.py`, the following files (filenames include timestamps) will be generated in the output directory:

1.  **`report_*.json`**: JSON file containing all summary metrics.
2.  **`report_*.md`**: Brief report in Markdown format.
3.  **`eval_details_*.jsonl`**: Detailed log containing every test record (prompt, response, individual scores, etc.).
4.  **`conversations_*.md`**: Human-readable conversation log file.
5.  **`knowledge_qa_*.md`**: Detailed results of the knowledge QA test (showing which questions were answered correctly/incorrectly).

---

## 4. Training & Helper Scripts (`script/`)

In addition to the evaluation module, this project includes scripts for model training, data construction, and interactive inference, located in the `script/` directory.

### Core Scripts

*   **`fine_tune_gemma3.py`**: **Model Fine-tuning Script**.
    *   Based on the Unsloth framework, performs LoRA fine-tuning on the Gemma-3N model.
    *   Automatically loads Chinese and English ChatML datasets from `dataset/`.
    *   Includes Pre-training Evaluation.
    *   Configures SFTTrainer for supervised fine-tuning.

*   **`inference_gemma3.py`**: **Gradio Interactive Inference**.
    *   Launches a Web UI chat interface for real-time conversation with the fine-tuned model.
    *   Supports streaming output (Streamer).
    *   Includes simple dialogue testing features (replaces the legacy `inference_gr_gemma.py`).

*   **`construct_firefly_data.py`**: **Data Construction Tool**.
    *   Converts raw JSON script data into ChatML format (`.jsonl`) suitable for fine-tuning.
    *   Supports multi-turn dialogue merging, character cleaning, and long sentence deduplication.
    *   **Usage Example**:
        ```bash
        python script/construct_firefly_data.py --input dataset/raw/SR_Talk_EN.json --output dataset/firefly_chatml_story_dataset_EN.jsonl --lang en
        ```

*   **`evaluate_gemma3_eval_only.py`**: **Interactive Evaluation Script (Legacy)**.
    *   Similar to `evaluate.py` but focuses more on manual evaluation via a Gradio interface.
    *   Includes simple RAG integration tests.

### Utility Library (`script/utils/`)

*   **`init_prompt.py`**: Stores System Prompts, including `NORMAL` (Standard Firefly setting), `TEST`, `OPENING_SCENE`, etc.
*   **`rag_utils.py`**: Implementation of the RAG retrieval module, responsible for retrieving relevant memories.
*   **`utils.py`**: General utility functions, such as parameter loading and data formatting.
