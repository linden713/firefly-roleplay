# firefly-roleplay

This project is for role-playing with the Firefly model.

## Install

Please follow the [Unsloth Installation Guide](https://docs.unsloth.ai/get-started/install-and-update/conda-install) to set up the environment.

## Usage

### Data

The data can be downloaded from: [Firefly-Roleplay Dataset](https://huggingface.co/datasets/linden713/Firefly-Roleplay)

The project provides scripts to construct data for pre-training and fine-tuning.

- **`construct_continue_pretrain_data.py`**: Constructs data for continued pre-training.
  ```bash
  python script/construct_continue_pretrain_data.py --merge-mode role --merge-max-messages 10 --lang zh
  ```

- **`construct_firefly_data.py`**: Constructs data in the Firefly format.

For more detailed information on the data processing methods, please refer to: https://github.com/linden713/firefly-roleplay

### Training

- **`continue_training_gemma3.py`**: Script for continuing the pre-training of a Gemma model.
  ```bash
  python script/continue_training_gemma3.py
  ```

- **`fine_tune_gemma3.py`**: Script for fine-tuning a Gemma model.
  ```bash
  python script/fine_tune_gemma3.py
  ```

### Inference

- **`inference_gr_gemma.py`**: Launch a Gradio interface for inference with the trained model.




© All rights reserved by miHoYo

Other properties and any right, title, and interest thereof and therein (intellectual property rights included) not derived from Honkai: Star Rail belong to their respective owners.