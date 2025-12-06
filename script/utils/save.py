from unsloth import FastModel
import torch
from unsloth.chat_templates import train_on_responses_only, get_chat_template, standardize_data_formats

model, tokenizer = FastModel.from_pretrained(
    model_name="outputs/checkpoint-732",
    max_seq_length=1500,
    load_in_4bit=True,
    full_finetuning=False,
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

model.save_pretrained_merged("gemma-3N-continue-learning-4bit", tokenizer, save_method="force_merged_4bit")