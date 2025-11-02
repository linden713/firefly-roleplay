from unsloth import FastModel
import torch
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it",
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0.0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = True,
    loftq_config = None,
)
# model = FastModel.get_peft_model(
#     model,
#     finetune_vision_layers     = False, # Turn off for just text!
#     finetune_language_layers   = True,  # Should leave on!
#     finetune_attention_modules = True,  # Attention good for GRPO
#     finetune_mlp_modules       = True,  # Should leave on always!

#     r = 8,           # Larger = higher accuracy, but might overfit
#     lora_alpha = 8,  # Recommended alpha == r at least
#     lora_dropout = 0,
#     bias = "none",
#     random_state = 3407,
# )

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")

from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_num_proc = 12,
    packing = False,
    eval_dataset = None,

    args = SFTConfig(
        dataset_text_field = "text",

        per_device_train_batch_size = 2, #8
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.05,
        num_train_epochs = 4,
        learning_rate = 2e-5,
        # fp16 = not torch.cuda.is_bf16_supported(),
        # bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01, #TODO 
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "tensorboard",
        save_strategy="epoch",
        save_total_limit=4,
    ),
)
# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = dataset,
#     eval_dataset = None, # Can set up evaluation!
#     args = SFTConfig(
#         dataset_text_field = "text",
#         per_device_train_batch_size = 1,
#         gradient_accumulation_steps = 4, # Use GA to mimic batch size!
#         warmup_steps = 5,
#         # num_train_epochs = 1, # Set this for 1 full training run.
#         max_steps = 60,
#         learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
#         logging_steps = 1,
#         optim = "adamw_8bit",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 3407,
#         report_to = "none", # Use this for WandB etc
#     ),
# )

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

trainer_stats = trainer.train()