from unsloth import FastModel
import torch
import os
from datasets import load_dataset
from utils.utils import load_chatml_dataset, load_params
from transformers import TextStreamer, TextIteratorStreamer
from trl import SFTTrainer, SFTConfig
from utils.utils import EvalCallback, formatting_prompts_func
from datasets import Dataset
from unsloth.chat_templates import train_on_responses_only, standardize_data_formats, get_chat_template


# Only keep generation from shared YAML; everything else is inlined below
gen_cfg = load_params("generation")

max_seq_length = 1024

print("📚 正在加载模型和分词器...")
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E4B-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)
print("✅ 模型和分词器加载完成")
model.load_adapter("outputs/highrl/checkpoint-732")
# print("🔄 正在格式化训练数据...")
# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template="gemma-3",
# )

# print("\n🔍 执行训练前基准测试 (Pre-training Evaluation)...")

# # Define evaluation messages
eval_messages = [
    {"role": "system", "content": "你是崩坏星穹铁道的角色流萤，请始终保持角色设定和语气"},
    {"role": "user", "content": "开拓者：流萤，你有什么一直想实现的愿望吗？"},
]
# text = tokenizer.apply_chat_template(
#     eval_messages,
#     tokenize=False,
#     enable_thinking=False,
#     add_generation_prompt=True,
# )

# # Tokenize
# inputs = tokenizer(text=text, return_tensors="pt").to("cuda")

# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=256,
#         temperature=0.7,
#         top_p=0.8,
#         top_k=40,
#         repetition_penalty=1.05,
#         pad_token_id=tokenizer.eos_token_id,
#         do_sample=True,
#         streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
#     )
# FastModel.for_training(model)


print("🔧 正在配置LoRA适配器...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)
print("✅ LoRA适配器配置完成")




# Fix: Extract the list from the dict returned by load_chatml_dataset
dataset_ch = load_chatml_dataset("dataset/firefly_chatml_story_dataset_CH.jsonl")["conversations"]
dataset_en = load_chatml_dataset("dataset/firefly_chatml_story_dataset_EN.jsonl")["conversations"]
dataset = dataset_ch + dataset_en
print(f"✅ 数据集加载完成，共 {len(dataset)} 条对话 (CH: {len(dataset_ch)}, EN: {len(dataset_en)})")
print("✅ 数据集格式化完成")
full_dataset = Dataset.from_dict({"conversations": dataset})
full_dataset = standardize_data_formats(full_dataset)
test_size = 0.05
split_seed = 42
train_val_split = full_dataset.train_test_split(test_size=test_size, seed=split_seed)
train_dataset = train_val_split["train"]
eval_dataset = train_val_split["test"]

# Map into `text` field
train_dataset = train_dataset.map(
    formatting_prompts_func,
    batched=True,
    fn_kwargs={"tokenizer": tokenizer},
)
eval_dataset = eval_dataset.map(
    formatting_prompts_func,
    batched=True,
    fn_kwargs={"tokenizer": tokenizer},
)




print("⚙️ 正在初始化训练器...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    max_seq_length=max_seq_length,
    dataset_num_proc=20,
    packing=True,
    eval_dataset=eval_dataset,

    args=SFTConfig(
        dataset_text_field="text",

        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=5,
        learning_rate=1e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="tensorboard",
        save_strategy="epoch",
        save_total_limit=5,
        
        per_device_eval_batch_size=1,
        eval_accumulation_steps=4,
        eval_strategy="epoch",
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)
# Add evaluation callback
eval_callback = EvalCallback(model, tokenizer, eval_messages, gen_config=gen_cfg)
trainer.add_callback(eval_callback)
print("✅ 训练器初始化完成")
# print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))
# print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]).replace(tokenizer.pad_token, " "))
print("🚀 开始模型训练...")



trainer.evaluate()
trainer_stats = trainer.train()
print("🎊 所有任务完成！")
