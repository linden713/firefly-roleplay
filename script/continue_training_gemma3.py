from unsloth import FastModel
import torch
from datasets import load_dataset
from utils import load_chatml_dataset, check_conversation_lengths, load_params
from trl import SFTTrainer, SFTConfig
from utils import EvalCallback, formatting_prompts_func
from datasets import Dataset
from unsloth.chat_templates import train_on_responses_only, get_chat_template

params = load_params("training_continue")

model_cfg = params.get("model", {})
lora_cfg = params.get("lora", {})
data_cfg = params.get("data", {})
trainer_cfg = params.get("trainer", {})
args_cfg = (trainer_cfg.get("args") or {})
gen_cfg = params.get("generation", {})
io_cfg = params.get("io", {})

max_seq_length = int(model_cfg.get("max_seq_length", 1024))

print("📚 正在加载模型和分词器...")
model, tokenizer = FastModel.from_pretrained(
    model_name=model_cfg.get("name", "unsloth/gemma-3n-E4B-it"),
    max_seq_length=max_seq_length,
    load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
    load_in_8bit=bool(model_cfg.get("load_in_8bit", False)),
    full_finetuning=bool(model_cfg.get("full_finetuning", False)),
)
print("✅ 模型和分词器加载完成")


print("🔧 正在配置LoRA适配器...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=bool(lora_cfg.get("finetune_vision_layers", False)),
    finetune_language_layers=bool(lora_cfg.get("finetune_language_layers", True)),
    finetune_attention_modules=bool(lora_cfg.get("finetune_attention_modules", True)),
    finetune_mlp_modules=bool(lora_cfg.get("finetune_mlp_modules", True)),
    target_modules=lora_cfg.get(
        "target_modules",
        [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    ),
    r=int(lora_cfg.get("r", 16)),
    lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
    lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
    bias=lora_cfg.get("bias", "none"),
    use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
    random_state=int(lora_cfg.get("random_state", 3407)),
    use_rslora=bool(lora_cfg.get("use_rslora", True)),
    loftq_config=lora_cfg.get("loftq_config", None),
)
print("✅ LoRA适配器配置完成")

print("🔄 正在格式化训练数据...")
tokenizer = get_chat_template(
    tokenizer,
    chat_template=str(data_cfg.get("chat_template", "gemma-3")),
)
# Load dataset
dataset_path = data_cfg.get("dataset_path", "dataset/multiturn_expand_all.jsonl")
dataset_dict = load_chatml_dataset(dataset_path)

dataset = dataset_dict["conversations"]
print("✅ 数据集格式化完成")

# Check token lengths of dataset
check_conversation_lengths(dataset, tokenizer)

# Split 5% validation set
full_dataset = Dataset.from_dict({"conversations": dataset})
test_size = float(data_cfg.get("split_test_size", 0.05))
split_seed = int(data_cfg.get("seed", 42))
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



# Define evaluation messages
eval_messages = params.get("eval_messages", [
    {"role": "system", "content": "你是崩坏星穹铁道的角色流萤，请始终保持角色设定和语气"},
    {"role": "user", "content": "开拓者：流萤，你有什么一直想实现的愿望吗？"},
])


print("⚙️ 正在初始化训练器...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    max_seq_length=max_seq_length,
    dataset_num_proc=int(trainer_cfg.get("dataset_num_proc", 12)),
    packing=bool(trainer_cfg.get("packing", True)),
    eval_dataset=eval_dataset,

    args=SFTConfig(
        dataset_text_field=str(args_cfg.get("dataset_text_field", "text")),

        per_device_train_batch_size=int(args_cfg.get("per_device_train_batch_size", 8)),
        gradient_accumulation_steps=int(args_cfg.get("gradient_accumulation_steps", 4)),
        warmup_ratio=float(args_cfg.get("warmup_ratio", 0.05)),
        num_train_epochs=float(args_cfg.get("num_train_epochs", 1)),
        learning_rate=float(args_cfg.get("learning_rate", 2e-5)),
        logging_steps=int(args_cfg.get("logging_steps", 1)),
        optim=str(args_cfg.get("optim", "adamw_8bit")),
        weight_decay=float(args_cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=str(args_cfg.get("lr_scheduler_type", "cosine")),
        seed=int(args_cfg.get("seed", 3407)),
        output_dir=str(args_cfg.get("output_dir", "outputs")),
        report_to=str(args_cfg.get("report_to", "tensorboard")),

        per_device_eval_batch_size=int(args_cfg.get("per_device_eval_batch_size", 8)),
        eval_accumulation_steps=int(args_cfg.get("eval_accumulation_steps", 4)),
        eval_strategy=str(args_cfg.get("eval_strategy", "steps")),
        eval_steps=int(args_cfg.get("eval_steps", 300)),

        save_strategy=str(args_cfg.get("save_strategy", "epoch")),
        save_total_limit=int(args_cfg.get("save_total_limit", 1)),
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part=str(io_cfg.get("instruction_part", "<start_of_turn>user\n")),
    response_part=str(io_cfg.get("response_part", "<start_of_turn>model\n")),
)
# Add evaluation callback
eval_callback = EvalCallback(model, tokenizer, eval_messages, gen_config=gen_cfg)
trainer.add_callback(eval_callback)
print("✅ 训练器初始化完成")
# print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
# print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " "))
print("🚀 开始模型训练...")
trainer_stats = trainer.train()
print("🎊 所有任务完成！")
