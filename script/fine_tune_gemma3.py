from unsloth import FastModel
import torch
from datasets import load_dataset
from utils import load_chatml_dataset, load_params
from trl import SFTTrainer, SFTConfig
from utils import EvalCallback, formatting_prompts_func
from datasets import Dataset
from unsloth.chat_templates import train_on_responses_only

# Only keep generation from shared YAML; everything else is inlined below
gen_cfg = load_params("generation").generation

max_seq_length = 1024

print("📚 正在加载模型和分词器...")
model, tokenizer = FastModel.from_pretrained(
    model_name="checkpoint-3062",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)
print("✅ 模型和分词器加载完成")


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

print("🔄 正在格式化训练数据...")
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3",
)
dataset_path = "firefly_chatml_final.jsonl"
dataset_dict = load_chatml_dataset(dataset_path)
dataset = dataset_dict["conversations"]
train_dataset = Dataset.from_dict({"conversations": dataset})
train_dataset = train_dataset.map(formatting_prompts_func, batched=True,
        fn_kwargs={'tokenizer': tokenizer})


# from datasets import load_dataset
# dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")
# from unsloth.chat_templates import standardize_data_formats
# dataset = standardize_data_formats(dataset)
# def formatting_prompts_func(examples):
#    convos = examples["conversations"]
#    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
#    return { "text" : texts, }

# train_dataset = dataset.map(formatting_prompts_func, batched = True)



# Define evaluation messages
eval_messages = [
    {"role": "system", "content": "你是崩坏星穹铁道的角色流萤，请始终保持角色设定和语气"},
    {"role": "user", "content": "开拓者：流萤，你有什么一直想实现的愿望吗？"},
]


print("⚙️ 正在初始化训练器...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    max_seq_length=max_seq_length,
    dataset_num_proc=12,
    packing=False,
    eval_dataset=None,

    args=SFTConfig(
        dataset_text_field="text",

        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=4,
        learning_rate=2e-5,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="tensorboard",
        save_strategy="epoch",
        save_total_limit=4,
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
print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))
print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]).replace(tokenizer.pad_token, " "))
print("🚀 开始模型训练...")
trainer_stats = trainer.train()
print("🎊 所有任务完成！")
