# Fine-Tuning Llama 2 on Hawaii Wildfires Dataset

## Project Overview
This project demonstrates how to fine-tune a Large Language Model (LLM) using Meta’s Llama-2-7B-Chat-HF on a custom dataset related to Hawaii wildfires. The code is implemented in Google Colab, which provides free GPU resources suitable for machine learning tasks. The main goal is to adapt a pre-trained model to a specific topic through efficient fine-tuning techniques like LoRA and quantization.

Fine-tuning allows a general-purpose language model to perform better on domain-specific tasks without training from scratch. This approach is memory-efficient and practical for small-scale or personal experiments.

---

## Key Concepts

### Large Language Model (LLM)
Models such as Llama-2 are transformer-based architectures capable of processing and generating human-like text by predicting the next token in a sequence.

### Fine-Tuning
Fine-tuning involves adjusting an existing model’s weights slightly based on a smaller dataset to make it perform well in a specific context.

### Parameter-Efficient Fine-Tuning (PEFT) with LoRA
Instead of updating all parameters of a massive model, LoRA (Low-Rank Adaptation) focuses on a smaller set of trainable parameters, making the process faster and less resource-intensive.

### Quantization
Quantization reduces model precision (e.g., from 32-bit to 4-bit), significantly decreasing memory usage and allowing large models to fit into limited GPU resources such as those available in Google Colab.

### Dataset
The dataset is a simple text file containing information and factual data about Hawaii wildfires. It is used to train the model to generate domain-specific responses.

---

## Libraries Used
- **transformers**: Provides tools to load and fine-tune pre-trained models.
- **datasets**: Manages and processes text datasets efficiently.
- **bitsandbytes**: Enables quantization (4-bit/8-bit) for reduced GPU memory usage.
- **peft**: Used for applying LoRA and other parameter-efficient fine-tuning methods.
- **accelerate**: Handles efficient multi-device and mixed precision training.
- **huggingface_hub**: Facilitates login and model download from Hugging Face.
- **torch (PyTorch)**: Core deep learning framework used for training and inference.
- **GPUtil**: Monitors GPU utilization and verifies hardware availability.

---

## Environment Setup

### 1. Install Dependencies
Run the following commands in Google Colab to install all required libraries:

```python
!pip install accelerate bitsandbytes datasets trl peft huggingface_hub transformers GPUtil

2. Checking GPU Availability

Before running any model operations, verify that the GPU is available and properly configured:

from GPUtil import showUtilization as gpu_usage
import torch, os

gpu_usage()
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

This ensures that the code runs on the GPU rather than the CPU for faster computation.

3. Hugging Face Login

Llama-2 requires authentication before downloading because it is a gated model. Use your Hugging Face account token to log in:

from huggingface_hub import notebook_login
notebook_login()


4. Loading the Base Model with Quantization

Load the pre-trained base model with 4-bit quantization to fit within Colab’s GPU memory:

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

base_model_id = "meta-llama/Llama-2-7b-chat-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

Quantization significantly reduces the memory footprint while maintaining most of the model’s performance.

5. Dataset Loading and Tokenization

First, clone the dataset repository and load the custom dataset file:

!git clone https://github.com/policolab/fine-tuning-LLMs.git

from datasets import load_dataset
train_dataset = load_dataset("text", data_files={"train": "./fine-tuning-LLMs/data/hawaii_wf.txt"})

Next, load the tokenizer and prepare the dataset for model training:

from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

tokenized_train_dataset = [tokenizer(t) for t in train_dataset['train']['text']]

Tokenization converts raw text into numerical tokens that the model can process.

6. Model Preparation for Fine-Tuning

Prepare the model for fine-tuning with LoRA by enabling gradient checkpointing and applying the LoRA configuration:

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

This step limits the number of trainable parameters, making fine-tuning efficient even with limited hardware.

7. Training the Model

Define the training loop using the Hugging Face Trainer API:

import transformers

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        save_total_limit=3,
        output_dir="experiments",
        optim="paged_adamw_8bit",
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False
trainer.train()

The Trainer handles training, evaluation, and checkpoint saving automatically.
The parameters above are optimized for low-memory environments such as Colab.

8. Loading the Fine-Tuned Model and Running Inference

Once training is complete, load the fine-tuned model for testing and generate predictions:

from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

tokenizer = LlamaTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True, add_eos_token=True)
model = PeftModel.from_pretrained(base_model, "experiments/checkpoint-28")

user_question = "When did Hawaii wildfires start?"
eval_prompt = f"Question: {user_question}\nAnswer: "
prompt_tokenized = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    output = model.generate(**prompt_tokenized, max_new_tokens=100)

print(tokenizer.decode(output[0], skip_special_tokens=True))

This code uses the fine-tuned model to answer questions related to Hawaii wildfires based on the information learned during fine-tuning.

9. Example Output

Question: When did Hawaii wildfires start?
Answer: The Hawaii wildfires began on August 8, 2023, due to dry conditions and strong winds...

10. Project Structure
├── data/
│   └── hawaii_wf.txt           # Custom dataset
├── experiments/                # Fine-tuning checkpoints
├── fine_tune_llama.ipynb       # Main Colab notebook
├── README.md                   # Project documentation

11. Requirements

Python version 3.10 or higher

GPU with at least 15 GB VRAM (Google Colab T4 recommended)

Hugging Face account for Llama model access

12. Acknowledgments

Meta AI for Llama 2 model release

Hugging Face for hosting and library support

Google Colab for providing free GPU resources

13. License

This project is open-source and available under the MIT License.


---


