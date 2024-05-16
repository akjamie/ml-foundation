from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import dotenv

dotenv.load_dotenv()

def prompt_llama3(prompt: str):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained( "meta-llama/Meta-Llama-3-8B", device_map = 'auto')
    inputs = tokenizer(prompt, return_tensors="pt")#.to("cuda")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=2000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def prompt_llama2(prompt):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained( "meta-llama/Llama-2-7b-chat-hf", device_map = 'auto')
    inputs = tokenizer(prompt, return_tensors="pt")#.to("cuda")
    outputs = model.generate(inputs["input_ids"], max_new_tokens=2000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == '__main__':
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    # prompt_llama2("What is the capital of France?")
    prompt_llama3("What is the capital of France?")