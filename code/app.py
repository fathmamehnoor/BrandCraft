import gradio as gr
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
from tqdm import trange
from transformers.utils import move_cache
from diffusers import AutoPipelineForText2Image
import os

# Define lazy loading functions
def load_tokenizer():
    if not hasattr(load_tokenizer, "tokenizer"):
        load_tokenizer.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    return load_tokenizer.tokenizer

def load_model():
    if not hasattr(load_model, "model"):
        load_model.model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    return load_model.model

def load_pipeline():
    if not hasattr(load_pipeline, "pipeline"):
        load_pipeline.pipeline = AutoPipelineForText2Image.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
    return load_pipeline.pipeline

# Global variables for model path and special tokens
MODEL_PATH = '../models'
context_tkn = None
slogan_tkn = None

# Lazy initialization for special tokens
def initialize_special_tokens():
    global context_tkn, slogan_tkn
    if context_tkn is None or slogan_tkn is None:
        tokenizer = load_tokenizer()
        context_tkn = tokenizer.additional_special_tokens_ids[0]
        slogan_tkn = tokenizer.additional_special_tokens_ids[1]

# Utility function for top-k and top-p filtering
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

# Sequence sampling function
def sample_sequence(context, length=20, num_samples=1, temperature=1, top_k=0, top_p=0.0):
    initialize_special_tokens()
    tokenizer = load_tokenizer()
    model = load_model()

    input_ids = [context_tkn] + tokenizer.encode(context)
    input_ids += [slogan_tkn]  # Add the slogan token

    context_tensor = torch.tensor(input_ids).unsqueeze(0).repeat(num_samples, 1).to('cpu')
    generated = context_tensor

    with torch.no_grad():
        for _ in trange(length):
            outputs = model(generated)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.0)
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

    # Decode the generated tokens
    slogans = []
    for g in generated:
        slogan = tokenizer.decode(g.squeeze().tolist())
        slogan = slogan.split('<|endoftext|>')[0].split('<slogan>')[1] if '<slogan>' in slogan else slogan
        slogans.append(slogan.strip())
    return slogans

# Functions for generating slogans and logos
def generate_slogans(company_name, description):
    context = f"{company_name}, {description}"  # Combine company name and description to form context
    return sample_sequence(context, num_samples=5)  # Change num_samples as needed

def generate_logo_description(company_name, description):
    return f"A logo for {company_name}, a company that {description.lower()}."

def generate_logo(company_name, description):
    logo_description = generate_logo_description(company_name, description)
    pipeline = load_pipeline()
    image = pipeline(logo_description).images[0]
    return image

# Create Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# BrandCraft")
    
    with gr.Row():
        with gr.Column():
            company_name_input = gr.Textbox(label="Company Name", placeholder="Enter the company name")
            description_input = gr.Textbox(label="Description", placeholder="Enter a description of the company")
            slogan_output = gr.Textbox(label="Generated Slogans", interactive=False)
            generate_slogan_btn = gr.Button("Generate Slogans")

        with gr.Column():
            logo_output = gr.Image(label="Generated Logo", type="numpy")
            generate_logo_btn = gr.Button("Generate Logo")

    generate_slogan_btn.click(generate_slogans, inputs=[company_name_input, description_input], outputs=slogan_output)
    generate_logo_btn.click(generate_logo, inputs=[company_name_input, description_input], outputs=logo_output)

proxy_prefix = os.environ.get("PROXY_PREFIX", "/")
iface.launch(server_name="0.0.0.0", server_port=8080, root_path=proxy_prefix, share=True)
