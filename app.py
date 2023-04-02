import torch
import gradio as gr
import re
import transformers
import peft
import traceback
import argparse

from queue import Queue
from threading import Thread
import gc

CUDA_AVAILABLE = torch.cuda.is_available()

device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

tokenizer = transformers.AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-2.7B")
tokenizer.pad_token_id = 0

model = transformers.AutoModelForCausalLM.from_pretrained(
    "cerebras/Cerebras-GPT-2.7B", 
    load_in_8bit=True, 
    torch_dtype=torch.float16,
    device_map={'':0} if CUDA_AVAILABLE else 'auto',
)

model = peft.PeftModel.from_pretrained(
    model,
    'lxe/lora-cerebras-gpt2.7b-alpaca-shortprompt',
    torch_dtype=torch.float16
)

model.half()

# Streaming functionality taken from https://github.com/oobabooga/text-generation-webui/blob/master/modules/text_generation.py#L105

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False

class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """
    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc=func
        self.c_callback=callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                traceback.print_exc()
                pass
            except:
                traceback.print_exc()
                pass

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True,None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()

def clear_torch_cache():
    gc.collect()
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()

def generate_text(
    history,  
    max_new_tokens, 
    do_sample, 
    temperature, 
    top_p, 
    top_k, 
    repetition_penalty, 
    typical_p, 
    num_beams
):
    # Create a conversation context of the last 4 entries in the history
    inp = ''.join([
        f"Human: {h[0]}\n\nAssistant: {'' if h[1] is None else h[1]}\n\n" for h in history[-4:]
    ]).strip()
     
    input_ids = tokenizer.encode(
        inp, 
        return_tensors='pt', 
        truncation=True, 
        add_special_tokens=False
    ).to(device) # type: ignore

    generate_params = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "typical_p": typical_p,
        "num_beams": num_beams,
        "stopping_criteria": transformers.StoppingCriteriaList(),
        "pad_token_id": tokenizer.pad_token_id,
    }

    def generate_with_callback(callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        clear_torch_cache()
        with torch.no_grad():
            model.generate(**kwargs) # type: ignore

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            new_tokens = len(output) - len(input_ids[0])
            reply = tokenizer.decode(output[-new_tokens:], skip_special_tokens=True)

            # If reply contains '^Human:' or '^Assistant:' 
            # then we have reached the end of the assistant's response
            stop_re = re.compile(r'^(Human|Assistant):', re.MULTILINE)
            if re.search(stop_re, reply):
                reply = ''.join(reply.split('\n')[:-1])
                history[-1][1] = reply.strip()
                yield history
                break

            # if reply contains 'EOS' then we have reached the end of the conversation
            if output[-1] in [tokenizer.eos_token_id]:
                yield history
                break

            history[-1][1] = reply.strip()
            yield history

with gr.Blocks() as demo:
    gr.Markdown("""
    ## 🐺🦙 Cerebras GPT-2.7B Alpcaca-Shortprompt LoRA Chatbot
    This is a very fast and relatively coherent (but hallucinating) chatbot. 
    It uses the [Cerebras-GPT-2.7B](https://huggingface.co/cerebras/Cerebras-GPT-2.7B), 
    with a LoRA finetuned on the [Alpcaca Dataset](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data_cleaned.json) dataset 
    using a shorter prompt. The chatbot keeps a very short conversation context of 4 entries. It's the fastest chatbot in the west! 
    More info [here](https://github.com/lxe/cerebras-lora-alpaca)
    """)
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(value="How old is the Earth?", placeholder="Type a message...")
            with gr.Row():
                clear = gr.Button("Clear")

        with gr.Column():
            max_new_tokens = gr.Slider(0, 2048, 200, step=1, label="max_new_tokens")
            do_sample = gr.Checkbox(True, label="do_sample")
            with gr.Row():
                with gr.Column():
                    temperature = gr.Slider(0, 2, 0.1, step=0.01, label="temperature")
                    top_p = gr.Slider(0, 1, 0.8, step=0.01, label="top_p")
                    top_k = gr.Slider(0, 100, 35, step=1, label="top_k")
                with gr.Column():
                    repetition_penalty = gr.Slider(0, 10, 1.1, step=0.01, label="repetition_penalty")
                    typical_p = gr.Slider(0, 1, 1, step=0.01, label="typical_p")
                    num_beams = gr.Slider(0, 10, 1, step=1, label="num_beams")

    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    def fix_history(history):
        update_history = False
        for i, (user, bot) in enumerate(history):
            if bot is None:
                update_history = True
                history[i][1] = "_silence_"
        if update_history:
            chatbot.update(history) 

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        generate_text, inputs=[
            chatbot,
            max_new_tokens, 
            do_sample, 
            temperature, 
            top_p, 
            top_k, 
            repetition_penalty, 
            typical_p, 
            num_beams
        ], outputs=[chatbot],
    ).then(fix_history, chatbot)

    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chatbot Demo")
    parser.add_argument("-s", "--share", action="store_true", help="Enable sharing of the Gradio interface")
    args = parser.parse_args()

    demo.queue().launch(share=args.share)