#
# Modified from LLaVA/predict.py
# Please see ACKNOWLEDGEMENTS for details about LICENSE
#
import os
import argparse

import torch
from PIL import Image
import time

from src.model.llava.utils import disable_torch_init
from src.model.llava.conversation import conv_templates
from src.model.llava.model.builder import load_pretrained_model
from src.model.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from src.model.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from src.model.llava.processing_fastvlm import FastVLMProcessor
from src.model.processor import load_processor, FastVLM_process_fn
from src.model.model import MMEBModel

def predict(args):
    # Remove generation config from model folder
    # to read generation parameters from args
    model_path = os.path.expanduser(args.model_name)
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'),
                  generation_config)

    # Load model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device="cuda")
    model = MMEBModel.build(args)
    model.to("cuda")
    # print(f"Context length: {context_len}")
    
    # image_processor = model.get_vision_tower().image_processor
    model_encoder = model.encoder
    # Construct prompt
    qs1 = args.prompt
    qs2 = args.prompt + " here"
    qs3 = args.prompt + " there there"
    qs4 = args.prompt + " everywhere please describe"
    qs5 = args.prompt + " here here here"
    qs6 = args.prompt + " there there there"
    qs7 = args.prompt + " everywhere everywhere everywhere"
    qs8 = args.prompt + " here there everywhere"

    if model_encoder.config.mm_use_im_start_end:
        qs1 = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs1
        qs2 = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs2
        qs3 = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs3
        # qs4 = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs4
    else:
        qs1 = '\n' + qs1
        qs2 = DEFAULT_IMAGE_TOKEN + '\n' + qs2
        qs3 = '\n' + qs3
        qs4 = DEFAULT_IMAGE_TOKEN + '\n' + qs4
        qs5 = DEFAULT_IMAGE_TOKEN + '\n' + qs5
        qs6 = DEFAULT_IMAGE_TOKEN + '\n' + qs6
        qs7 = DEFAULT_IMAGE_TOKEN + '\n' + qs7
        qs8 = DEFAULT_IMAGE_TOKEN + '\n' + qs8

    prompts = [qs1, qs2, qs3, qs4, qs5, qs6, qs7, qs8]

    # Tokenize both prompts into a batch
    # input_ids = [
    #     tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    #     for p in prompts
    # ]
    # print("Tokenized input IDs:", input_ids)
    # # Pad to the same length
    # input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    # attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # input_ids = input_ids.to("cuda")
    # attention_mask = attention_mask.to("cuda")

    # print("Input IDs shape:", input_ids.shape)
    # print("Input IDs:", input_ids)
    # print("Attention Mask shape:", attention_mask.shape)
    # print("Attention Mask:", attention_mask)

    # # Load and preprocess four identical images (just for testing)
    image = Image.open("/workspace/ComfyUI/models/gligen/VLM_Embed/src/model/img.png").convert('RGB')
    images = [None, image, None, image, image, image, image, image]
    # image_tensors = process_images([None, image, None, image, image, image, image, image], image_processor, model.config)
    # image_tensors = image_tensors.to(model.device, dtype=model.dtype)
    # print(f"Image tensors shape: {image_tensors.shape}, type: {image_tensors.dtype}")
    # processor = FastVLMProcessor(image_processor=image_processor, tokenizer=tokenizer)
    # inputs = processor(
    #     images=images,
    #     texts=prompts,
    # )
    model_input = {
        'images': images,
        'text': prompts,
    }
    
    # processor = FastVLMProcessor(image_processor=image_processor, tokenizer=tokenizer)
    processor = load_processor(args) 
    inputs = FastVLM_process_fn(model_input, processor, max_length=2048)
    print(f"Last 20 input IDs: {inputs['input_ids'][:, -20:]}")
    input_ids = inputs['input_ids'].to(model_encoder.device)
    attention_mask = inputs['attention_mask'].to(model_encoder.device)
    image_tensors = inputs['pixel_values'].to(model_encoder.device, dtype=model_encoder.dtype)

    # Run inference
    with torch.inference_mode():
        start_time = time.time()
        out = model_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=image_tensors,
            output_attentions=True,
            output_hidden_states=True,
        )

    print("Key output:", out.keys())
    print("Output hidden state shape:", out.hidden_states[-1].shape)
    print("Number of hidden layers:", len(out.hidden_states))
    print("Output attentions shape:", out.attentions[-1].shape)
    print("Number of attention layers:", len(out.attentions))
    print("Generation time: %.2f seconds" % (time.time() - start_time))
    for i in range(len(out.batch_image_embeds)):
        if out.batch_image_embeds[i] is not None:
            print(f"Image features for sample {i} shape: {out.batch_image_embeds[i].shape}")
        else: 
            print(f"No image features for sample {i}")

    # Restore generation config
    # if generation_config is not None:
    #     os.rename(generation_config, os.path.join(model_path, 'generation_config.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="apple/FastVLM-0.5B")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--model-backbone", type=str, default="llava_qwen2")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--init-lora-model", type=bool, default=False)
    parser.add_argument("--lora", type=bool, default=False)
    parser.add_argument("--pooling", type=bool, default=True)
    parser.add_argument("--normalize", type=bool, default=True)
    # parser.add_argument("--image-file", type=str, default=None, help="location of image file")
    parser.add_argument("--prompt", type=str, default="Describe the image.", help="Prompt for VLM.")
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    predict(args)