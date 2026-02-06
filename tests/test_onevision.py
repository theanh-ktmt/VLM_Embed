# import requests
# from PIL import Image

# import torch
# from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
# model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
# model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True, 
# ).to(0)

# processor = AutoProcessor.from_pretrained(model_id)

# # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# # Each value in "content" has to be a list of dicts with types ("text", "image") 
# conversation = [
#     {
#       "role": "user",
#       "content": [
#           {"type": "text", "text": "What are these?"},
#           {"type": "image"},
#           {"type": "image"},

#         ],
#     },
# ]
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
# raw_image = Image.open(requests.get(image_file, stream=True).raw)

# images = [raw_image, raw_image]
# import ipdb; ipdb.set_trace()
# inputs = processor(images=images, text=prompt, return_tensors='pt').to(0, torch.float16)

# output = model(**inputs, return_dict=True, output_hidden_states=True)
# print(output.keys())
# print(output.hidden_states.keys())

import torch
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# 1. Khởi tạo
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
# Thiết lập padding_side='left' cho inference
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left" 
processor.tokenizer.pad_token = processor.tokenizer.eos_token # Đảm bảo có pad token

# 2. Dữ liệu Test
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_url, stream=True).raw)

conv_1 = [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image"}]}]
conv_2 = [{"role": "user", "content": [{"type": "text", "text": "Describe these two images briefly."}, {"type": "image"}, {"type": "image"}]}]

prompt_1 = processor.apply_chat_template(conv_1, add_generation_prompt=True)
prompt_2 = processor.apply_chat_template(conv_2, add_generation_prompt=True)

model_inputs = {
    "text": [prompt_1, prompt_2],
    "images": [raw_image, [raw_image, raw_image]]
}

# 3. Gọi hàm B (Giả sử hàm B đã được định nghĩa ở trên)
# processed_outputs = Llava_ONEVISION_process_fn(model_inputs, processor)

# 4. Hàm Collator chuẩn cho OneVision
def prepare_batch_for_model(processed_batch, device="cuda"):
    # Chú ý: OneVision dùng concat cho pixel_values vì mỗi image có thể ra số lượng patches khác nhau
    pixel_values = torch.cat([p for p in processed_batch["pixel_values"] if p is not None], dim=0)
    image_sizes = torch.cat([s for s in processed_batch["image_sizes"] if s is not None], dim=0)
    
    return {
        "input_ids": processed_batch["input_ids"].to(device),
        "attention_mask": processed_batch["attention_mask"].to(device),
        "pixel_values": pixel_values.to(device, dtype=torch.float16),
        "image_sizes": image_sizes.to(device)
    }


def Llava_ONEVISION_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes = [], [], []
    texts = model_inputs["text"]
    images = model_inputs["images"]
    image_token_id = processor.image_token_id
    image_token = processor.image_token
    
    # 1. Iterate each pair and process (Logic giống Qwen2_VL)
    for text, image in zip(texts, images):
        # Trường hợp: Không có ảnh hoặc list ảnh rỗng
        if image is None or (isinstance(image, list) and all(i is None for i in image)):
            # Process text only
            inputs = processor(
                images=None,
                text=text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            )
            
            # Xử lý input_ids (ensure list format)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                input_id = [input_id]
            input_ids.append(input_id)
            
            # Append None cho các phần visual
            pixel_values.append(None)
            image_sizes.append(None)
            
        else:
            # Trường hợp: Có ảnh
            # Lưu ý: processor của Llava OneVision thường nhận images dưới dạng list kể cả single sample
            if not isinstance(image, list):
                image = [image]

            # count number of image token in text
            num_image_token = text.count(image_token)
            if num_image_token > len(image):
                # remove extra image token
                text = text.replace(image_token, "", num_image_token - len(image))
            elif num_image_token < len(image):
                # add extra image token
                text = (image_token + '\n' ) * (len(image) - num_image_token) + text
            else:
                pass
                
            inputs = processor(
                images=image,
                text=text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
                do_resize=True,
    size={"height": 384, "width": 384},
            )
            
            # Append input_ids
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            
            # Append pixel_values (giữ nguyên Tensor của sample này)
            # Llava OneVision: (num_patches, C, H, W)
            pixel_values.append(inputs["pixel_values"])
            
            # Append image_sizes
            # inputs['image_sizes'] trả về Tensor (1, 2) hoặc (N, 2), ta lấy value ra
            if "image_sizes" in inputs:
                img_size = inputs["image_sizes"]
                # Chuẩn hóa về tensor hoặc list tuỳ collator, ở đây giữ nguyên logic lấy raw value
                image_sizes.append(img_size) 
            else:
                image_sizes.append(None)

    # 2. Padding text inputs (Batch processing for text)
    batch_encoding = processor.tokenizer.pad(
        {'input_ids': input_ids}, 
        return_tensors="pt"
    )
    
    # 3. Construct final dictionary
    inputs = {
        "input_ids": batch_encoding["input_ids"].long(),
        "attention_mask": batch_encoding["attention_mask"].long(),
        "texts": texts,
        "images": images,
        "pixel_values": pixel_values, # List of Tensors (or None)
        "image_sizes": image_sizes,   # List of Tensors (or None)
    }

    return inputs
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).to(0)

# 1. Chạy hàm B để lấy inputs
processed_outputs = Llava_ONEVISION_process_fn(model_inputs, processor)

# 2. Gom nhóm (Collator)
def prepare_for_forward(processed_batch, device="cuda"):
    pixel_values = torch.cat([p for p in processed_batch["pixel_values"] if p is not None], dim=0)
    image_sizes = torch.cat([s for s in processed_batch["image_sizes"] if s is not None], dim=0)
    
    return {
        "input_ids": processed_batch["input_ids"].to(device),
        "attention_mask": processed_batch["attention_mask"].to(device),
        "pixel_values": pixel_values.to(device, dtype=torch.float16),
        "image_sizes": image_sizes.to(device)
    }

final_inputs = prepare_for_forward(processed_outputs)

# 3. Forward pass
with torch.no_grad():
    outputs = model(**final_inputs, output_hidden_states=True, return_dict=True)

# 4. Trích xuất Last Hidden State
# Shape: (batch_size, sequence_length, hidden_size)
last_hidden_states = outputs.hidden_states[-1] 
print(last_hidden_states.shape)

# 5. Lấy embedding thực sự (không lấy PAD)
# Chúng ta dùng attention_mask để biết đâu là token cuối cùng hợp lệ
attention_mask = final_inputs["attention_mask"]
last_token_indices = attention_mask.sum(dim=1) - 1 # Tìm vị trí token cuối cùng không phải PAD

batch_embeddings = []
for i, idx in enumerate(last_token_indices):
    # Lấy vector tại đúng vị trí idx
    actual_embedding = last_hidden_states[i, idx, :]
    batch_embeddings.append(actual_embedding)

print(f"Số lượng vector: {len(batch_embeddings)}")
print(f"Kích thước mỗi vector: {batch_embeddings[0].shape}")


print(batch_embeddings[0].mean())
print(batch_embeddings[1].mean())