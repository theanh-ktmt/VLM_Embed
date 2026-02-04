import torch
import PIL.Image
import numpy as np
from transformers import Qwen2VLProcessor
# from transformers.image_utils import ChannelDimension # <-- KHÔNG CẦN DÒNG NÀY NỮA

# --- 1. MOCK CÁC CONSTANT ---
QWEN2_VL = "qwen2-vl"
VLM_IMAGE_TOKENS = {QWEN2_VL: "<|image_pad|>"}
VLM_VIDEO_TOKENS = {QWEN2_VL: "<|video_pad|>"}

# --- 2. HÀM ĐÃ FIX LỖI ---
def Qwen2_VL_process_fn(model_inputs: dict, processor: Qwen2VLProcessor, max_length=None):
    input_ids, pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = [], [], [], [], []
    texts, visual_inputs = model_inputs['text'], model_inputs['images']
    
    vlm_image_token, vlm_video_token = VLM_IMAGE_TOKENS[QWEN2_VL], VLM_VIDEO_TOKENS[QWEN2_VL]

    for text, images in zip(texts, visual_inputs):
        # Case 1: Text only
        if images is None or (type(images)==list and any(i is None for i in images)):
            inputs = processor(text=[text], images=None, return_tensors="pt", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int): input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_grid_thw.append(None)
            pixel_values_videos.append(None)
            video_grid_thw.append(None)
        else:
            # Case 2: Image Input
            if vlm_image_token in text:
                if isinstance(images, PIL.Image.Image):
                    images = [images]
                    
                for iid, image in enumerate(images):
                    # Đảm bảo ảnh luôn là RGB
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Logic resize đặc biệt
                    if image.size[0] < 28 or image.size[1] < 28:
                        print(f"-> Resize ảnh nhỏ {image.size} lên (56, 56)")
                        image = image.resize((56, 56))
                        images[iid] = image
                
                # --- FIX: BỎ input_data_format=ChannelDimension.LAST ---
                # Processor tự động xử lý PIL Image rất tốt, không cần ép format thủ công
                inputs = processor(text=[text], images=images, return_tensors="pt", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            
            # Case 3: Video Input
            elif vlm_video_token in text:
                # Tương tự cho video
                inputs = processor(text=[text], videos=[images], return_tensors="pt", max_length=None, truncation=False, input_data_format=ChannelDimension.LAST)
            else:
                raise NotImplementedError
            
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            
            if 'pixel_values' in inputs:
                pixel_values.append(inputs['pixel_values'])
                image_grid_thw.append(inputs['image_grid_thw'])
                pixel_values_videos.append(None)
                video_grid_thw.append(None)
            else:
                pixel_values.append(None)
                image_grid_thw.append(None)
                pixel_values_videos.append(inputs['pixel_values_videos'])
                video_grid_thw.append(inputs['video_grid_thw'])

    # Padding
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(), 
        'texts': texts,
        'images': visual_inputs,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'pixel_values_videos': pixel_values_videos,
        'video_grid_thw': video_grid_thw
    }
    return inputs

# --- 3. TEST CASE ---
if __name__ == "__main__":
    print("--- Đang tải Processor ---")
    import ipdb; ipdb.set_trace()
    try:
        processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    except:
        print("Lỗi tải model. Kiểm tra mạng.")
        exit()

    # Tạo ảnh RGB chuẩn
    normal_image = PIL.Image.new('RGB', (512, 512), color='red')
    tiny_image = PIL.Image.new('RGB', (10, 10), color='blue')
    
    test_inputs = {
        'text': [
            "Text only sample.",
            f"Image normal: {VLM_IMAGE_TOKENS[QWEN2_VL]} desc.",
            f"Image tiny: {VLM_IMAGE_TOKENS[QWEN2_VL]} desc." * 30,
        ],
        'images': [
            None,
            normal_image,
            tiny_image,
        ]
    }

    print("\n--- Processing ---")
    # Chạy hàm
    out = Qwen2_VL_process_fn(test_inputs, processor, max_length=128)

    print("\n--- SUCCEEDED ---")
    print(f"Batch Size: {out['input_ids'].shape[0]}")
    # In ra kích thước pixel_values của ảnh normal (index 1)
    # Lưu ý: Qwen2-VL flatten patches nên shape sẽ là (num_patches, hidden_dim) chứ ko phải (C, H, W)
    if out['pixel_values'][1] is not None:
        print(f"Shape pixel_values (Normal): {out['pixel_values'][1].shape}")
        print(f"Grid THW (Normal): {out['image_grid_thw'][1]}")
    
    if out['pixel_values'][2] is not None:
        print(f"Shape pixel_values (Tiny - Resized): {out['pixel_values'][2].shape}")


    print(out['input_ids'])