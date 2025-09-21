import os
import io
from typing import Dict, Tuple, Optional
import time
import json
import pickle
from datasets import load_dataset, concatenate_datasets
import torch
import torch.nn as nn
import PIL
import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser
)
from peft import (
    PeftModel,
    LoraConfig,
    TaskType,
    get_peft_model
)
from src.model.model import MMEBModel
from src.model.processor import VLM_IMAGE_TOKENS, load_processor, get_backbone_name, process_vlm_inputs_fns, backbone2model, \
    LLAVA_NEXT, QWEN2_VL, LLAVA_ONEVISION, QWEN2_5_VL_TOKENSELECTION, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, PHI3V
from src.data.collator.train_collator import MultimodalDataCollator, TrainTextImageDataCollator
from src.data.dataset.mmeb_dataset import TrainTextImageDataset
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from src.utils import print_rank, print_master
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel 
from transformers import ProcessorMixin
from qwen_vl_utils import smart_resize
from PIL import Image

def process_image(image, resolution, max_dim=1344):
    if image is None:
        return None
    if resolution == "high":
        image = image.resize((1344, 1344))
    elif resolution == "mid":
        image = image.resize((672, 672))
    elif resolution == "low":
        image = image.resize((128, 128))
    else:
        cur_max_dim = max(image.size)
        if cur_max_dim > max_dim:
            image = image.resize((max_dim, max_dim))
    return image

class Distiller(nn.Module):
    def __init__(self, model_args, training_args, device):
        super(Distiller, self).__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.device = device
        self.student = self._load_student()
        self.temperature = model_args.temperature
    
    def _load_student(self):
        print("Load student with lora rank:", self.model_args.lora_r)
        print("Student use lora:", self.model_args.lora)
        student = MMEBModel.build(self.model_args, is_trainable=True)
        print("Student model built.")
        return student 
    def get_student_processor(self):
        processor = load_processor(self.model_args, None)
        print("Student processor loaded.")
        return processor
    
    def student_forward(self, qry: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]= None, *args, **kwargs):
        qry_reps = self.student.encode_input(qry) if qry is not None else None
        tgt_reps = self.student.encode_input(tgt) if tgt is not None else None
        
        scores = self.student.compute_similarity(qry_reps, tgt_reps)
        scores = scores.view(qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (qry_reps.size(0) // tgt_reps.size(0))
        loss = nn.CrossEntropyLoss()(scores / self.temperature, target)
        
        return {"contrastive_loss": loss, "stu_qry_reps": qry_reps, "stu_tgt_reps": tgt_reps}
    
    def compute_loss(self, teacher_qry_reps, teacher_pos_reps, stu_qry: Dict[str, torch.Tensor], stu_tgt: Dict[str, torch.Tensor]= None,
                     *args, **kwargs):
        student_outputs = self.student_forward(stu_qry, stu_tgt, *args, **kwargs)
        
        loss = student_outputs["contrastive_loss"]
        kd_loss_1 = nn.MSELoss()(torch.matmul(student_outputs["stu_qry_reps"], student_outputs["stu_qry_reps"].t()),
                                 torch.matmul(teacher_qry_reps, teacher_qry_reps.t()))
        kd_loss_2 = nn.MSELoss()(torch.matmul(student_outputs["stu_tgt_reps"], student_outputs["stu_tgt_reps"].t()),
                                 torch.matmul(teacher_pos_reps, teacher_pos_reps.t()))
        kd_loss = self.training_args.kd_weight * (kd_loss_1 + kd_loss_2)
        loss += kd_loss
        
        return {
            'loss': loss,
            'contrastive_loss': student_outputs["contrastive_loss"],
            'kd_loss': kd_loss
        }
    def forward(self, teacher_qry_reps, teacher_pos_reps, stu_qry: Dict[str, torch.Tensor], stu_tgt: Dict[str, torch.Tensor]= None,
                *args, **kwargs):
        return self.compute_loss(teacher_qry_reps, teacher_pos_reps, stu_qry, stu_tgt, *args, **kwargs)

class DistillationCollator:
    def __init__(self, student_processor: ProcessorMixin, 
                 model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments,
                 batch_size: Optional[int] = None):
        self.student_processor = student_processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.batch_size = batch_size
    
    def _get_batch_inputs(self, batch, text_keyname, image_keyname):
        # print("Processing batch for keys:", text_keyname, image_keyname)
        texts, visual_inputs = [], []
        for example in batch:
            if example is None or not example:
                text, visual_input = ' ', None
                texts.append(text)
                visual_inputs.append(visual_input)
            else:
                text, raw_images = example[text_keyname], example[image_keyname]
                visual_input = []
                for image in raw_images:
                    if image is None:
                        visual_input.append(None)
                    else:
                        visual_input.append(image)
                texts.extend(text)
                visual_inputs.extend(visual_input)
        inputs = {'text': texts, 'images': visual_inputs}
        return inputs
    
    def __call__(self, examples):
        student_qry_inputs = self._get_batch_inputs(examples, "student_query_text", "student_query_image")
        student_pos_inputs = self._get_batch_inputs(examples, "student_pos_text", "student_pos_image")

        bs = len(student_qry_inputs['text'])
        assert bs > 0, 'An empty batch is detected!'
        
        if self.batch_size is not None and bs < self.batch_size:
            raise RuntimeError(f"Expected batch size {self.batch_size}, but got {bs}.")
        
        process_student_fn = process_vlm_inputs_fns[self.model_args.model_backbone]
        processed_student_qry_inputs = process_student_fn(student_qry_inputs, processor=self.student_processor, max_length=self.data_args.max_len)
        processed_student_pos_inputs = process_student_fn(student_pos_inputs, processor=self.student_processor, max_length=self.data_args.max_len)

        # processed_student_qry_inputs['text'] = [e['student_query_text'] for e in examples]
        # processed_student_pos_inputs['text'] = [e['student_pos_text'] for e in examples]
        # processed_student_qry_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        # processed_student_pos_inputs['global_dataset_name'] = [e['global_dataset_name'] for e in examples]
        teacher_qry_reps = torch.stack([e["teacher_qry_reps"] for e in examples]).to(dtype=torch.bfloat16)
        teacher_pos_reps = torch.stack([e["teacher_pos_reps"] for e in examples]).to(dtype=torch.bfloat16)
        return {
            'qry': processed_student_qry_inputs,
            'pos': processed_student_pos_inputs,
            'teacher_qry_reps': teacher_qry_reps,
            'teacher_pos_reps': teacher_pos_reps,
        }
        
class DistillationDataset(Dataset):
    def __init__(self, data_args, model_args):
        self.data_args = data_args
        self.model_args = model_args
        train_data = []
        
        for subset in data_args.subset_name:
            subset_data = load_dataset(
                self.data_args.dataset_name, 
                subset,
                split=f"{self.data_args.dataset_split}"
            )
            subset_data = subset_data.remove_columns(set(['neg_text', 'neg_image_path']) & set(subset_data.column_names))
            encoded_file_path = f"/workspace/VLM_Embed/encoded_data/B2_Qwen2_2B/{subset}_{self.data_args.dataset_split}_encoded.pkl"
            with open(encoded_file_path, "rb") as f:
                data = pickle.load(f)
            qry_reps = data['qry_reps']
            pos_reps = data['tgt_reps']
            
            assert len(subset_data) == len(qry_reps) == len(pos_reps), "Mismatch in dataset length!"
            subset_data = subset_data.add_column("teacher_qry_reps", qry_reps.tolist())
            subset_data = subset_data.add_column("teacher_pos_reps", pos_reps.tolist())
            
            train_data.append(subset_data)
            
        self.train_data = concatenate_datasets(train_data)
        print(f"Loaded {len(self.train_data)} samples from {self.data_args.dataset_name} with subsets {self.data_args.subset_name}")
    
    def __len__(self):
        return len(self.train_data)
    def _get_image(self, img_path):
        if not img_path:
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        backbone = self.model_args.model_backbone
        if backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
        
    def __getitem__(self, data_idx):
        # print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>get image called, {data_idx}", flush=True)
        
        qry_texts, qry_image_paths, pos_texts, pos_image_paths = (
            self.train_data[data_idx]["qry"], self.train_data[data_idx]["qry_image_path"],
            self.train_data[data_idx]["pos_text"], self.train_data[data_idx]["pos_image_path"]
        )

        if not isinstance(qry_texts, list):
            qry_texts = [qry_texts]
            qry_image_paths = [qry_image_paths]
            pos_texts = [pos_texts]
            pos_image_paths = [pos_image_paths]
            
        student_qry_texts, student_qry_images, student_pos_texts, student_pos_images = [], [], [], []
                
        student_backbone = self.model_args.model_backbone

        for qry_text, qry_image_path, pos_text, pos_image_path \
            in zip(qry_texts, qry_image_paths, pos_texts, pos_image_paths):
            # instructions were hardcoded with Phi3 image special tokens
            # Update image token for llava and colqwen2, qwenvl
            if student_backbone != PHI3V:
                stu_qry_text = qry_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone])
                stu_pos_text = pos_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone])
                # stu_neg_text = neg_text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[student_backbone]) if neg_text else None
            stu_qry_image = self._get_image(qry_image_path)
            stu_pos_image = self._get_image(pos_image_path)
            # stu_neg_image = self._get_image(neg_image_path) if neg_image_path else None
            if (not stu_qry_text and not stu_qry_image) or (not stu_pos_text and not stu_pos_image):
                print("empty inputs")
                continue
            student_qry_texts.append(stu_qry_text)
            student_qry_images.append(stu_qry_image)
            student_pos_texts.append(stu_pos_text)
            student_pos_images.append(stu_pos_image)
        
        teacher_qry_reps = torch.tensor(self.train_data[data_idx]['teacher_qry_reps'], dtype=torch.float16)
        teacher_pos_reps = torch.tensor(self.train_data[data_idx]['teacher_pos_reps'], dtype=torch.float16)
            
        return {
            "student_query_text": student_qry_texts,
            "student_query_image": student_qry_images,
            "student_pos_text": student_pos_texts,
            "student_pos_image": student_pos_images,
            "teacher_qry_reps": teacher_qry_reps,
            "teacher_pos_reps": teacher_pos_reps
        }