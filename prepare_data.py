import json
import sys
from collections import OrderedDict
from contextlib import contextmanager
import time
from PIL import Image

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoConfig

from src.model.model import MMEBModel
from src.data.dataset.mmeb_dataset import EvalDataset
from src.data.collator.eval_collator import EvalCollator
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
import datasets
from datasets import load_dataset, concatenate_datasets
from evaluation.mmeb_baselines.eval_utils import get_pred
from src.utils import print_rank
from src.model.processor import get_backbone_name, load_processor, COLPALI, PHI3V, VLM_IMAGE_TOKENS
from torch.nn.utils.rnn import pad_sequence
import shutil 

from transformers import ProcessorMixin

def batch_to_device(batch, device):
    _batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            _batch[k] = v.to(device)
        else:
            _batch[k] = v
    return _batch

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

class TextImageDataset(Dataset):
    def __init__(self, data_args, model_args, subset, text_field, img_path_field, mod_instruction=None):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        self.data_args = data_args
        self.model_args = model_args
        self.backbone = self.model_args.model_backbone

        self.eval_data = load_dataset(
            self.data_args.dataset_name,
            subset,
            split=self.data_args.dataset_split,
        )
        self.paired_data = self.get_paired_data(text_field, img_path_field)
        # self.paired_dataset = datasets.Dataset.from_dict({
        #     "text": [pair["text"] for pair in self.paired_data],
        #     "img_path": [pair["img_path"] for pair in self.paired_data]
        # })
        if(("tgt" in text_field) and (mod_instruction is not None) and self.data_args.tgt_prefix_mod):
            print("Using TGT mod", mod_instruction, flush=True)            
            self.paired_dataset = datasets.Dataset.from_dict({
                                "text": [mod_instruction + pair["text"] for pair in self.paired_data],
                                "img_path": [pair["img_path"] for pair in self.paired_data]
                                })
            print(">>>>>>>>>>>>>inside tgt_mod_txt", flush=True)
            # import ipdb; ipdb.set_trace()
        else:
            print("Not using TGT mod", mod_instruction, flush=True)            
            self.paired_dataset = datasets.Dataset.from_dict({
                                "text": [pair["text"] for pair in self.paired_data],
                                "img_path": [pair["img_path"] for pair in self.paired_data]
                                })

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, item):
        text, img_path = self.paired_dataset[item]["text"], self.paired_dataset[item]["img_path"]
        if self.backbone != PHI3V:
            text = text.replace(VLM_IMAGE_TOKENS[PHI3V], VLM_IMAGE_TOKENS[self.backbone])
        # if(self.backbone==INTERN_VL3):
        #     full_img_path = os.path.join(self.data_args.image_dir, img_path)
        #     return text, [full_img_path]
        return text, self._get_image(img_path)

    def _process_image(self, image, resolution):
        if image is None:
            return None
        if resolution == "high":
            image = image.resize((1344, 1344))
        else:
            image = image.resize((336, 336))
        return image

    def _get_image(self, img_path):
        if img_path == "":
            return None
        full_img_path = os.path.join(self.data_args.image_dir, img_path)
        image = Image.open(full_img_path)
        if self.model_args.model_backbone != PHI3V and self.data_args.image_resolution:
            return process_image(image, self.data_args.image_resolution)
        else:
            return image
        return image

    def get_paired_data(self, text_field, img_path_field):
        """
        (text_field, image_field) -> ("qry_text", "qry_img_path") or ("tgt_text", "tgt_img_path")
        """
        paired_data = []
        for row in self.eval_data:
            if isinstance(row[text_field], str):
                if isinstance(row[img_path_field], list):
                    for img_path in row[img_path_field]:
                        paired_data.append({"text": row[text_field], "img_path": img_path})
                else:
                    paired_data.append({"text": row[text_field], "img_path": row[img_path_field]})
            elif isinstance(row[text_field], list):
                assert isinstance(row[img_path_field], list) and len(row[img_path_field]) == len(row[text_field])
                for text, img_path in zip(row[text_field], row[img_path_field]):
                    paired_data.append({"text": text, "img_path": img_path})
        return paired_data
    

def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(data_args.encode_output_path, exist_ok=True)
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
        model_backbone = get_backbone_name(hf_config=hf_config, model_type=model_args.model_type)
        setattr(model_args, 'model_backbone', model_backbone)
        setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_args.model_backbone}')
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)
    
    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )
    
    for idx, subset in enumerate(data_args.subset_name):
        
        encode_output_path = os.path.join(data_args.encode_output_path, f"{subset}_{data_args.dataset_split}_encoded.pkl")
        if os.path.exists(encode_output_path):
            print_rank(f"Found existing encoded file: {encode_output_path}, skipping...")
            continue
        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        eval_qry_dataset = TextImageDataset(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="qry",
            img_path_field="qry_image_path",
        )
        
        eval_tgt_dataset = TextImageDataset(
            data_args=data_args,
            model_args=model_args,
            subset=subset,
            text_field="pos_text",
            img_path_field="pos_image_path",
        )
        
        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        
        
        if not os.path.exists(encode_output_path):
            encoded_qry_tensor = []
            with torch.no_grad():
                for batch in tqdm(eval_qry_loader, desc=f"Encoding {subset} query set"):
                    batch = batch_to_device(batch, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(qry=batch)
                    encoded_qry_tensor.append(output['qry_reps'].cpu().detach().to(torch.float16))
                encoded_qry_tensor = np.concatenate([x.numpy() for x in encoded_qry_tensor])
            print(f"encoded_qry_tensor shape: {encoded_qry_tensor.shape}")    
            
            encoded_tgt_tensor = []
            with torch.no_grad():
                for batch in tqdm(eval_tgt_loader, desc=f"Encoding {subset} target set"):
                    batch = batch_to_device(batch, training_args.device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        output = model(tgt=batch)
                    encoded_tgt_tensor.append(output['tgt_reps'].cpu().detach().to(torch.float16))
                encoded_tgt_tensor = np.concatenate([x.numpy() for x in encoded_tgt_tensor])
            print(f"encoded_tgt_tensor shape: {encoded_tgt_tensor.shape}")
            assert len(eval_qry_dataset) == len(encoded_qry_tensor), f"len(eval_qry_dataset)={len(eval_qry_dataset)} vs len(encoded_qry_tensor)={len(encoded_qry_tensor)}"
            assert len(eval_tgt_dataset) == len(encoded_tgt_tensor), f"len(eval_tgt_dataset)={len(eval_tgt_dataset)} vs len(encoded_tgt_tensor)={len(encoded_tgt_tensor)}"
            with open(encode_output_path, "wb") as f:
                pickle.dump({
                    'qry_reps': encoded_qry_tensor,
                    'tgt_reps': encoded_tgt_tensor,
                    'index': list(range(len(eval_qry_dataset))),
                }, f)
            print_rank(f"Encoded file saved to {encode_output_path}")
        else:
            print_rank(f"Encoded file already exists: {encode_output_path}, skipping...")
            
if __name__ == "__main__":
    main()