import os
import json
import torch
import torch.nn as nn
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
from utils import log_rank
from src.model.model import MMEBModel
from src.model.processor import load_processor, get_backbone_name, process_vlm_inputs_fns, backbone2model, \
    LLAVA_NEXT, QWEN2_VL, LLAVA_ONEVISION, QWEN2_5_VL_TOKENSELECTION, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION
from src.data.collator.train_collator import MultimodalDataCollator, TrainTextImageDataCollator
from src.data.dataset.mmeb_dataset import TrainTextImageDataset
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from src.utils import print_rank, print_master
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel 

def add_distiller_arguments(parser):
    """Thêm arguments cho Distiller"""
    # Student model arguments
    parser.add_argument('--student_model_path', type=str, required=True,
                       help='Path to student model')
    parser.add_argument('--student_checkpoint_path', type=str, default=None,
                       help='Path to student checkpoint')
    parser.add_argument('--student_lora', action='store_true',
                       help='Whether student uses LoRA')
    
    # Teacher model arguments  
    parser.add_argument('--teacher_model_path', type=str, required=True,
                       help='Path to teacher model')
    parser.add_argument('--teacher_checkpoint_path', type=str, default=None,
                       help='Path to teacher checkpoint')
    parser.add_argument('--teacher_lora', action='store_true',
                       help='Whether teacher uses LoRA')
    
    # Common model arguments
    parser.add_argument('--pooling', type=str, default='last',
                       help='Pooling strategy')
    parser.add_argument('--normalize', action='store_true',
                       help='Whether to normalize embeddings')
    parser.add_argument('--temperature', type=float, default=0.02,
                       help='Temperature for similarity')
    parser.add_argument('--model_type', type=str, default=None,
                       help='Model type')
    
    return parser

class Distiller(nn.Module):
    def __init__(self, args, device):
        super(Distiller, self).__init__()
        self.args = args
        self.device = device
        self.student = self._load_student()
        self.teacher = self._load_teacher()
        self.student.to(self.device)
        self.teacher.to(self.device)
        self.temperature = args.temperature
    
    def _create_model_args(self, model_type='student'):
        """Tạo ModelArguments từ args hiện tại"""
        if model_type == 'student':
            model_args = ModelArguments(
                model_name=self.args.student_model_path,
                checkpoint_path=getattr(self.args, 'student_checkpoint_path', None),
                lora=self.args.student_lora,
                pooling=self.args.pooling,
                normalize=self.args.normalize,
                temperature=self.args.temperature,
                model_type=getattr(self.args, 'model_type', None)
            )
        else:  # teacher
            model_args = ModelArguments(
                model_name=self.args.teacher_model_path,
                checkpoint_path=getattr(self.args, 'teacher_checkpoint_path', None),
                lora=self.args.teacher_lora,
                pooling=self.args.pooling,
                normalize=self.args.normalize,
                temperature=self.args.temperature,
                model_type=getattr(self.args, 'model_type', None)
            )
        return model_args
    
    def _load_teacher(self):
        model_args = self._create_model_args('teacher')
        teacher = MMEBModel.load(model_args, is_trainable=False)
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        return teacher
    
    def _load_student(self):
        model_args = self._create_model_args('student')
        student = MMEBModel.load(model_args, is_trainable=True)
        return student 
    
def main():
    parser = argparse.ArgumentParser(description="Distiller Training")
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser = add_distiller_arguments(parser)
    parser.add_argument('')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    distiller = Distiller(args, device)
    print(f"Distiller initialized on device: {distiller.device}")
    print(f"Loaded student model: {args.student_model_path}")
    print(f"Loaded teacher model: {args.teacher_model_path}")
    print(f"Student device: {distiller.student.device}")
    print(f"Teacher device: {distiller.teacher.device}")
