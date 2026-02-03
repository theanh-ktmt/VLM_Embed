"""
This module defines the PGA-KD (Principal Geometry Alignment) criterion for VLM2Vec Distillation.
It implements:
1. PGA: Spectral denoising and geometric alignment via CKA.
2. SCL: Semantic Consistency Learning via Mutual Information Maximization with specialized Cross-Modal Projectors.
3. Base: Standard InfoNCE + MSE.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Type
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedModel, ProcessorMixin
from ..arguments import SSAAguments

logger = logging.getLogger(__name__)

class SCLProjectors(nn.Module):
    """
    Lightweight projection heads for Semantic Consistency Learning (SCL).
    
    Structure:
    - Intra-modal Projectors: Align same-modality features (Img-Img, Txt-Txt).
    - Cross-modal Projectors: Align across modalities (Img-Txt, Txt-Img).
    
    Uses simple Linear layers initialized robustly to ensure stable gradient flow.
    """
    def __init__(self, t_dim: int, s_dim: int):
        super().__init__()
        
        # --- Intra-Modal Projectors ---
        # Maps Teacher Img -> Student Space (for matching Student Img)
        self.img_proj = nn.Linear(t_dim, s_dim, bias=False)
        # Maps Teacher Txt -> Student Space (for matching Student Txt)
        self.txt_proj = nn.Linear(t_dim, s_dim, bias=False)

        # --- Cross-Modal Projectors ---
        # Maps Teacher Img -> Student Space (for matching Student Txt)
        self.cross_img_proj = nn.Linear(t_dim, s_dim, bias=False)
        # Maps Teacher Txt -> Student Space (for matching Student Img)
        self.cross_txt_proj = nn.Linear(t_dim, s_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        """
        Robust initialization using Kaiming Normal (He initialization).
        """
        for m in [self.img_proj, self.txt_proj, self.cross_img_proj, self.cross_txt_proj]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, t_img: torch.Tensor, t_txt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns 4 projections:
        1. t_img_intra: Teacher Image projected for Image alignment
        2. t_txt_intra: Teacher Text projected for Text alignment
        3. t_img_cross: Teacher Image projected for Text alignment (Cross)
        4. t_txt_cross: Teacher Text projected for Image alignment (Cross)
        """
        t_img_intra = self.img_proj(t_img)
        t_txt_intra = self.txt_proj(t_txt)
        
        t_img_cross = self.cross_img_proj(t_img)
        t_txt_cross = self.cross_txt_proj(t_txt)
        
        return t_img_intra, t_txt_intra, t_img_cross, t_txt_cross


class PGAKDLoss(nn.Module):
    """
    Implements the PGA-KD framework:
    Total Loss = L_Base + lambda_PGA * L_PGA + lambda_SCL * L_SCL
    """

    def __init__(self, args: Type[SSAAguments], teacher_dim: int = 4096, student_dim: int = 1024):
        """
        Args:
            args: Configuration arguments.
            teacher_dim: Hidden dimension of the teacher model.
            student_dim: Hidden dimension of the student model.
        """
        super(PGAKDLoss, self).__init__()
        self.args = args
        
        # --- Weights ---
        self.pga_weight = getattr(args, 'pga_loss_weight', getattr(args, 'kd_weight', 1.0))
        self.scl_weight = getattr(args, 'scl_loss_weight', 1.0)
        self.mse_weight = getattr(args, 'mse_loss_weight', 0.3)
        
        # --- PGA Hyperparameters ---
        self.variance_threshold = getattr(args, 'pga_spectral_variance_threshold', 0.95)

        # --- Components ---
        self.mse_loss_fn = nn.MSELoss(reduction='mean')
        
        # SCL Projectors
        self.scl_projectors = SCLProjectors(teacher_dim, student_dim)

        # --- Distributed Setup ---
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

        logger.info(f"PGAKDLoss initialized | PGA W: {self.pga_weight} | SCL W: {self.scl_weight} | MSE W: {self.mse_weight}")

    def _dist_gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Gathers tensors from all GPUs."""
        if self.world_size <= 1:
            return t
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing Base, PGA, and SCL losses.
        """
        student_model: PreTrainedModel = distiller.student
        teacher_model: PreTrainedModel = distiller.teacher

        # --- 1. Forward Pass & Feature Extraction ---
        with torch.no_grad():
            teacher_model.eval()
            t_qry, _, _, t_qry_hiddens = teacher_model.encode_input(input_data['teacher_inputs']['qry'])
            t_pos, _, _, t_pos_hiddens = teacher_model.encode_input(input_data['teacher_inputs']['pos'])

        s_qry, _, _, s_qry_hiddens = student_model.encode_input(input_data['student_inputs']['qry'])
        s_pos, _, _, s_pos_hiddens = student_model.encode_input(input_data['student_inputs']['pos'])

        # --- 2. Base Objective: Contrastive (InfoNCE) ---
        if self.world_size > 1:
            all_s_qry = self._dist_gather_tensor(s_qry)
            all_s_pos = self._dist_gather_tensor(s_pos)
        else:
            all_s_qry = s_qry
            all_s_pos = s_pos

        scores = student_model.compute_similarity(all_s_qry, all_s_pos)
        scores = scores.view(all_s_qry.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_s_pos.size(0) // all_s_qry.size(0))
        
        loss_infonce = nn.CrossEntropyLoss()(scores / distiller.temperature, target)

        # --- 3. Base Objective: MSE ---
        loss_mse = torch.tensor(0.0, device=s_qry.device)
        if hasattr(distiller, "projectors") and "t2s_txt" in distiller.projectors:
            projected_t_qry = distiller.projectors["t2s_txt"](t_qry)
            loss_mse = self.mse_loss_fn(s_qry, projected_t_qry)

        # --- 4. PGA Objective: Principal Geometry Alignment ---
        loss_pga = self._compute_pga_loss(s_qry, s_pos, t_qry, t_pos)

        # --- 5. SCL Objective: Semantic Consistency Learning ---
        loss_scl = self._compute_scl_loss(
            s_qry_hiddens, t_qry_hiddens, 
            input_data['student_inputs']['qry']['input_ids'],
            input_data['teacher_inputs']['qry']['input_ids'],
            student_model.processor if hasattr(student_model, 'processor') else None,
            distiller.temperature
        )

        # --- Total Loss ---
        total_loss = loss_infonce + \
                     (self.mse_weight * loss_mse) + \
                     (self.pga_weight * loss_pga) + \
                     (self.scl_weight * loss_scl)

        return {
            "loss": total_loss,
            "contrastive_loss": loss_infonce,
            "mse_loss": loss_mse,
            "pga_loss": loss_pga,
            "scl_loss": loss_scl
        }

    # ==========================
    # PGA Helper Methods
    # ==========================
    def _compute_pga_loss(self,
                          s_qry: torch.Tensor, s_pos: torch.Tensor,
                          t_qry: torch.Tensor, t_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the geometric alignment loss via Spectral CKA.
        """
        s_batch = torch.cat([s_qry, s_pos], dim=0)
        t_batch = torch.cat([t_qry, t_pos], dim=0)

        K_T = torch.matmul(t_batch, t_batch.t())
        K_S = torch.matmul(s_batch, s_batch.t())

        K_T_Clean = self._spectral_filter(K_T, self.variance_threshold)

        return 1.0 - self._centered_kernel_alignment(K_S, K_T_Clean)

    def _spectral_filter(self, gram_matrix: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Performs Eigendecomposition and reconstructs matrix using only principal components.
        """
        orig_dtype = gram_matrix.dtype
        gram_matrix_fp32 = gram_matrix.to(torch.float32)

        L, V = torch.linalg.eigh(gram_matrix_fp32)
        L = L.flip(0)
        V = V.flip(1)
        L = torch.clamp(L, min=0.0)

        total_variance = torch.sum(L)
        if total_variance <= 1e-8:
            return gram_matrix

        cumulative_variance = torch.cumsum(L, dim=0) / total_variance
        k = torch.searchsorted(cumulative_variance, threshold) + 1
        k = min(k.item(), len(L))

        L_k = torch.diag(L[:k])
        V_k = V[:, :k]
        K_reconstructed = V_k @ L_k @ V_k.t()
        
        return K_reconstructed.to(orig_dtype)

    def _centered_kernel_alignment(self, K_S: torch.Tensor, K_T: torch.Tensor) -> torch.Tensor:
        """Computes Linear CKA (Centered Kernel Alignment)."""
        orig_dtype = K_S.dtype
        K_S = K_S.to(torch.float32)
        K_T = K_T.to(torch.float32)

        n = K_S.size(0)
        unit = torch.ones([n, n], device=K_S.device)
        I = torch.eye(n, device=K_S.device)
        H = I - unit / n

        K_S_c = torch.matmul(torch.matmul(H, K_S), H)
        K_T_c = torch.matmul(torch.matmul(H, K_T), H)

        hsic = torch.trace(torch.matmul(K_S_c, K_T_c))
        var_s = torch.sqrt(torch.trace(torch.matmul(K_S_c, K_S_c)))
        var_t = torch.sqrt(torch.trace(torch.matmul(K_T_c, K_T_c)))

        cka_score = hsic / (var_s * var_t + 1e-8)
        return cka_score.to(orig_dtype)

    # ==========================
    # SCL Helper Methods
    # ==========================
    def _compute_scl_loss(self, 
                          s_hidden: Any, t_hidden: Any, 
                          s_ids: torch.Tensor, t_ids: torch.Tensor,
                          processor: Optional[ProcessorMixin],
                          temperature: float) -> torch.Tensor:
        """
        Computes Semantic Consistency Learning loss via MI maximization on 4 pathways.
        Uses specialized projectors for cross-modal alignment.
        """
        if processor is None:
            return torch.tensor(0.0, device=s_ids.device)

        # 1. Extract modality-specific representations (Raw)
        s_img, s_txt = self._get_image_text_representation(s_hidden, processor, s_ids)
        t_img_raw, t_txt_raw = self._get_image_text_representation(t_hidden, processor, t_ids)

        # 2. Project Teacher features to Student space using specific projectors
        # Returns: (Intra-Img, Intra-Txt, Cross-Img, Cross-Txt)
        t_img_intra, t_txt_intra, t_img_cross, t_txt_cross = self.scl_projectors(t_img_raw, t_txt_raw)

        # 3. Compute MI (via InfoNCE) for all 4 pathways
        
        # A. Intra-modal: Align Student Img with Teacher Img (Intra Projector)
        loss_img_img = self._info_nce(s_img, t_img_intra, temperature)
        
        # B. Intra-modal: Align Student Txt with Teacher Txt (Intra Projector)
        loss_txt_txt = self._info_nce(s_txt, t_txt_intra, temperature)
        
        # C. Inter-modal: Align Student Img with Teacher Txt (Cross Projector)
        # We transform Teacher Txt using cross_txt_proj to match Student Img space
        loss_img_txt = self._info_nce(s_img, t_txt_cross, temperature)
        
        # D. Inter-modal: Align Student Txt with Teacher Img (Cross Projector)
        # We transform Teacher Img using cross_img_proj to match Student Txt space
        loss_txt_img = self._info_nce(s_txt, t_img_cross, temperature)

        return (loss_img_img + loss_txt_txt + loss_img_txt + loss_txt_img) / 4.0

    def _info_nce(self, s_features: torch.Tensor, t_features: torch.Tensor, temp: float) -> torch.Tensor:
        """Computes InfoNCE loss between two sets of features."""
        s_norm = F.normalize(s_features, dim=-1)
        t_norm = F.normalize(t_features, dim=-1)
        
        if self.world_size > 1:
            s_norm = self._dist_gather_tensor(s_norm)
            t_norm = self._dist_gather_tensor(t_norm)

        logits = torch.matmul(s_norm, t_norm.T) / temp
        labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        
        return nn.CrossEntropyLoss()(logits, labels)

    def _get_image_text_representation(self, output_hidden_states: torch.Tensor, processor: ProcessorMixin, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts the image and text representations from the hidden states.
        """
        import pdb; pdb.set_trace()

        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        pad_token_id = processor.tokenizer.pad_token_id
        # Create mask for image and text tokens
        image_mask = (input_ids == image_token_id).float()
        text_mask = (input_ids != image_token_id).float()
        text_mask = (input_ids != image_token_id and input_ids != pad_token_id).float()


        # Get attention scores of last token in last layer
        last_attention = output_hidden_states.attentions[-1] # (batch_size, num_heads, seq_len, seq_len)
        last_token_attention = last_attention[:, :, -1, :] # (batch_size, num_heads, seq_len)
        attention_weights = last_token_attention.mean(dim=1) # mean over heads, shape: (batch_size, seq_len)

        # Get last hidden state
        last_hidden_state = output_hidden_states.hidden_states[-1] # (batch, seq_len, hidden_dim)

        def get_weighted_rep(mask):
            masked_attn = attention_weights.masked_fill(~mask, float('-inf'))
            normalized_weights = torch.softmax(masked_attn, dim=-1) # (batch, seq_len)
            
            weighted_rep = torch.bmm(normalized_weights.unsqueeze(1), last_hidden_state)
            return weighted_rep.squeeze(1) # (batch, hidden_dim)

        image_representation = get_weighted_rep(image_mask)
        text_representation = get_weighted_rep(text_mask)

        if image_mask.sum() == 0:
            # Use text representation as image representation
            image_representation = text_representation

        return image_representation, text_representation
        
        
        