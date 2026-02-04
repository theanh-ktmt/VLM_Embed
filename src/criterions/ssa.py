"""
This module defines the SSA (Spectral-Structural Alignment) criterion for Cross-Architecture VLM Distillation,
enhanced with MSE-based sequence alignment and distributed contrastive learning support.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedModel
from transformers import ProcessorMixin
from ..arguments import SSAAguments

logger = logging.getLogger(__name__)


class SSALoss(nn.Module):
    """
    Implements a unified distillation loss combining:
    1.  **SSA (Spectral-Structural Alignment):** Geometric alignment (Spectral CKA + Modality Gap).
    2.  **MSE (Mean Squared Error):** Direct feature alignment via projection.
    3.  **Contrastive Loss (InfoNCE):** Distributed-aware representation learning.
    """

    def __init__(self, args: Type[SSAAguments]):
        """
        Args:
            args: Configuration arguments containing:
                - kd_weight: Weight for the SSA distillation loss.
                - mse_weight: Weight for the MSE distillation loss.
                - spectral_variance_threshold: Ratio of variance to keep in Teacher's spectrum.
                - modality_gap_weight: Weight for the Modality Gap Consistency loss.
                - matryoshka_dims: List of dimensions for Matryoshka learning.
        """
        super(SSALoss, self).__init__()
        self.args = args
        
        # Loss Weights
        self.ssa_weight = args.kd_weight
        self.mse_weight = getattr(args, 'mse_loss_weight', 0.3) # Default from mse.py
        
        # SSA Hyperparameters
        self.variance_threshold = getattr(args, 'spectral_variance_threshold', 0.95)
        self.gap_weight = getattr(args, 'modality_gap_weight', 1.0)
        self.matryoshka_dims = getattr(args, 'ssa_matryoshka_dims', [64])

        # Loss Functions
        self.mse_loss_fn = nn.MSELoss(reduction='mean')

        # Distributed Setup
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.process_rank = dist.get_rank()
        else:
            self.world_size = 1
            self.process_rank = 0

        logger.info(f"SSALoss initialized | SSA Weight: {self.ssa_weight} | MSE Weight: {self.mse_weight}")
        logger.info(f"SSA Params | Threshold: {self.variance_threshold} | Gap Weight: {self.gap_weight} | Matryoshka: {self.matryoshka_dims}")

    def _dist_gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """
        Gathers tensors from all GPUs to create a global batch for contrastive learning.
        """
        if self.world_size <= 1:
            return t
            
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute Combined SSA + MSE + Contrastive loss.
        """
        student_model: PreTrainedModel = distiller.student
        teacher_model: PreTrainedModel = distiller.teacher

        # --- 1. Encode Inputs ---
        with torch.no_grad():
            teacher_model.eval()
            t_qry, _, _, t_qry_output_hidden_states = teacher_model.encode_input(input_data['teacher_inputs']['qry'])
            t_pos, _, _, t_pos_output_hidden_states = teacher_model.encode_input(input_data['teacher_inputs']['pos'])

        s_qry, _, _, s_qry_output_hidden_states = student_model.encode_input(input_data['student_inputs']['qry'])
        s_pos, _, _, s_pos_output_hidden_states = student_model.encode_input(input_data['student_inputs']['pos'])

        # --- 2. Compute Distributed Contrastive Loss (InfoNCE) ---
        # Gather embeddings from all GPUs to maximize batch size for negative samples
        if self.world_size > 1:
            all_s_qry = self._dist_gather_tensor(s_qry)
            all_s_pos = self._dist_gather_tensor(s_pos)
        else:
            all_s_qry = s_qry
            all_s_pos = s_pos

        scores = student_model.compute_similarity(all_s_qry, all_s_pos)
        scores = scores.view(all_s_qry.size(0), -1)
        
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        # Handle multiple positives if necessary (broadcasting target indices)
        target = target * (all_s_pos.size(0) // all_s_qry.size(0))
        
        contrastive_loss = nn.CrossEntropyLoss()(scores / distiller.temperature, target)

        # --- 3. Compute SSA Distillation Loss (Geometric Alignment) ---
        # Note: SSA is computed on the LOCAL batch to reduce communication overhead.
        ssa_loss_full = self._compute_ssa_components(s_qry, s_pos, t_qry, t_pos)

        # Matryoshka Slicing for SSA
        ssa_loss_matryoshka = 0.0
        if self.matryoshka_dims:
            for dim in self.matryoshka_dims:
                if dim < s_qry.shape[-1]:
                    ssa_loss_matryoshka += self._compute_ssa_components(
                        s_qry[:, :dim], s_pos[:, :dim],
                        t_qry, t_pos
                    )

        ssa_loss_total = ssa_loss_full
        if self.matryoshka_dims:
            ssa_loss_total = (ssa_loss_total + ssa_loss_matryoshka) / (1 + len(self.matryoshka_dims))

        # --- 4. Compute MSE Distillation Loss (Direct Feature Alignment) ---
        # Project teacher embeddings to student dimension
        # We check if the specific projector exists in the distiller
        mse_loss_seq = torch.tensor(0.0, device=s_qry.device)
        
        if hasattr(distiller, "projectors") and "t2s_txt" in distiller.projectors:
            # MSE.py logic: Project teacher query reps and compare with student query reps
            projected_t_qry = distiller.projectors["t2s_txt"](t_qry)
            
            # You can also opt to project positive reps if desired, but here we mirror MSE.py
            # projected_t_pos = distiller.projectors["t2s_txt"](t_pos) 
            
            mse_loss_seq = self.mse_loss_fn(s_qry, projected_t_qry)

        # --- 5. Combine All Losses ---
        # Total = Contrastive + (w_1 * SSA) + (w_2 * MSE)
        total_distillation = (self.ssa_weight * ssa_loss_total) + (self.mse_weight * mse_loss_seq)
        total_loss = contrastive_loss + total_distillation

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "ssa_loss": ssa_loss_total,
            "mse_loss": mse_loss_seq,
            "kd_loss": total_distillation # Combined KD loss for logging
        }

    def _compute_ssa_components(self,
                                s_qry: torch.Tensor, s_pos: torch.Tensor,
                                t_qry: torch.Tensor, t_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined Spectral CKA and Modality Gap loss for a specific embedding dimension.
        """
        # Concatenate Query and Positive to capture the full batch geometry
        s_batch = torch.cat([s_qry, s_pos], dim=0)
        t_batch = torch.cat([t_qry, t_pos], dim=0)

        # A. Spectral CKA (Macro-Alignment)
        K_T = torch.matmul(t_batch, t_batch.t())
        K_T_Clean = self._spectral_filter(K_T, self.variance_threshold)
        K_S = torch.matmul(s_batch, s_batch.t())
        scka_loss = 1.0 - self._centered_kernel_alignment(K_S, K_T_Clean)

        # B. Modality Gap Consistency (Micro-Alignment)
        gap_loss = self._compute_gap_loss(s_qry, s_pos, t_qry, t_pos)

        return scka_loss + (self.gap_weight * gap_loss)

    def _spectral_filter(self, gram_matrix: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Performs Eigendecomposition and reconstructs using principal components (Float32 precision).
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
        """
        Computes Linear CKA given two Gram matrices (Float32 precision).
        """
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

    def _compute_gap_loss(self,
                          s_qry: torch.Tensor, s_pos: torch.Tensor,
                          t_qry: torch.Tensor, t_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the Modality Gap Consistency loss.
        """
        dist_s = torch.norm(s_qry - s_pos, p=2, dim=1)
        dist_t = torch.norm(t_qry - t_pos, p=2, dim=1)

        dist_s_norm = dist_s / (dist_s.mean() + 1e-8)
        dist_t_norm = dist_t / (dist_t.mean() + 1e-8)

        return F.mse_loss(dist_s_norm, dist_t_norm)

    def _get_image_text_representation(self, output_hidden_states: torch.Tensor, processor: ProcessorMixin, input_ids: torch.Tensor) -> torch.Tensor:
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
        
        
        