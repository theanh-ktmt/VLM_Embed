"""
This module defines the SSA (Spectral-Structural Alignment) criterion for Cross-Architecture VLM Distillation.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from ..arguments import SSAAguments

logger = logging.getLogger(__name__)


class SSALoss(nn.Module):
    """
    Implements Spectral-Structural Alignment (SSA) Loss.

    SSA is designed for cross-architecture distillation (e.g., Large VLM Teacher -> Small Student)
    where dimension sizes and patch counts differ.

    The total loss is composed of:
    1.  **Contrastive Loss (InfoNCE):** Standard representation learning objective.
    2.  **Spectral CKA (Macro-Alignment):** Aligns the geometry of the batch. Unlike standard CKA,
        this filters the Teacher's Gram matrix to only retain principal components (spectral filtering),
        preventing the student from trying to learn the teacher's noise or excessive complexity.
    3.  **Modality Gap Consistency (Micro-Alignment):** Ensures the geometric distance between
        modalities (Query vs Positive) in the Student matches the Teacher, resolving mismatches
        caused by different token/patch counts.
    4.  **Matryoshka Training:** Applies the above objectives to sliced embeddings (e.g., first 64 dims)
        to enable adaptive retrieval efficiency.
    """

    def __init__(self, args: Type[SSAAguments]):
        """
        Args:
            args: Configuration arguments containing:
                - kd_weight: Weight for the total distillation loss.
                - spectral_variance_threshold: Ratio of variance to keep in Teacher's spectrum (e.g., 0.95).
                - modality_gap_weight: Weight for the Modality Gap Consistency loss.
                - matryoshka_dims: List of dimensions for Matryoshka learning (e.g., [64]).
        """
        super(SSALoss, self).__init__()
        self.args = args
        self.kd_weight = args.kd_weight
        self.variance_threshold = getattr(args, 'spectral_variance_threshold', 0.95)
        self.gap_weight = getattr(args, 'modality_gap_weight', 1.0)
        self.matryoshka_dims = getattr(args, 'ssa_matryoshka_dims', [64])

        logger.info(f"SSA Loss initialized with Threshold={self.variance_threshold}, "
                    f"Gap Weight={self.gap_weight}, Matryoshka Dims={self.matryoshka_dims}")

    def forward(self, distiller: nn.Module, input_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute the SSA loss.
        """
        student_model: PreTrainedModel = distiller.student
        teacher_model: PreTrainedModel = distiller.teacher

        # --- 1. Encode Inputs ---
        with torch.no_grad():
            teacher_model.eval()
            t_qry, _, _, _ = teacher_model.encode_input(input_data['teacher_inputs']['qry'])
            t_pos, _, _, _ = teacher_model.encode_input(input_data['teacher_inputs']['pos'])

        s_qry, _, _, _ = student_model.encode_input(input_data['student_inputs']['qry'])
        s_pos, _, _, _ = student_model.encode_input(input_data['student_inputs']['pos'])

        # --- 2. Compute Contrastive Loss (InfoNCE) ---
        # This ensures the student learns basic discrimination
        scores = student_model.compute_similarity(s_qry, s_pos)
        scores = scores.view(s_qry.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        # Handle multiple positives if necessary
        target = target * (s_pos.size(0) // s_qry.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / distiller.temperature, target)

        # --- 3. Compute SSA Distillation Loss (Full Dimension) ---
        ssa_loss_full = self._compute_ssa_components(s_qry, s_pos, t_qry, t_pos)

        # --- 4. Compute SSA Distillation Loss (Matryoshka Slices) ---
        ssa_loss_matryoshka = 0.0
        if self.matryoshka_dims:
            for dim in self.matryoshka_dims:
                # Slice embeddings
                if dim < s_qry.shape[-1]:
                    ssa_loss_matryoshka += self._compute_ssa_components(
                        s_qry[:, :dim], s_pos[:, :dim],
                        t_qry, t_pos  # We usually compare sliced student against full teacher topology
                    )

        # Combine losses
        # If using Matryoshka, average the nested losses
        total_distillation = ssa_loss_full
        if self.matryoshka_dims:
            total_distillation = (total_distillation + ssa_loss_matryoshka) / (1 + len(self.matryoshka_dims))

        total_loss = contrastive_loss + (self.kd_weight * total_distillation)

        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "kd_loss": total_distillation
        }

    def _compute_ssa_components(self,
                                s_qry: torch.Tensor, s_pos: torch.Tensor,
                                t_qry: torch.Tensor, t_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the combined Spectral CKA and Modality Gap loss for a specific embedding dimension.
        """
        # Concatenate Query and Positive to capture the full batch geometry
        # Shape: [2*B, dim]
        s_batch = torch.cat([s_qry, s_pos], dim=0)
        t_batch = torch.cat([t_qry, t_pos], dim=0)

        # A. Spectral CKA (Macro-Alignment)
        # 1. Compute Teacher Gram Matrix
        K_T = torch.matmul(t_batch, t_batch.t())

        # 2. Spectral Filtering: Reconstruct K_T using only top eigenvectors
        K_T_Clean = self._spectral_filter(K_T, self.variance_threshold)

        # 3. Compute Student Gram Matrix
        K_S = torch.matmul(s_batch, s_batch.t())

        # 4. Compute CKA Loss (1 - CKA)
        scka_loss = 1.0 - self._centered_kernel_alignment(K_S, K_T_Clean)

        # B. Modality Gap Consistency (Micro-Alignment)
        # Calculates the magnitude of the semantic bridge between Query and Positive
        gap_loss = self._compute_gap_loss(s_qry, s_pos, t_qry, t_pos)

        return scka_loss + (self.gap_weight * gap_loss)

    def _spectral_filter(self, gram_matrix: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Performs Eigendecomposition on the Gram matrix and reconstructs it using
        principal components.
        
        [FIX]: Explicitly casts to float32 because torch.linalg.eigh is not 
        implemented for BFloat16/FP16 on CUDA.
        """
        # 1. Save original dtype (likely bfloat16) and upcast to float32
        orig_dtype = gram_matrix.dtype
        gram_matrix_fp32 = gram_matrix.to(torch.float32)

        # 2. Perform Eigendecomposition in FP32
        L, V = torch.linalg.eigh(gram_matrix_fp32)

        # Sort eigenvalues/vectors descending
        L = L.flip(0)
        V = V.flip(1)

        # Remove negative eigenvalues (numerical noise)
        L = torch.clamp(L, min=0.0)

        # Find k components for cumulative variance
        total_variance = torch.sum(L)
        if total_variance <= 1e-8:
            return gram_matrix  # Return original if empty/zero

        cumulative_variance = torch.cumsum(L, dim=0) / total_variance
        
        # Find index where variance exceeds threshold
        k = torch.searchsorted(cumulative_variance, threshold) + 1
        k = min(k.item(), len(L))

        # Reconstruct: V_k @ diag(L_k) @ V_k.T
        L_k = torch.diag(L[:k])
        V_k = V[:, :k]
        
        K_reconstructed = V_k @ L_k @ V_k.t()
        
        # 3. Cast back to original dtype to maintain compatibility with the rest of the graph
        return K_reconstructed.to(orig_dtype)

    def _centered_kernel_alignment(self, K_S: torch.Tensor, K_T: torch.Tensor) -> torch.Tensor:
        """
        Computes Linear CKA given two Gram matrices.
        
        [FIX]: Performs calculation in float32 to prevent underflow/overflow 
        typical in Gram matrix traces, then casts result back.
        """
        # Save original dtype
        orig_dtype = K_S.dtype
        
        # Upcast inputs to float32
        K_S = K_S.to(torch.float32)
        K_T = K_T.to(torch.float32)

        # Centering Matrix H = I - 1/n J
        n = K_S.size(0)
        unit = torch.ones([n, n], device=K_S.device)
        I = torch.eye(n, device=K_S.device)
        H = I - unit / n

        # Center the Gram matrices
        K_S_c = torch.matmul(torch.matmul(H, K_S), H)
        K_T_c = torch.matmul(torch.matmul(H, K_T), H)

        # Compute HSIC (Hilbert-Schmidt Independence Criterion)
        hsic = torch.trace(torch.matmul(K_S_c, K_T_c))

        # Normalization terms
        var_s = torch.sqrt(torch.trace(torch.matmul(K_S_c, K_S_c)))
        var_t = torch.sqrt(torch.trace(torch.matmul(K_T_c, K_T_c)))

        cka_score = hsic / (var_s * var_t + 1e-8)
        
        # Cast scalar back to original dtype (bf16)
        return cka_score.to(orig_dtype)

    def _compute_gap_loss(self,
                          s_qry: torch.Tensor, s_pos: torch.Tensor,
                          t_qry: torch.Tensor, t_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the Modality Gap Consistency loss.
        Calculates the Euclidean distance between Query and Positive for both Student and Teacher,
        then penalizes the difference via MSE.

        This teaches the student the 'semantic distance' between modalities, independent
        of embedding dimension or patch count.
        """
        # Euclidean distance between Qry and Pos
        # Shape: [B]
        dist_s = torch.norm(s_qry - s_pos, p=2, dim=1)
        dist_t = torch.norm(t_qry - t_pos, p=2, dim=1)

        # Normalize distances to be scale invariant relative to the model's latent space size
        # (Since dimension 4096 naturally yields larger L2 norms than dimension 768)
        dist_s_norm = dist_s / (dist_s.mean() + 1e-8)
        dist_t_norm = dist_t / (dist_t.mean() + 1e-8)

        return F.mse_loss(dist_s_norm, dist_t_norm)