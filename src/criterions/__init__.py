from .contrastive_loss_with_RKD import ContrastiveLossWithRKD
from .proposal_loss_with_DTW import ProposalLossWithDTW
from .universal_logit_distillation import UniversalLogitDistillation

criterion_list = {
    "contrastive_rkd": ContrastiveLossWithRKD,
    "proposal_dtw": ProposalLossWithDTW,
    "universal_logit": UniversalLogitDistillation,
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_list:
        raise ValueError(f"Criterion {args.criterion} not found.")
    return criterion_list[args.criterion](args)