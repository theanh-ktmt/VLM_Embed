import torch
import torch.nn as nn 


class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super(ContrastiveLoss, self).__init__()
        self.args = args
        self.kd_loss_weight = self.args.kd_weight
        self.distance_weight = self.args.rkd_distance_weight
        self.angle_weight = self.args.rkd_angle_weight
        
    def forward(self, distiller, input_data):
        self.distiller = distiller
        student_model = distiller.student
        
        student_input_qry = input_data['student_inputs']['qry']
        student_input_pos = input_data['student_inputs']['pos']
        
        student_qry_reps, _, _, output_hidden_states = student_model.encode_input(student_input_qry)
        student_pos_reps, _, _, output_hidden_states = student_model.encode_input(student_input_pos)

        scores = student_model.compute_similarity(student_qry_reps, student_pos_reps)
        scores = scores.view(student_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (student_qry_reps.size(0) // student_pos_reps.size(0))
        contrastive_loss = nn.CrossEntropyLoss()(scores / self.distiller.temperature, target)

        total_loss = contrastive_loss
        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss,
        }
        