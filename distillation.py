import torch
import torch.nn as nn
import torch.nn.functional as F



class PatientDistillation(nn.Module):
    def __init__(self, t_config, s_config):
        super(PatientDistillation, self).__init__()
        self.t_config = t_config
        self.s_config = s_config

    def forward(self, t_model, s_model,input_ids, token_type_ids, attention_mask, labels, k,args):
        with torch.no_grad():
            t_outputs = t_model(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=True)

        s_outputs = s_model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            output_hidden_states=True)

        t_logits, t_features = t_outputs[0], t_outputs[-1]
        train_loss, s_logits, s_features = s_outputs[0], s_outputs[1], s_outputs[-1]
        T = args.temperature
        soft_targets = F.softmax(t_logits / T, dim=-1)
        log_probs = F.log_softmax(s_logits / T, dim=-1)
        soft_loss = F.kl_div(log_probs, soft_targets.detach(), reduction='batchmean') * T * T
        t_features = t_features[-1]
        s_features = s_features[-1]
        dicti = {}
        labels_unique = torch.unique(labels)
        for label in labels_unique:
            dicti[str(label.item())] = (labels == label).nonzero(as_tuple=True)[0]
        distill_loss_curr = 0
        for i in range(s_features.size(0)):
            label = labels[i]
            indices = dicti[str(label.item())]
            elements_orig = torch.index_select(t_features, 0, indices)
            s_feature = torch.reshape(s_features[i], (1,-1))
            elements = torch.reshape(elements_orig, (elements_orig.size(0),-1))
            dist_sub = torch.sub(s_feature, elements)
            dist = torch.sum(dist_sub ** 2, 1)
            sorted_dist, indices_to_use = torch.sort(dist)
            if  sorted_dist.size(0)<k:
                k =  sorted_dist.size(0)
            indices_to_use = indices_to_use[:k]
            final_elements = torch.index_select(elements_orig, 0, indices_to_use)
            curr_Loss = F.mse_loss(final_elements, s_features[i].unsqueeze(0), reduction='none').mean(dim=(1, 2))
            curr_Loss = torch.sum(curr_Loss)
            distill_loss_curr = distill_loss_curr + curr_Loss
        distill_loss = distill_loss_curr
        return train_loss, soft_loss, distill_loss