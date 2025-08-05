import torch
import torch.nn.functional as F

def info_nce_loss(anchor, positive, negatives, temperature=0.07):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    pos_sim = torch.matmul(anchor, positive.T) / temperature  # [1,1]
    neg_sim = torch.matmul(anchor, negatives.T) / temperature  # [1,N]
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # [1, 1+N]
    labels = torch.zeros(1, dtype=torch.long, device=anchor.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def circle_loss(anchor, positive, negatives, margin=0.25, gamma=32):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Không batch: anchor [1, D], positive [1, D], negatives [N, D]
    s_p = (anchor * positive).sum(dim=-1)  # [1]
    s_n = torch.matmul(negatives, anchor.T).squeeze(-1)  # [N]
    alpha_p = torch.clamp_min(margin - s_p, min=0.)  # [1]
    alpha_n = torch.clamp_min(s_n + margin, min=0.)  # [N]
    delta_p = 1 - margin
    delta_n = margin
    pos_term = torch.exp(-gamma * alpha_p * (s_p - delta_p))  # [1]
    neg_term = torch.exp(gamma * alpha_n * (s_n - delta_n))   # [N]
    loss = torch.log1p(pos_term * neg_term.sum()).mean()
    return loss

def triplet_loss(anchor, positive, negatives, margin=0.2, metric='cosine'):
    # Non-batched version: anchor [1, D], positive [1, D], negatives [N, D]
    if metric == 'cosine':
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor.repeat(negatives.size(0), 1), negatives)
        pos_dist = 1 - pos_sim
        neg_dist = 1 - neg_sim
    else:
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor.repeat(negatives.size(0), 1), negatives)
    loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0).mean()
    return loss
