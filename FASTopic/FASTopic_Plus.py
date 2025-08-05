import torch
import torch.nn.functional as F
import numpy as np
from .FASTopic import FASTopic
from .TP import TP
from torch import nn
import torch_kmeans
import logging
import sentence_transformers
import hdbscan
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from utils import static_utils
from .contrastive_losses import info_nce_loss, circle_loss, triplet_loss

class FASTopic_Plus(FASTopic):
    def __init__(self, vocab_size, embed_size, num_topics,
                 contextual_embed_size=None, vocab=None, DT_alpha=3.0, TW_alpha=2.0, theta_temp=1.0,
                 weight_loss_CLT=1.0, weight_loss_CLC=1.0, weight_loss_DTC=1.0, weight_loss_TP=1.0,
                 threshold_epoch=10, threshold_cluster=30, metric_CL='cosine', max_clusters=9,
                 sinkhorn_alpha=20.0, OT_max_iter=1000, dropout=0.,
                 positive_sampling='random', negative_sampling='random', contrastive_loss_type='triplet', **kwargs):

        super().__init__(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_topics=num_topics,
            DT_alpha=DT_alpha,
            TW_alpha=TW_alpha,
            theta_temp=theta_temp,
            sinkhorn_alpha=sinkhorn_alpha,
            sinkhorn_max_iter=OT_max_iter,
            **kwargs
        )

        self.weight_loss_CLT = weight_loss_CLT
        self.weight_loss_CLC = weight_loss_CLC
        self.weight_loss_DTC = weight_loss_DTC
        self.weight_loss_TP = weight_loss_TP
        self.threshold_epoch = threshold_epoch
        self.threshold_cluster = threshold_cluster
        self.metric_CL = metric_CL
        self.positive_sampling = positive_sampling
        self.negative_sampling = negative_sampling
        self.contrastive_loss_type = contrastive_loss_type
        self.max_clusters = max_clusters
        self.vocab = vocab
        self.group_topic = None
        self.topics = []
        self.topic_index_mapping = {}
        
        input_dim = contextual_embed_size if contextual_embed_size is not None else embed_size
        self.document_emb_prj = nn.Sequential(
            nn.Linear(input_dim, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.topic_embeddings.device)
        self.TP = TP(self.weight_loss_TP, sinkhorn_alpha, OT_max_iter)

    def create_group_topic(self):
        with torch.no_grad():
            distances = torch.cdist(self.topic_embeddings, self.topic_embeddings, p=2).cpu().numpy()
        Z = linkage(distances, method='average', optimal_ordering=True)
        group_id = fcluster(Z, t=self.max_clusters, criterion='maxclust') - 1
        self.group_topic = [[] for _ in range(self.max_clusters)]
        for i in range(self.num_topics):
            self.group_topic[group_id[i]].append(i)
        word_topic_assignments = self.get_word_topic_assignments()
        self.topics = [word_topic_assignments[topic_idx] for topic_idx in range(self.num_topics)]
        self.topic_index_mapping = {topic_idx: topic_idx for topic_idx in range(self.num_topics)}

    def get_word_topic_assignments(self):
        word_topic_assignments = [[] for _ in range(self.num_topics)]
        if self.vocab is not None:
            for word_idx, word in enumerate(self.vocab):
                topic_idx = self.word_to_topic_by_similarity(word)
                word_topic_assignments[topic_idx].append(word_idx)
        else:
            for word_idx in range(self.word_embeddings.shape[0]):
                topic_idx = self.word_to_topic_by_similarity(word_idx)
                word_topic_assignments[topic_idx].append(word_idx)
        return word_topic_assignments

    def word_to_topic_by_similarity(self, word):
        word_idx = self.vocab.index(word) if self.vocab is not None and isinstance(word, str) else word
        word_embedding = self.word_embeddings[word_idx].unsqueeze(0)
        similarity_scores = F.cosine_similarity(word_embedding, self.topic_embeddings)
        return torch.argmax(similarity_scores).item()

    def get_loss_CLC(self, margin=0.2, num_negatives=10, gamma=32):
        loss_CLC = 0.0
        for group_idx, group_topics in enumerate(self.group_topic):
            if len(group_topics) < 2:
                continue
            anchor = torch.mean(self.topic_embeddings[group_topics], dim=0, keepdim=True)
            pos_embs = self.topic_embeddings[group_topics]
            pos_distances = F.pairwise_distance(anchor, pos_embs)
            
            if self.positive_sampling == 'random':
                positive_topic_idx = np.random.choice(group_topics)
            elif self.positive_sampling == 'hard':
                median_dist_idx = torch.argsort(pos_distances)[len(pos_distances) // 2]
                positive_topic_idx = group_topics[median_dist_idx]
            else:
                raise ValueError(f"Invalid positive_sampling strategy: {self.positive_sampling}")
            positive = self.topic_embeddings[positive_topic_idx].unsqueeze(0)
            
            negative_candidates = [t for i, g in enumerate(self.group_topic) if i != group_idx for t in g]
            if len(negative_candidates) < num_negatives:
                continue
            neg_embs = self.topic_embeddings[negative_candidates]
            neg_distances = F.pairwise_distance(anchor.repeat(len(negative_candidates), 1), neg_embs)
            if self.negative_sampling == 'random':
                negative_topic_idxes = np.random.choice(negative_candidates, size=num_negatives, replace=False)
            elif self.negative_sampling == 'hard':
                hard_negative_indices = torch.argsort(neg_distances)[:num_negatives]
                negative_topic_idxes = [negative_candidates[i] for i in hard_negative_indices]
            else:
                raise ValueError(f"Invalid negative_sampling strategy: {self.negative_sampling}")
            
            negatives = self.topic_embeddings[negative_topic_idxes]
            if self.contrastive_loss_type == 'infonce':
                loss = info_nce_loss(anchor, positive, negatives)
            elif self.contrastive_loss_type == 'circle':
                loss = circle_loss(anchor, positive, negatives, margin=margin, gamma=gamma)
            elif self.contrastive_loss_type == 'triplet':
                loss = triplet_loss(anchor, positive, negatives, margin=margin, metric=self.metric_CL)
            else:
                raise ValueError(f"Unknown contrastive_loss_type: {self.contrastive_loss_type}")
            loss_CLC += loss.mean()
        return loss_CLC * self.weight_loss_CLC


    def get_loss_CLT(self, margin=0.2, num_negatives=10, gamma=32):
        loss_CLT = 0.0
        for group_idx, group_topics in enumerate(self.group_topic):
            for anchor_topic_idx in group_topics:
                anchor_words_idxes = self.topics[self.topic_index_mapping[anchor_topic_idx]]
                if len(anchor_words_idxes) < 2:
                    continue
                anchor = torch.mean(self.word_embeddings[anchor_words_idxes], dim=0, keepdim=True)
                pos_embs = self.word_embeddings[anchor_words_idxes]
                pos_distances = F.pairwise_distance(anchor, pos_embs)
                
                if self.positive_sampling == 'random':
                    positive_word_idx = np.random.choice(anchor_words_idxes)
                elif self.positive_sampling == 'hard':
                    median_dist_idx = torch.argsort(pos_distances)[len(pos_distances) // 2]
                    positive_word_idx = anchor_words_idxes[median_dist_idx]
                else:
                    raise ValueError(f"Invalid positive_sampling strategy: {self.positive_sampling}")
                positive = self.word_embeddings[positive_word_idx].unsqueeze(0)
                
                negative_candidates = []
                for neg_topic_idx in range(self.num_topics):
                    if neg_topic_idx not in group_topics:
                        negative_candidates.extend(self.topics[self.topic_index_mapping[neg_topic_idx]])
                if len(negative_candidates) < num_negatives:
                    continue
                neg_embs = self.word_embeddings[negative_candidates]
                neg_distances = F.pairwise_distance(anchor.repeat(len(negative_candidates), 1), neg_embs)
                if self.negative_sampling == 'random':
                    negative_word_idxes = np.random.choice(negative_candidates, size=num_negatives, replace=False)
                elif self.negative_sampling == 'hard':
                    hard_negative_indices = torch.argsort(neg_distances)[:num_negatives]
                    negative_word_idxes = [negative_candidates[i] for i in hard_negative_indices]
                else:
                    raise ValueError(f"Invalid negative_sampling strategy: {self.negative_sampling}")
                
                negatives = self.word_embeddings[negative_word_idxes]
                if self.contrastive_loss_type == 'infonce':
                    loss = info_nce_loss(anchor, positive, negatives)
                elif self.contrastive_loss_type == 'circle':
                    loss = circle_loss(anchor, positive, negatives, margin=margin, gamma=gamma)
                elif self.contrastive_loss_type == 'triplet':
                    loss = triplet_loss(anchor, positive, negatives, margin=margin, metric=self.metric_CL)
                else:
                    raise ValueError(f"Unknown contrastive_loss_type: {self.contrastive_loss_type}")
                loss_CLT += loss.mean()

        return loss_CLT * self.weight_loss_CLT


    def get_loss_DTC(self, contextual_emb, theta, margin=0.2, k=5, gamma=32):
        anchor_doc_emb = contextual_emb
        _, top_positive_indices = torch.topk(theta, k=1, dim=1)
        pos_emb = self.topic_embeddings[top_positive_indices].squeeze(1)

        if self.negative_sampling == 'random':
            _, bottom_negative_indices = torch.topk(theta, k=k, dim=1, largest=False)
            neg_emb = self.topic_embeddings[bottom_negative_indices].mean(dim=1)
        elif self.negative_sampling == 'hard':
            all_topic_distances = F.pairwise_distance(anchor_doc_emb.unsqueeze(1), self.topic_embeddings.unsqueeze(0))
            all_topic_distances.scatter_(1, top_positive_indices, float('inf'))
            _, hard_negative_indices = torch.topk(all_topic_distances, k=k, dim=1, largest=False)
            neg_emb = self.topic_embeddings[hard_negative_indices].mean(dim=1)
        else:
            raise ValueError(f"Invalid negative_sampling strategy: {self.negative_sampling}")

        if self.contrastive_loss_type == 'infonce':
            loss = 0.0
            for i in range(anchor_doc_emb.size(0)):
                loss += info_nce_loss(anchor_doc_emb[i:i+1], pos_emb[i:i+1], neg_emb[i:i+1])
            return (loss / anchor_doc_emb.size(0)) * self.weight_loss_DTC
        
        elif self.contrastive_loss_type == 'circle':
            loss = 0.0
            for i in range(anchor_doc_emb.size(0)):
                loss += circle_loss(anchor_doc_emb[i:i+1], pos_emb[i:i+1], neg_emb[i:i+1], margin=margin, gamma=gamma)
            return (loss / anchor_doc_emb.size(0)) * self.weight_loss_DTC
        
        elif self.contrastive_loss_type == 'triplet':
            loss = 0.0
            for i in range(anchor_doc_emb.size(0)):
                loss += triplet_loss(anchor_doc_emb[i:i+1], pos_emb[i:i+1], neg_emb[i:i+1], margin=margin, metric=self.metric_CL)
            return (loss / anchor_doc_emb.size(0)) * self.weight_loss_DTC
        
        else:
            raise ValueError(f"Unknown contrastive_loss_type: {self.contrastive_loss_type}")

    def get_loss_TP(self, doc_embeddings, indices):
        minibatch_embeddings = doc_embeddings
        cost = self.pairwise_euclidean_distance(minibatch_embeddings, minibatch_embeddings) \
            + 1e1 * torch.ones(minibatch_embeddings.size(0), minibatch_embeddings.size(0)).to(minibatch_embeddings.device)
        matrixP = self.create_matrixP(minibatch_embeddings, indices)
        loss_TP = self.TP(cost, matrixP)
        return loss_TP

    def create_matrixP(self, minibatch_embeddings, indices):
        num_minibatch = len(indices)
        matrixP = torch.ones((num_minibatch, num_minibatch), device=self.topic_embeddings.device) / num_minibatch
        norm_embeddings = F.normalize(minibatch_embeddings, p=2, dim=1).clamp(min=1e-6)
        matrixP = torch.matmul(norm_embeddings, norm_embeddings.T)
        matrixP = matrixP.clamp(min=1e-4)
        return matrixP

    def compute_theta_entropy(self, theta):
        eps = 1e-8
        entropy = -torch.sum(theta * torch.log(theta + eps), dim=1)
        return entropy.mean().item()

    def forward(self, indices, input, epoch_id=None, doc_embeddings=None):
        train_bow = input[0].to(self.topic_embeddings.device)
        doc_emb = doc_embeddings.to(self.topic_embeddings.device)
        doc_emb_proj = self.document_emb_prj(doc_emb)

        loss_DT, transp_DT = self.DT_ETP(doc_emb_proj, self.topic_embeddings)
        loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
        loss_ETP = loss_DT + loss_TW

        theta = transp_DT * transp_DT.shape[0]
        beta = transp_TW * transp_TW.shape[0]
        recon = torch.matmul(theta, beta)
        loss_DSR = -(train_bow * (recon + 1e-12).log()).sum(axis=1).mean()

        loss_TP = self.get_loss_TP(doc_emb_proj, indices)
        loss_CLC = 0.0
        loss_CLT = 0.0
        loss_DTC = 0.0

        if epoch_id is not None and epoch_id >= self.threshold_epoch and (
            epoch_id == self.threshold_epoch or 
            (epoch_id > self.threshold_epoch and epoch_id % self.threshold_cluster == 0)
        ):
            self.create_group_topic()

        if epoch_id is not None and epoch_id >= self.threshold_epoch and self.group_topic is not None:
            if self.weight_loss_CLC != 0:
                loss_CLC = self.get_loss_CLC()
            if self.weight_loss_CLT != 0:
                loss_CLT = self.get_loss_CLT()
            if self.weight_loss_DTC != 0:
                loss_DTC = self.get_loss_DTC(doc_emb_proj, theta)

        loss = loss_DSR + loss_ETP + loss_TP + loss_CLC + loss_CLT + loss_DTC

        rst_dict = {
            'loss': loss,
            'loss_DSR': loss_DSR,
            'loss_ETP': loss_ETP,
            'loss_TP': loss_TP,
            'loss_CLC': loss_CLC,
            'loss_CLT': loss_CLT,
            'loss_DTC': loss_DTC
        }
        return rst_dict

    def pairwise_euclidean_distance(self, x, y):
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost