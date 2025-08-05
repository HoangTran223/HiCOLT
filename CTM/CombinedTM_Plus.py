import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .CombinedTM import CombinedTM
from .TP import TP
from .contrastive_losses import info_nce_loss, circle_loss, triplet_loss
from scipy.cluster.hierarchy import linkage, fcluster

class CombinedTM_Plus(CombinedTM):
    def __init__(self, vocab=None, weight_loss_CLT=1.0, weight_loss_CLC=1.0, weight_loss_DTC=1.0, weight_loss_TP=1.0,
                 threshold_epoch=10, threshold_cluster=30, metric_CL='cosine', max_clusters=9,
                 alpha_TP=20.0, sinkhorn_max_iter=1000, positive_sampling='random', negative_sampling='random',
                 contrastive_loss_type='triplet', **kwargs):
        
        embed_size = 200
        super().__init__(**kwargs)
        self.weight_loss_CLT = weight_loss_CLT
        self.weight_loss_CLC = weight_loss_CLC
        self.weight_loss_DTC = weight_loss_DTC
        self.weight_loss_TP = weight_loss_TP
        self.alpha_TP = alpha_TP
        self.threshold_epoch = threshold_epoch
        self.threshold_cluster = threshold_cluster
        self.metric_CL = metric_CL
        self.max_clusters = max_clusters
        self.vocab = vocab
        self.group_topic = None
        self.topics = []
        self.topic_index_mapping = {}
        self.positive_sampling = positive_sampling
        self.negative_sampling = negative_sampling
        self.contrastive_loss_type = contrastive_loss_type

        self.topic_embeddings = nn.Parameter(F.normalize(torch.randn((self.num_topics, embed_size))))
        self.word_embeddings = nn.Parameter(F.normalize(torch.randn((self.vocab_size, embed_size))))

        self.document_emb_prj = nn.Sequential(
            nn.Linear(self.contextual_embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(kwargs.get('dropout', 0.))
        ).to(self.fcd1.weight.device)

        self.TP = TP(self.weight_loss_TP, self.alpha_TP, sinkhorn_max_iter)

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
        return word_topic_assignments

    def word_to_topic_by_similarity(self, word):
        word_idx = self.vocab.index(word) if isinstance(word, str) else word
        word_embedding = self.word_embeddings[word_idx].unsqueeze(0)
        similarity_scores = F.cosine_similarity(word_embedding, self.topic_embeddings)
        return torch.argmax(similarity_scores).item()

    def get_loss_CLC(self, margin=0.2, num_negatives=10, gamma=32):
        if self.group_topic is None:
            return 0.0
        loss_CLC = 0.0
        for group_idx, group_topics in enumerate(self.group_topic):
            if not group_topics:
                continue
            anchor = torch.mean(self.topic_embeddings[group_topics], dim=0, keepdim=True)
            # Positive sampling
            if self.positive_sampling == 'random':
                positive_topic_idx = np.random.choice(group_topics)
            elif self.positive_sampling == 'hard':
                pos_embs = self.topic_embeddings[group_topics]
                pos_distances = F.pairwise_distance(anchor, pos_embs)
                median_dist_idx = torch.argsort(pos_distances)[len(pos_distances) // 2]
                positive_topic_idx = group_topics[median_dist_idx]
            else:
                continue
            positive = self.topic_embeddings[positive_topic_idx].unsqueeze(0)

            # Negative sampling
            negative_candidates = [t for i, g in enumerate(self.group_topic) if i != group_idx for t in g]
            if len(negative_candidates) < num_negatives:
                continue
            if self.negative_sampling == 'random':
                negative_topic_idxes = np.random.choice(negative_candidates, size=num_negatives, replace=False)
            elif self.negative_sampling == 'hard':
                neg_embs = self.topic_embeddings[negative_candidates]
                neg_distances = F.pairwise_distance(anchor.repeat(len(negative_candidates), 1), neg_embs)
                hard_negative_indices = torch.argsort(neg_distances)[:num_negatives]
                negative_topic_idxes = [negative_candidates[i] for i in hard_negative_indices]
            else:
                continue
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
        if self.group_topic is None or not self.topics:
            return 0.0
        loss_CLT = 0.0
        for group_idx, group_topics in enumerate(self.group_topic):
            for anchor_topic_idx in group_topics:
                anchor_words_idxes = self.topics[self.topic_index_mapping[anchor_topic_idx]]
                if not anchor_words_idxes:
                    continue
                anchor = torch.mean(self.word_embeddings[anchor_words_idxes], dim=0, keepdim=True)
                
                # Positive sampling
                if self.positive_sampling == 'random':
                    positive_word_idx = np.random.choice(anchor_words_idxes)
                elif self.positive_sampling == 'hard':
                    pos_embs = self.word_embeddings[anchor_words_idxes]
                    pos_distances = F.pairwise_distance(anchor, pos_embs)
                    median_dist_idx = torch.argsort(pos_distances)[len(pos_distances) // 2]
                    positive_word_idx = anchor_words_idxes[median_dist_idx]
                else:
                    continue
                positive = self.word_embeddings[positive_word_idx].unsqueeze(0)
                
                # Negative sampling
                negative_candidates = []
                for neg_topic_idx in range(self.num_topics):
                    if neg_topic_idx not in group_topics:
                        negative_candidates.extend(self.topics[self.topic_index_mapping[neg_topic_idx]])
                if len(negative_candidates) < num_negatives:
                    continue
                if self.negative_sampling == 'random':
                    negative_word_idxes = np.random.choice(negative_candidates, size=num_negatives, replace=False)
                elif self.negative_sampling == 'hard':
                    neg_embs = self.word_embeddings[negative_candidates]
                    neg_distances = F.pairwise_distance(anchor.repeat(len(negative_candidates), 1), neg_embs)
                    hard_negative_indices = torch.argsort(neg_distances)[:num_negatives]
                    negative_word_idxes = [negative_candidates[i] for i in hard_negative_indices]
                else:
                    continue
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

    def get_loss_TP(self, contextual_emb):
        cost = self.pairwise_euclidean_distance(contextual_emb, contextual_emb)
        matrixP = self.create_matrixP(contextual_emb)
        return self.TP(cost, matrixP)

    def create_matrixP(self, minibatch_embeddings):
        norm_embeddings = F.normalize(minibatch_embeddings, p=2, dim=1)
        return torch.matmul(norm_embeddings, norm_embeddings.T).clamp(min=1e-4)

    def get_loss_DTC(self, contextual_emb, theta, margin=0.2, k=5, gamma=32):
        anchor_doc_emb = self.document_emb_prj(contextual_emb)
        _, top_positive_indices = torch.topk(theta, k=1, dim=1)
        _, bottom_negative_indices = torch.topk(theta, k=k, dim=1, largest=False)
        pos_emb = self.topic_embeddings[top_positive_indices].squeeze(1)
        neg_emb = self.topic_embeddings[bottom_negative_indices].mean(dim=1)

        loss = 0.0
        for i in range(anchor_doc_emb.size(0)):
            if self.contrastive_loss_type == 'infonce':
                loss += info_nce_loss(anchor_doc_emb[i:i+1], pos_emb[i:i+1], neg_emb[i:i+1])
            elif self.contrastive_loss_type == 'circle':
                loss += circle_loss(anchor_doc_emb[i:i+1], pos_emb[i:i+1], neg_emb[i:i+1], margin=margin, gamma=gamma)
            elif self.contrastive_loss_type == 'triplet':
                loss += triplet_loss(anchor_doc_emb[i:i+1], pos_emb[i:i+1], neg_emb[i:i+1], margin=margin, metric=self.metric_CL)
            else:
                raise ValueError(f"Unknown contrastive_loss_type: {self.contrastive_loss_type}")
        return (loss / anchor_doc_emb.size(0)) * self.weight_loss_DTC


    def pairwise_euclidean_distance(self, x, y):
        return torch.sum(x ** 2, dim=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())

    def forward(self, indices, input, epoch_id=None):
        bow, contextual_emb = input[0], input[1]
        combined_x = torch.cat((bow, contextual_emb), dim=1)
        theta, mu, logvar = self.get_theta(combined_x)
        recon_x = self.decode(theta)
        loss_TM = self.loss_function(bow, recon_x, mu, logvar)

        loss_TP = loss_CLC = loss_CLT = loss_DTC = 0.0

        if epoch_id is not None and epoch_id >= self.threshold_epoch:
            loss_TP = self.get_loss_TP(contextual_emb)
            if epoch_id == self.threshold_epoch or (epoch_id > self.threshold_epoch and epoch_id % self.threshold_cluster == 0):
                self.create_group_topic()
            if self.group_topic is not None:
                if self.weight_loss_CLC > 0:
                    loss_CLC = self.get_loss_CLC()
                if self.weight_loss_CLT > 0:
                    loss_CLT = self.get_loss_CLT()
                if self.weight_loss_DTC > 0:
                    loss_DTC = self.get_loss_DTC(contextual_emb, theta)

        loss = loss_TM + loss_TP + loss_CLC + loss_CLT + loss_DTC
        return {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_TP': loss_TP,
            'loss_CLC': loss_CLC,
            'loss_CLT': loss_CLT,
            'loss_DTC': loss_DTC,
        }