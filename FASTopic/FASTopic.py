import torch
from torch import nn
import torch.nn.functional as F
from ._ETP import ETP
from ._model_utils import pairwise_euclidean_distance

class FASTopic(nn.Module):
    def __init__(self,
                 vocab_size: int, embed_size: 200, num_topics: int,
                 cluster_distribution=None,
                 cluster_mean=None,
                 cluster_label=None,
                 theta_temp: float=1.0,
                 DT_alpha: float=3.0,
                 TW_alpha: float=2.0,
                 sinkhorn_alpha=20.0, sinkhorn_max_iter=1000,
                 device = 'cuda'
                ):
        super().__init__()
        self.device = device
        self.DT_alpha = DT_alpha
        self.TW_alpha = TW_alpha
        self.theta_temp = theta_temp
        self.num_topics = num_topics
        self.epsilon = 1e-12
        
        self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))
        
        self.topic_embeddings = torch.empty((self.num_topics, embed_size))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        self.word_weights = nn.Parameter((torch.ones(vocab_size) / vocab_size).unsqueeze(1))
        self.topic_weights = nn.Parameter((torch.ones(self.num_topics) / self.num_topics).unsqueeze(1))

        self.DT_ETP = ETP(self.DT_alpha, init_b_dist=self.topic_weights)
        self.TW_ETP = ETP(self.TW_alpha, init_b_dist=self.word_weights)

    def get_transp_DT(self,
                      doc_embeddings,
                    ):

        topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
        _, transp = self.DT_ETP(doc_embeddings, topic_embeddings)

        return transp.detach().cpu().numpy()

    # only for testing
    def get_beta(self):
        _, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)
        # use transport plan as beta
        beta = transp_TW * transp_TW.shape[0]

        return beta

    # only for testing
    def get_theta(self,
                  doc_embeddings,
                  train_doc_embeddings
                ):
        topic_embeddings = self.topic_embeddings.detach().to(doc_embeddings.device)
        dist = pairwise_euclidean_distance(doc_embeddings, topic_embeddings)
        train_dist = pairwise_euclidean_distance(train_doc_embeddings, topic_embeddings)

        exp_dist = torch.exp(-dist / self.theta_temp)
        exp_train_dist = torch.exp(-train_dist / self.theta_temp)

        theta = exp_dist / (exp_train_dist.sum(0))
        theta = theta / theta.sum(1, keepdim=True)

        return theta
 
    def forward(self, indices, input, epoch_id=None, iseval=False, dataset_handler=None):
        train_bow = input[0]
        doc_embeddings = input[1]
        loss_DT, loss_TW = 0, 0
        if iseval == False:
            loss_DT, transp_DT = self.DT_ETP(doc_embeddings, self.topic_embeddings)
            loss_TW, transp_TW = self.TW_ETP(self.topic_embeddings, self.word_embeddings)

            loss_ETP = loss_DT + loss_TW

            theta = transp_DT * transp_DT.shape[0]
            beta = transp_TW * transp_TW.shape[0]
        else:
            theta = self.get_theta(doc_embeddings, dataset_handler.train_contextual_embed)
            beta = self.get_beta()
            loss_ETP = 0

        # Dual Semantic-relation Reconstruction
        recon = torch.matmul(theta, beta)

        loss_DSR = -(train_bow * (recon + self.epsilon).log()).sum(axis=1).mean()
        loss = loss_DSR + loss_ETP

        rst_dict = {
            'loss_': loss,
            'lossDSR': loss_DSR,
            'lossETP': loss_ETP,
        }

        return rst_dict



