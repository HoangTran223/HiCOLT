import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from utils import static_utils
import logging
import os
import scipy


class BasicTrainer:
    def __init__(self, model, model_name='model_name', epochs=200, learning_rate=0.002, batch_size=200, lr_scheduler=None, lr_step_size=125, log_interval=5, 
                     device='cuda'):
        self.model = model
        self.model_name = model_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval
        self.device = device
        self.logger = logging.getLogger('main')


    def make_adam_optimizer(self,):
        args_dict = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
        }

        optimizer = torch.optim.Adam(**args_dict)
        return optimizer
    
    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            lr_scheduler = StepLR(
                optimizer, step_size=self.lr_step_size, gamma=0.5)
        else:
            raise NotImplementedError(self.lr_scheduler)
        return lr_scheduler

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
    
        if self.model_name in ['ETM_Plus', 'ECRTM_Plus', 'FASTopic_Plus', 'NeuroMax_Plus', 'CombinedTM_Plus']:
            train_theta = self.test(dataset_handler.train_data, dataset_handler.train_doc_embeddings)
        elif self.model_name in ['ETM', 'CombinedTM']:
            train_theta = self.test(dataset_handler.train_data)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return top_words, train_theta

    def train(self, dataset_handler, verbose=True):  
        adam_optimizer = self.make_adam_optimizer()

        if self.lr_scheduler:
            print("===>using lr_scheduler")
            self.logger.info("===>using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(adam_optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)

        for epoch_id, epoch in enumerate(tqdm(range(1, self.epochs + 1))):
            self.model.train()
            loss_rst_dict = defaultdict(float)

            for batch_id, batch in enumerate(dataset_handler.train_dataloader): 
                if self.model_name in ['NeuroMax_Plus', 'CombinedTM', 'CombinedTM_Plus']:
                    bow, contextual_emb, indices = batch
                    rst_dict = self.model(indices, (bow, contextual_emb), epoch_id=epoch)

                elif self.model_name in ['ETM_Plus', 'ECRTM_Plus']:
                    bow, indices, doc_embeddings = batch
                    rst_dict = self.model(indices, (bow,), epoch_id=epoch, doc_embeddings=doc_embeddings)
                
                elif self.model_name == 'FASTopic_Plus':
                    bow, indices, doc_embeddings = batch
                    rst_dict = self.model(indices=indices, input=(bow,), epoch_id=epoch, doc_embeddings=doc_embeddings)
                
                elif self.model_name == 'ETM':
                    bow, indices = batch
                    rst_dict = self.model(indices, (bow,), epoch_id=epoch)
                else:
                    # Fallback for other models
                    *inputs, indices = batch
                    rst_dict = self.model(indices=indices, input=inputs, epoch_id=epoch)

                batch_data = batch[0]
                batch_loss = rst_dict['loss']

                batch_loss.backward()
                adam_optimizer.step()
                adam_optimizer.zero_grad()

                for key in rst_dict:
                    try:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data[0])
                    except:
                        loss_rst_dict[key] += rst_dict[key] * len(batch_data)

            if self.lr_scheduler:
                lr_scheduler.step()

            if epoch % 1 == 0: 
                output_log = f'Epoch: {epoch:03d}'
                for key in loss_rst_dict:
                    loss_val = loss_rst_dict[key] / data_size
                    output_log += f' {key}: {loss_val:.6f}'

                self.logger.info(output_log)

    def test(self, input_data, train_doc_embeddings=None):
        data_size = input_data.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)
        with torch.no_grad():
            self.model.eval()
            for idx in all_idx:
                batch_input = input_data[idx]
                # Convert to tensor if needed
                if isinstance(batch_input, np.ndarray):
                    batch_input = torch.from_numpy(batch_input).float().to(self.device)
                if train_doc_embeddings is not None and isinstance(train_doc_embeddings, np.ndarray):
                    train_doc_embeddings_tensor = torch.from_numpy(train_doc_embeddings).float().to(self.device)
                else:
                    train_doc_embeddings_tensor = train_doc_embeddings
                
                if hasattr(self.model, 'get_theta') and self.model.__class__.__name__ in ['FASTopic', 'FASTopic_Plus']:
                    batch_proj = self.model.document_emb_prj(batch_input)
                    train_proj = self.model.document_emb_prj(train_doc_embeddings_tensor)
                    batch_theta = self.model.get_theta(batch_proj, train_proj)

                elif self.model_name in ['CombinedTM', 'CombinedTM_Plus']:
                    batch_theta = self.model.get_theta(batch_input)

                elif hasattr(self.model, 'get_theta'):
                    batch_theta = self.model.get_theta(batch_input)
                    if isinstance(batch_theta, tuple):
                        batch_theta = batch_theta[0]
                else:
                    raise ValueError(f"Model {self.model.__class__.__name__} does not have get_theta method")
                
                theta.extend(batch_theta.cpu().tolist())
        theta = np.asarray(theta)
        return theta


    def export_beta(self):
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        top_words = static_utils.print_topic_words(beta, vocab, num_top_words)
        return top_words

    def export_theta(self, dataset_handler):
        # For FASTopic/FASTopic_Plus, use doc_embeddings, not BOW
        if hasattr(self.model, 'get_theta') and self.model.__class__.__name__ in ['FASTopic', 'FASTopic_Plus']:
            if hasattr(dataset_handler, 'train_contextual_embed') and dataset_handler.train_contextual_embed is not None:
                train_embeddings = dataset_handler.train_contextual_embed
                test_embeddings = dataset_handler.test_contextual_embed
            else:
                train_embeddings = dataset_handler.train_doc_embeddings
                test_embeddings = dataset_handler.test_doc_embeddings
            train_theta = self.test(train_embeddings, train_embeddings)
            test_theta = self.test(test_embeddings, train_embeddings)

        elif self.model_name in ['ETM_Plus']:
            train_theta = self.test(dataset_handler.train_data, dataset_handler.train_doc_embeddings)
            test_theta = self.test(dataset_handler.test_data, dataset_handler.train_doc_embeddings)
        
        elif self.model_name in ['CombinedTM_Plus', 'CombinedTM']:
            train_contextual_tensor = torch.from_numpy(dataset_handler.train_contextual_embed).float().to(self.device)
            test_contextual_tensor = torch.from_numpy(dataset_handler.test_contextual_embed).float().to(self.device)
            combined_train_data = torch.cat([dataset_handler.train_data, train_contextual_tensor], dim=1)
            combined_test_data = torch.cat([dataset_handler.test_data, test_contextual_tensor], dim=1)
            train_theta = self.test(combined_train_data)
            test_theta = self.test(combined_test_data)

        else:
            train_theta = self.test(dataset_handler.train_data)
            test_theta = self.test(dataset_handler.test_data)
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
            for i, words in enumerate(top_words):
                f.write(words + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
        train_argmax_theta = np.argmax(train_theta, axis=1)
        test_argmax_theta = np.argmax(test_theta, axis=1)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        word_embeddings = None
        topic_embeddings = None
        
        if hasattr(self.model, 'word_embeddings'):
            word_embeddings = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
            self.logger.info(f'word_embeddings size: {word_embeddings.shape}')

        if hasattr(self.model, 'topic_embeddings'):
            topic_embeddings = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                    topic_embeddings)
            self.logger.info(
                f'topic_embeddings size: {topic_embeddings.shape}')

            if topic_embeddings is not None:
                topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
                np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

        return word_embeddings, topic_embeddings