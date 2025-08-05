from utils import config, log, miscellaneous, seed
import os
import numpy as np
import basic_trainer
from ECRTM.ECRTM_Plus import ECRTM_Plus
from ETM.ETM_Plus import ETM_Plus
from FASTopic.FASTopic_Plus import FASTopic_Plus
from FASTopic.FASTopic import FASTopic
from ETM.ETM import ETM
from NeuroMax.NeuroMax_Plus import NeuroMax_Plus
from CTM.CombinedTM_Plus import CombinedTM_Plus
from CTM.CombinedTM import CombinedTM
import evaluations
import datasethandler
import scipy
import torch
from tqdm import tqdm

RESULT_DIR = 'results'
DATA_DIR = 'datasets'

def get_model_params_vector(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)

def set_model_params_vector(model, vector):
    pointer = 0
    for param in model.parameters():
        num_param = param.numel() 
        param.data.copy_(vector[pointer:pointer + num_param].view(param.size()))
        pointer += num_param

def random_directions(param_vector):
    direction1 = torch.randn_like(param_vector)
    direction2 = torch.randn_like(param_vector)
    return direction1, direction2

if __name__ == "__main__":
    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_model_argument(parser)
    config.add_training_argument(parser)
    config.add_eval_argument(parser)
    args = parser.parse_args()
    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR, current_time)
    miscellaneous.create_folder_if_not_exist(current_run_dir)
    config.save_config(args, os.path.join(current_run_dir, 'config.txt'))
    seed.seedEverything(args.seed)
    print(args)

    if args.dataset in ['20NG', 'AGNews', 'IMDB']:
        read_labels = True
    else:
        read_labels = False
    print(f"read labels = {read_labels}")

    contextual_needed = args.model in ["CombinedTM", "CombinedTM_Plus", "ECRTM_Plus", "NeuroMax_Plus", "ETM_Plus", "FASTopic_Plus"]

    dataset = datasethandler.BasicDatasetHandler(
        os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
        as_tensor=True, contextual_embed=contextual_needed, args=args)
    
    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, args.dataset, "word_embeddings.npz")).toarray()
    
    if args.model == "ECRTM_Plus":
        print(f"ECRTM_Plus parameters:")
        model = ECRTM_Plus(
            vocab_size=dataset.vocab_size,
            data_name=args.dataset,
            num_topics=args.num_topics,
            dropout=args.dropout,
            pretrained_WE=pretrainWE if args.use_pretrainWE else None,
            weight_loss_ECR=args.weight_ECR,
            alpha_ECR=args.alpha_ECR,
            weight_loss_TP=args.weight_loss_TP,
            alpha_TP=args.alpha_TP,
            beta_temp=args.beta_temp,
            vocab=dataset.vocab,
            weight_loss_CLC=args.weight_loss_CLC,
            max_clusters=args.max_clusters,
            weight_loss_CLT=args.weight_loss_CLT,
            threshold_epoch=args.threshold_epoch,
            threshold_cluster=args.threshold_cluster,
            doc_embeddings=torch.tensor(dataset.train_doc_embeddings).float().to(args.device),
            method_CL=args.method_CL,
            metric_CL=args.metric_CL,
            weight_loss_DTC=args.weight_loss_DTC,
            positive_sampling=args.positive_sampling,
            negative_sampling=args.negative_sampling,
            contrastive_loss_type=getattr(args, 'contrastive_loss_type', 'triplet')
        )
        model = model.to(args.device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")

    elif args.model == "ETM_Plus":
        print(f"ETM_Plus parameters:")
        model = ETM_Plus(
            vocab_size=dataset.vocab_size,
            num_topics=args.num_topics,
            dropout=args.dropout,
            pretrained_WE=pretrainWE if args.use_pretrainWE else None,
            weight_loss_TP=args.weight_loss_TP,
            weight_loss_CLC=args.weight_loss_CLC,
            alpha_TP=args.alpha_TP,
            weight_loss_CLT=args.weight_loss_CLT,
            weight_loss_DTC=args.weight_loss_DTC,
            threshold_epoch=args.threshold_epoch,
            threshold_cluster=args.threshold_cluster,
            metric_CL=args.metric_CL,
            max_clusters=args.max_clusters,
            doc_embeddings=torch.tensor(dataset.train_doc_embeddings).float().to(args.device),
            vocab=dataset.vocab,
            device=args.device,
            positive_sampling=args.positive_sampling,
            negative_sampling=args.negative_sampling,
            contrastive_loss_type=getattr(args, 'contrastive_loss_type', 'triplet')
        )
        model = model.to(args.device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")


    elif args.model == "FASTopic_Plus":
        print(f"FASTopic_Plus parameters:")

        if not (hasattr(dataset, 'train_contextual_embed') and dataset.train_contextual_embed is not None):
            raise ValueError("FASTopic_Plus requires contextual embeddings (SBERT), but they were not loaded. Check dataset handler.")
        contextual_embed_size = dataset.train_contextual_embed.shape[1]

        model = FASTopic_Plus(
            vocab_size=dataset.vocab_size,
            embed_size=args.embed_size,
            num_topics=args.num_topics,
            contextual_embed_size=contextual_embed_size,
            DT_alpha=args.DT_alpha,
            TW_alpha=args.TW_alpha,
            theta_temp=args.theta_temp,
            weight_loss_CLT=args.weight_loss_CLT,
            weight_loss_CLC=args.weight_loss_CLC,
            weight_loss_DTC=args.weight_loss_DTC,
            weight_loss_TP=args.weight_loss_TP,
            threshold_epoch=args.threshold_epoch,
            threshold_cluster=args.threshold_cluster,
            metric_CL=args.metric_CL,
            max_clusters=args.max_clusters,
            sinkhorn_alpha=args.sinkhorn_alpha,
            OT_max_iter=getattr(args, 'sinkhorn_max_iter', 500),
            device=args.device,
            positive_sampling=args.positive_sampling,
            negative_sampling=args.negative_sampling,
            contrastive_loss_type=getattr(args, 'contrastive_loss_type', 'triplet')
        )
        model = model.to(args.device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")

    elif args.model == "NeuroMax_Plus":
        print(f"NeuroMax_Plus parameters:")
        model = NeuroMax_Plus(
            vocab_size=dataset.vocab_size,
            data_name=args.dataset,
            embed_size=args.embed_size if hasattr(args, 'embed_size') else 200,
            num_topics=args.num_topics,
            num_groups=args.num_groups if hasattr(args, 'num_groups') else 10,
            en_units=args.en_units if hasattr(args, 'en_units') else 200,
            dropout=args.dropout,
            pretrained_WE=pretrainWE if args.use_pretrainWE else None,
            beta_temp=args.beta_temp,
            weight_loss_ECR=args.weight_ECR,
            weight_loss_GR=getattr(args, 'weight_loss_GR', 250.0),
            alpha_GR=getattr(args, 'alpha_GR', 20.0),
            alpha_ECR=args.alpha_ECR,
            sinkhorn_alpha=getattr(args, 'sinkhorn_alpha', 20.0),
            sinkhorn_max_iter=getattr(args, 'sinkhorn_max_iter', 500),
            weight_loss_InfoNCE=getattr(args, 'weight_loss_InfoNCE', 10.0),
            weight_loss_CLT=args.weight_loss_CLT,
            weight_loss_CLC=args.weight_loss_CLC,
            alpha_TP=args.alpha_TP,
            weight_loss_DTC=args.weight_loss_DTC,
            weight_loss_TP=args.weight_loss_TP,
            threshold_epoch=args.threshold_epoch,
            threshold_cluster=args.threshold_cluster,
            metric_CL=args.metric_CL,
            max_clusters=args.max_clusters,
            vocab=dataset.vocab,
            device=args.device,
            contextual_embed_size=dataset.contextual_embed_size,
            positive_sampling=args.positive_sampling,
            negative_sampling=args.negative_sampling,
            contrastive_loss_type=getattr(args, 'contrastive_loss_type', 'triplet')
        )
        model = model.to(args.device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")
    
    elif args.model == "CombinedTM_Plus":
        print(f"CombinedTM_Plus parameters:")
        model = CombinedTM_Plus(
            vocab_size=dataset.vocab_size,
            contextual_embed_size=dataset.contextual_embed_size,
            num_topics=args.num_topics,
            dropout=args.dropout,
            vocab=dataset.vocab,
            weight_loss_CLT=args.weight_loss_CLT,
            weight_loss_CLC=args.weight_loss_CLC,
            weight_loss_DTC=args.weight_loss_DTC,
            weight_loss_TP=args.weight_loss_TP,
            alpha_TP=args.alpha_TP,
            threshold_epoch=args.threshold_epoch,
            threshold_cluster=args.threshold_cluster,
            metric_CL=args.metric_CL,
            max_clusters=args.max_clusters,
            sinkhorn_max_iter=getattr(args, 'sinkhorn_max_iter', 1000),
            positive_sampling=args.positive_sampling,
            negative_sampling=args.negative_sampling,
            contrastive_loss_type=getattr(args, 'contrastive_loss_type', 'triplet')
        )
        model = model.to(args.device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")

    else:
        raise ValueError(f"Unknown model: {args.model}")
    trainer = basic_trainer.BasicTrainer(model, model_name=args.model,
                                            epochs=args.epochs,
                                            learning_rate=args.lr,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size,
                                            device=args.device
                                            )

    trainer.train(dataset)
    trainer.save_embeddings(current_run_dir)
    beta = trainer.save_beta(current_run_dir)
    train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)

    top_words_10 = trainer.save_top_words(
        dataset.vocab, 10, current_run_dir)
    top_words_15 = trainer.save_top_words(
        dataset.vocab, 15, current_run_dir)
    top_words_20 = trainer.save_top_words(
        dataset.vocab, 20, current_run_dir)
    top_words_25 = trainer.save_top_words(
        dataset.vocab, 25, current_run_dir)

    train_theta_argmax = train_theta.argmax(axis=1)
    test_theta_argmax = test_theta.argmax(axis=1)    

    TD_15 = evaluations.compute_topic_diversity(
        top_words_15, _type="TD")
    print(f"TD_15: {TD_15:.5f}")

    if read_labels:

        clustering_results = evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])

    TC_15_list, TC_15 = evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_15.txt'))
    print(f"TC_15: {TC_15:.5f}")

    # if read_labels:
    #     classification_results = evaluations.evaluate_classification(
    #         train_theta, test_theta, dataset.train_labels, dataset.test_labels, tune=args.tune_SVM)
    #     print(f"Accuracy: ", classification_results['acc'])
    #     print(f"Macro-f1", classification_results['macro-F1'])

    filename = (
        f"results_"
        f"dataset{args.dataset}_model{args.model}_topics{args.num_topics}_epochs{args.epochs}"
        f"_lr{args.lr}_batch{args.batch_size}_seed{args.seed}"
        f"_ECR{args.weight_ECR}_aECR{args.alpha_ECR}"
        f"_TP{args.weight_loss_TP}_aTP{args.alpha_TP}"
        f"_dropout{args.dropout}_beta{args.beta_temp}"
        f"_DTC{getattr(args, 'weight_loss_DTC', 'NA')}_CLT{getattr(args, 'weight_loss_CLT', 'NA')}_CLC{getattr(args, 'weight_loss_CLC', 'NA')}"
        f"_maxclu{getattr(args, 'max_clusters', 'NA')}_threpoch{getattr(args, 'threshold_epoch', 'NA')}_thrclu{getattr(args, 'threshold_cluster', 'NA')}"
        f"_metric{getattr(args, 'metric_CL', 'NA')}"
        f".txt"
    )
    
    filename = filename.replace(' ', '_')
    filepath = os.path.join(current_run_dir, filename)
    with open(filepath, 'w') as f:
        if read_labels:
            f.write(f"NMI: {clustering_results['NMI']}\n")
            f.write(f"Purity: {clustering_results['Purity']}\n")
        else:
            f.write("NMI: N/A\n")
            f.write("Purity: N/A\n")
        f.write(f"TD_15: {TD_15:.5f}\n")
        f.write(f"TC_15: {TC_15:.5f}\n")
        # f.write(f"Accuracy: {classification_results['acc']:.5f}\n")
        # f.write(f"Macro-f1: {classification_results['macro-F1']:.5f}\n")
    print(f"Done in {filepath}")



