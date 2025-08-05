import argparse


def new_parser(name=None):
    return argparse.ArgumentParser(prog=name)


def add_dataset_argument(parser):
    parser.add_argument('--dataset', type=str,
                        help='dataset name, currently support datasets are: \
                            AGNews', default='AGNews')


def add_model_argument(parser):
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--num_topics', type=int, default=50)
    parser.add_argument('--embed_size', type=int, default=200, help='Embedding size for words and topics.')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_pretrainWE', action='store_true', default=False, help='Enable use_pretrainWE mode')
    
    parser.add_argument('--sinkhorn_alpha', type=float, default=20.0, help='Sinkhorn alpha for ETP in FASTopic.')
    parser.add_argument('--DT_alpha', type=float, default=3.0, help='Alpha for document-topic transport plan.')
    parser.add_argument('--TW_alpha', type=float, default=2.0, help='Alpha for topic-word transport plan.')
    parser.add_argument('--theta_temp', type=float, default=1.0, help='Temperature for theta calculation during inference.')

    parser.add_argument('--weight_ECR', type=float, default=40.)
    parser.add_argument('--alpha_ECR', type=float, default=20.)

    parser.add_argument('--alpha_TP', type=float, default=20.)
    parser.add_argument('--weight_loss_TP', type=float, default=250.)
    parser.add_argument('--weight_loss_CLC', type=float, default=1.)
    parser.add_argument('--max_clusters', type=int, default=9)
    parser.add_argument('--threshold_epoch', type=int, default=10)  
    parser.add_argument('--threshold_cluster', type=int, default=10)  
    parser.add_argument('--weight_loss_CLT', type=float, default=1.0)
    parser.add_argument('--weight_loss_DTC', type=float, default=1.0)
    parser.add_argument('--method_CL', type=str, default='HAC')
    parser.add_argument('--metric_CL', type=str, default='euclidean')
    parser.add_argument('--beta_temp', type=float, default=0.2)
    parser.add_argument('--positive_sampling', type=str, default='random', help='Strategy for positive sampling: random, hard')
    parser.add_argument('--negative_sampling', type=str, default='random', help='Strategy for negative sampling: random, hard')
    parser.add_argument('--contrastive_loss_type', type=str, default='triplet',
                        choices=['triplet', 'infonce', 'circle'],
                        help='Type of contrastive loss: triplet, infonce, or circle')



def add_eval_argument(parser):
    parser.add_argument('--tune_SVM', action='store_true', default=False)


def add_training_argument(parser):
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the model, cuda or cpu')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--lr_scheduler', type=str,
                        help='learning rate scheduler, dont use if not needed, \
                            currently support: step')
    parser.add_argument('--lr_step_size', type=int, default=125,
                        help='step size for learning rate scheduler')


def save_config(args, path):
    with open(path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')


def load_config(path):
    args = argparse.Namespace()
    with open(path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if value.isdigit():
                if value.find('.') != -1:
                    value = float(value)
                else:
                    value = int(value)
            setattr(args, key, value)
    print(args)
    return args
