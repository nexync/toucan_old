import argparse
import yaml
import inspect
import os
import torch

from hourglass import MemTransformerLM
from data_utils import get_lm_corpus
from utils.exp_utils import init_seed

def generate(model, corpus, prompt, num_tokens=1):
    with torch.no_grad():
        x = corpus.vocab.convert_to_tensor(prompt).to(list(model.parameters())[0].device)[..., None]
        out = model.generate_next(x, num_tokens=num_tokens)
        return corpus.vocab.convert_to_sent(out)
    
def tokenize(model, corpus, prompt):
    with torch.no_grad():
        x = corpus.vocab.convert_to_tensor(prompt).to(list(model.parameters())[0].device)[..., None]
        _, boundaries, _, _ = model(x, None, None)
    tokens = []
    token = ""
    for i, p in enumerate(boundaries.cpu().squeeze().numpy()):
        if p > 0.5:
            tokens.append(token)
            token = prompt[i]
        else:
            token = token + prompt[i]
    if len(token) > 0:
        tokens.append(token)
    return ':'.join(tokens)
    

def parse_args():
    parent_parser = argparse.ArgumentParser(add_help=False)

    parser = argparse.ArgumentParser(parents=[parent_parser])
    cfg_parser = argparse.ArgumentParser(parents=[parent_parser])
    cfg_parser.add_argument('--config', default='default')
    cfg_parser.add_argument('--config_file', default=None)

    config_args, _ = cfg_parser.parse_known_args()

    assert config_args.config is not None and config_args.config_file is not None
    with open(config_args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[config_args.config]['train']

    # Main args
    general = parser.add_argument_group('general setup')
    general.add_argument('--work_dir', default='LM-TFM', type=str,
                         help='Directory for the results')
    general.add_argument('--cuda', action='store_true',
                         help='Run training on a GPU using CUDA')
    general.add_argument('--log_interval', type=int, default=10,
                         help='Report interval')

    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--data', type=str, help='Location of the data corpus')
    dataset.add_argument('--dataset', type=str, help='Dataset name')

    model = parser.add_argument_group('model setup')
    model.add_argument('--n_head', type=int, default=8,
                       help='Number of heads')
    model.add_argument('--d_head', type=int, default=64,
                       help='Head dimension')
    model.add_argument('--d_model', type=int, default=512,
                       help='Model dimension')
    model.add_argument('--d_inner', type=int, default=2048,
                       help='Inner dimension in feedforward layer')
    model.add_argument('--dropout', type=float, default=0.1,
                       help='Global dropout rate')
    model.add_argument('--dropatt', type=float, default=0.0,
                       help='Attention probability dropout rate')
    model.add_argument('--pre_lnorm', action='store_true',
                       help='Apply LayerNorm to the input instead of the output')
    model.add_argument('--init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--emb_init', default='normal', type=str,
                       help='Parameter initializer to use')
    model.add_argument('--init_range', type=float, default=0.1,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--emb_init_range', type=float, default=0.01,
                       help='Parameters initialized by U(-init_range, init_range)')
    model.add_argument('--init_std', type=float, default=0.02,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--proj_init_std', type=float, default=0.01,
                       help='Parameters initialized by N(0, init_std)')
    model.add_argument('--model_config', type=str, default="[3, (8,) ,3]",
                       help="[pre_layers, (shortened_layers, ), post_layers]")
    model.add_argument('--activation_function', type=str, default='relu')
    model.add_argument('--add_eot', default=False, type=bool)

    boundaries = parser.add_argument_group('boundary creator')
    boundaries.add_argument('--boundaries_type', type=str)
    boundaries.add_argument('--tokenizer_path', type=str)
    boundaries.add_argument('--fixed_sf', type=int)
    boundaries.add_argument('--spikes_left', type=int)
    boundaries.add_argument('--temp', type=float)
    boundaries.add_argument('--prior', type=float)

    opt = parser.add_argument_group('optimizer setup')
    opt.add_argument('--optim', default='adam', type=str, choices=['adam'],
                     help='Optimizer to use')
    opt.add_argument('--lr', type=float, default=0.00025,
                     help='Initial learning rate')
    opt.add_argument('--scheduler', default='cosine', type=str,
                     choices=['cosine'], help='LR scheduler to use')
    opt.add_argument('--warmup_step', type=int, default=1000,
                     help='Number of iterations for LR warmup')
    opt.add_argument('--clip', type=float, default=0.25,
                     help='Gradient clipping')
    opt.add_argument('--weight_decay', type=float, default=0.0,
                     help='Weight decay for adam')
    opt.add_argument('--adam_b1', type=float, default=0.9)
    opt.add_argument('--adam_b2', type=float, default=0.999)
    opt.add_argument('--adam_eps', type=float, default=1e-8)

    training = parser.add_argument_group('training setup')
    training.add_argument('--max_step', type=int, default=40000,
                          help='Max number of training steps')
    training.add_argument('--batch_size', type=int, default=256,
                          help='Global batch size')
    training.add_argument('--batch_chunk', type=int, default=1,
                          help='Split batch into chunks and train with '
                          'gradient accumulation')
    training.add_argument('--roll', action='store_true',
                          help='Enable random shifts within each data stream')
    training.add_argument('--shuffle', action='store_true',
                          help='Shuffle text chunks')
    training.add_argument('--fp16', action='store_true', help='Use cuda fp16')
    training.add_argument('--tgt_len', type=int, default=192,
                          help='Number of tokens to predict')
    training.add_argument('--seed', type=int, default=1111,
                          help='Random seed')
    training.add_argument('--nw', type=int, default=0,
                          help='Number of workers')

    val = parser.add_argument_group('validation setup')
    val.add_argument('--eval_tgt_len', type=int)
    val.add_argument('--eval_total_len', type=int)
    val.add_argument('--eval_max_steps', type=int, default=-1,
                     help='Max eval steps')
    val.add_argument('--eval_interval', type=int, default=5000,
                     help='Evaluation interval')
    val.add_argument('--ckpt_path', type=str, default="")

    dist = parser.add_argument_group('distributed setup')
    dist.add_argument('--local_rank', type=int,
                      default=os.getenv('LOCAL_RANK', 0),
                      help='Used for multi-process training.')

    parser.set_defaults(**config)
    args, _ = parser.parse_known_args()

    args.ckpt_path = '/'.join(config_args.config_file.split('/')[:-1])
    args.eval_batch_size = args.batch_size * 2

    if args.batch_size % args.batch_chunk != 0:
        raise RuntimeError('Batch size needs to be divisible by batch chunk')

    assert args.boundaries_type in [
        "none",
        "fixed",
        "whitespaces",
        "unigram",
        "entropy",
        "gumbel",
        "eot"
    ]

    return args
    

def main():
    args = parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    init_seed(args.seed + 1)

    boundary_kwargs = {
        'boundaries_type': args.boundaries_type,
        'fixed_sf': args.fixed_sf,
        'tokenizer_path': args.tokenizer_path,
    }

    corpus = get_lm_corpus(args.data,
                          args.dataset,
                        **boundary_kwargs)
    vocab = corpus.vocab
    args.n_token = len(vocab)

    def get_model_config():
        model_args = inspect.getfullargspec(MemTransformerLM).args
        assert model_args.index('self') == 0
        model_args = model_args[1:]
        return {arg: getattr(args, arg) for arg in model_args}

    # Load model
    model = MemTransformerLM(**get_model_config())
    model_state = torch.load(args.work_dir+'/checkpoint_last.pt')['model_state']
    model_state = {'.'.join(key.split('.')[1:]): val for key,val in model_state.items()}
    model.load_state_dict(model_state)
    
    # Model to GPU
    model = model.to(device)
    model.eval()
    
    print(tokenize(model, corpus, "toucan is pretty cool"))
    print(generate(model, corpus, "george washington was the ", num_tokens=10))
    
if __name__ == '__main__':
    main()