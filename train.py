from json import decoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

from argparse import ArgumentParser

from stablediff.params import params_simple
from stablediff.learner import tfdiffLearner
from stablediff.models import stablediff_Simple
from stablediff.clip import CLIP
from stablediff.dataset import from_path_split
from transformers import CLIPTokenizer
from stablediff.CVAE import ComplexEncoder, ComplexDecoder
from stablediff.model_converter import load_from_standard_weights

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _train_impl(replica_id, model, clip_model, encoder, decoder, tokenizer, dataset, params, val_dataset=None):
    opt = torch.optim.AdamW(model.parameters(), lr=params.learning_rate, capturable=False, foreach=False, fused=False)
    learner = tfdiffLearner(
        params.log_dir,
        params.model_dir,
        model,
        clip_model,
        encoder,
        decoder,
        tokenizer,
        dataset,
        opt,
        params,
        val_dataset=val_dataset,
    )
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()
    learner.train(max_iter=params.max_iter)


def train(params):
    dataset, val_dataset = from_path_split(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = load_from_standard_weights(
        os.path.join(params.tokenizer_dir,"v1-5-pruned-emaonly.ckpt"), device)

    model = stablediff_Simple(params).to(device)

    clip_model = CLIP().to(device)
    clip_model.load_state_dict(state_dict['clip'], strict=True)
    tokenizer = CLIPTokenizer(os.path.join(params.tokenizer_dir, "vocab.json"), 
                            merges_file=os.path.join(params.tokenizer_dir, "merges.txt"))
    
    encoder = ComplexEncoder(params.latent_dims).to(device)
    decoder = ComplexDecoder(params.latent_dims).to(device)

    checkpoint = torch.load(
        params.cvae_model_dir,
        map_location=device
    )

    encoder.load_state_dict(
        checkpoint["encoder_state_dict"]
    )

    decoder.load_state_dict(
        checkpoint["decoder_state_dict"]
    )

    encoder.eval()
    decoder.eval()
    clip_model.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    for param in decoder.parameters():
        param.requires_grad = False

    for param in clip_model.parameters():
        param.requires_grad = False
    _train_impl(0, model, clip_model, encoder, decoder, tokenizer, dataset, params, val_dataset=val_dataset)

def main(args):
    params = params_simple
    if args.batch_size is not None:
        params.batch_size = args.batch_size
    if args.model_dir is not None:
        params.model_dir = args.model_dir
    if args.data_dir is not None:
        params.data_dir = args.data_dir
    if args.log_dir is not None:
        params.log_dir = args.log_dir
    if args.max_iter is not None:
        params.max_iter = args.max_iter
    if args.animate_training:
        params.animate_after_training = True
    if args.animation_out is not None:
        params.training_animation_out = args.animation_out
    if args.use_tfdiff_loss:
        params.use_tfdiff_loss = True
    if args.loss_w_fft is not None:
        params.loss_w_fft = args.loss_w_fft
    if args.loss_w_time is not None:
        params.loss_w_time = args.loss_w_time
    if args.learning_rate is not None:
        params.learning_rate = args.learning_rate
    if args.training_metrics_csv is not None:
        params.training_metrics_csv = args.training_metrics_csv
    params.test_per_mod = args.test_per_mod
    params.test_mods = args.test_mods
    params.split_seed = args.split_seed
    train(params)


# python train.py  --model_dir [model_dir] --data_dir [data_dir]
# HF_ENV_NAME=py38-202207 hfai python train.py --model_dir [model_dir] --data_dir [data_dir] --max_iter [iter_num] --batch_size [batch_size] -- -n [node_num] --force
if __name__ == '__main__':
    parser = ArgumentParser(
        description='train (or resume training) a tfdiff model')
    parser.add_argument('--model_dir', default=None,
                        help='directory in which to store model checkpoints and training logs')
    parser.add_argument('--data_dir', default=None, nargs='+',
                        help='space separated list of directories from which to read csi files for training')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--max_iter', default=None, type=int,
                        help='maximum number of training iteration')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--animate_training', action='store_true',
                        help='generate training animation automatically when training ends')
    parser.add_argument('--animation_out', default=None,
                        help='output path for training animation (.gif or .mp4)')
    parser.add_argument('--use_tfdiff_loss', action='store_true',
                        help='use tfdiffLoss instead of IQPlusBitsLoss')
    parser.add_argument('--loss_w_fft', default=None, type=float,
                        help='weight for the noise term in tfdiffLoss')
    parser.add_argument('--loss_w_time', default=None, type=float,
                        help='weight for L_time in IQPlusBitsLoss; L_evm is derived as 1 - loss_w_time')
    parser.add_argument('--learning_rate', default=None, type=float,
                        help='optimizer learning rate override')
    parser.add_argument('--training_metrics_csv', default=None,
                        help='output CSV path for per-epoch convergence tracking')
    parser.add_argument('--test_per_mod', default=1, type=int,
                        help='number of held-out test samples per modulation')
    parser.add_argument('--test_mods', default=['BPSK', 'QPSK', '8PSK', '16QAM'], nargs='+',
                        help='modulations to hold out for the test set')
    parser.add_argument('--split_seed', default=42, type=int,
                        help='random seed used for train/test holdout selection')
    main(parser.parse_args())
