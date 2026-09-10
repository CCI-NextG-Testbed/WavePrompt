import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

params_simple = AttrDict(
    log_dir='./log/simple',
    model_dir='./model/simple',
    cvae_model_dir='./cvae_training/cvae_model',
    out_dir='./dataset/simple/output/prediction.mat',
    data_dir=['./dataset/simple/raw'],  # list of folders
    max_iter=None,
    inference_batch_size=1,
    robust_sampling=True,
    batch_size=1,
    learning_rate=1e-5,
    max_grad_norm=None,
    use_tfdiff_loss=False,
    loss_w_fft=0.1,
    loss_w_time=0.3,
    symbol_tau=0.1,
    mod_loss_weight=0.0,
    test_per_mod=3,
    test_mods=["BPSK", "QPSK", "8PSK", "16QAM"],
    split_seed=42,
    sample_rate=4096,              # length of each signal
    input_dim=1,
    extra_dim=[1],
    embed_dim=128,
    hidden_dim=64,
    num_heads=4,
    num_block=8,
    dropout=0.0,
    mlp_ratio=4,
    learn_tfdiff=False,
    signal_diffusion=True,        # use GaussianDiffusion or set True if you prepared blur schedule
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
    animate_after_training=False,
    training_animation_out="./results/training_mods.gif",
    training_animation_mods=["BPSK", "QPSK", "8PSK", "16QAM"],
    training_animation_wave_samples=400,
    training_animation_wave_stride=2,
    training_animation_max_symbols=256,
    training_animation_fps=2,
)
