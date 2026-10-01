import numpy as np
import os

import torch
import torch.nn as nn
import math
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from stablediff.diffusion import SignalDiffusion
from stablediff.dataset import _nested_map
try:
    from rfml.nn.F import evm as rfml_evm
except Exception:
    rfml_evm = None



class tfdiffLearner:
    def __init__(self, log_dir, model_dir, model, clip_model, encoder, decoder, tokenizer, dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.model = model
        self.clip = clip_model
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.val_dataset = kwargs.get("val_dataset", None)
        self.optimizer = optimizer
        self.device = model.device
        self.diffusion = SignalDiffusion(params)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
        )
        self.params = params
        self.iter = 0
        self.grad_norm = 0.0
        self.is_master = True
        self.summary_writer = None
        self.epoch_history = []
        self.epoch_train_losses = []
        self.epoch_test_losses = []
        self.snapshot_dir = os.path.join(self.model_dir, "training_snapshots")
        self.train_sample_count = int(getattr(self.dataset, "sample_count", len(getattr(self.dataset, "dataset", [])) if hasattr(self.dataset, "dataset") else 0))
        self.test_sample_count = int(getattr(self.val_dataset, "sample_count", len(getattr(self.val_dataset, "dataset", [])) if hasattr(self.val_dataset, "dataset") else 0))
        self.train_modulation_counts = dict(getattr(self.dataset, "modulation_counts", {}))
        self.test_modulation_counts = dict(getattr(self.val_dataset, "modulation_counts", {}))
        self.metrics_csv_path = str(
            getattr(self.params, "training_metrics_csv", os.path.join(self.model_dir, "training_convergence.csv"))
        )
        if self.val_dataset is None:
            raise ValueError("A test/validation dataloader is mandatory in this training configuration.")

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'iter': self.iter,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'params': dict(self.params),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if 'lr_scheduler' in state_dict:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.iter = state_dict['iter']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.iter}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.lexists(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            ckpt_path = f'{self.model_dir}/{filename}.pt'
            try:
                checkpoint = torch.load(ckpt_path, weights_only=True)
            except TypeError:
                checkpoint = torch.load(ckpt_path)
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def _build_bits_cond(self, bits: np.ndarray, mod: str, sps: int, N: int) -> np.ndarray:
        bits = np.asarray(bits).reshape(-1)
        bits = (bits != 0).astype(np.float32)
        k = self._bits_per_symbol(mod)
        M = self._mod_order(mod)
        sym_idx = self._bits_to_symbol_index(bits, k)
        if sym_idx.size == 0:
            bits_cond = np.zeros((N,), dtype=np.float32)
        else:
            if M > 1:
                sym_idx = sym_idx / float(M - 1)
            bits_cond = np.repeat(sym_idx, max(1, int(sps))).astype(np.float32)
            if bits_cond.size < N:
                bits_cond = np.pad(bits_cond, (0, N - bits_cond.size), mode="constant")
            elif bits_cond.size > N:
                bits_cond = bits_cond[:N]
        return bits_cond

    def _prepare_model_input_ri(self, x_c: np.ndarray, N: int, device):
        x = x_c.reshape(-1)
        if x.size < N:
            x = np.pad(x, (0, N - x.size), mode="constant")
        elif x.size > N:
            x = x[:N]
        x_t = torch.from_numpy(x.astype(np.complex64)).to(device).view(N, 1)
        x_ri = torch.view_as_real(x_t).to(torch.float32)  # [N,1,2]
        return x_ri


    def train(self, max_iter=None):
        device = next(self.model.parameters()).device
        stop_training = False

        while True:  # epoch

            epoch_idx = self.iter // len(self.dataset)

            epoch_loss_sum = 0.0
            epoch_loss_count = 0

            iterator = tqdm(self.dataset, desc=f"Epoch {epoch_idx}") if self.is_master else self.dataset

            for features in iterator:
                if max_iter is not None and self.iter >= max_iter:
                    stop_training = True
                    break

                features = _nested_map(
                    features,
                    lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
                )

                loss = self.train_iter(features)

                # -------- loss value extraction --------
                try:
                    loss_val = float(loss.item()) if hasattr(loss, "item") else float(loss)
                except Exception:
                    loss_val = None

                if loss_val is not None:
                    epoch_loss_sum += loss_val
                    epoch_loss_count += 1

                if loss_val is not None and math.isnan(loss_val):
                    raise RuntimeError(
                        f"Detected NaN loss at iteration {self.iter}."
                    )

                # -------- periodic summaries --------
                if self.is_master:
                    if self.iter % 50 == 0 and loss_val is not None:
                        self._write_summary(self.iter, features, loss)

                self.iter += 1

            # ===== END OF EPOCH =====
            if epoch_loss_count > 0:
                epoch_loss_mean = epoch_loss_sum / epoch_loss_count
            else:
                epoch_loss_mean = float("nan")

            val_loss = float("nan")
            evm_by_mod = {}

            if self.is_master:
                # ---- checkpoint once per epoch ----
                self.save_to_checkpoint()

                val_loss, evm_by_mod = self._evaluate_reverse_diffusion()

                self.epoch_train_losses.append(float(epoch_loss_mean))
                self.epoch_test_losses.append(float(val_loss))

                self._write_convergence_csv(
                    epoch_idx,
                    epoch_loss_mean,
                    val_loss,
                    evm_by_mod
                )

                evm_summary = " ".join(
                    [
                        f"{mod}_evm={evm_by_mod[mod]:.4f}"
                        for mod in sorted(evm_by_mod.keys())
                    ]
                )

                tqdm.write(
                    f"\n=== Epoch {epoch_idx} complete === "
                    f"train_loss={epoch_loss_mean:.6f} "
                    f"test_loss={val_loss:.6f} "
                    f"{evm_summary}"
                )

            self.epoch_history.append(int(epoch_idx))

            # ---- scheduler step ----
            step_metric = val_loss if self.is_master else epoch_loss_mean
            self.lr_scheduler.step(step_metric)

            if self.is_master and bool(
                getattr(self.params, "animate_after_training", False)
            ):
                self._save_epoch_snapshot(epoch_idx)

            if stop_training:
                break

        self._generate_training_animation()

        return

    def train_iter(self, features):
        self.optimizer.zero_grad()

        data = features['data']
        prompts = features['prompt']
        bits_cond = features.get('bits_cond', features.get('bits', None))

        # CVAE latent
        latent = self.encoder(data)         # [B,1024]

        B = latent.shape[0]

        # Convert latent vector into Conv1d latent representation
        latent = latent.view(B, 4, -1)      # [B,4,256]

        t = torch.randint(
            0,
            self.diffusion.max_step,
            [B],
            dtype=torch.int64,
            device=data.device
        )

        degrade_data, noise = self.diffusion.degrade_fn(latent, t)

        cond = {
            'prompt': prompts,
            'bits': bits_cond
        }

        tokens = self.tokenizer.batch_encode_plus(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )["input_ids"].to(data.device)

        context = self.clip(tokens)

        cond["prompt"] = context

        predicted_noise = self.model(degrade_data, t, cond)

        loss = torch.mean(
            torch.abs(predicted_noise - noise) ** 2
        )

        loss.backward()

        grad_norm_sq = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm_sq += param.grad.detach().norm(2).item() ** 2

        self.grad_norm = grad_norm_sq ** 0.5

        self.optimizer.step()

        return loss.item()


    def _write_summary(self, iter, features, loss):
        writer = self.summary_writer or SummaryWriter(self.log_dir, purge_step=iter)
        # writer.add_scalars('feature/csi', features['csi'][0].abs(), step)
        # writer.add_image('feature/stft', features['stft'][0].abs(), step)
        writer.add_scalar('train/loss', loss, iter)
        writer.add_scalar('train/grad_norm', self.grad_norm, iter)
        writer.flush()
        self.summary_writer = writer
