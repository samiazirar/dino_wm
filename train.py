import os
import signal
import subprocess
import time
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from metrics.image_metrics import eval_images
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
from training_resume import (
    CHECKPOINT_SCHEMA,
    PROGRESS_SCHEMA,
    SerializableConstantScheduler,
    StepBatchSampler,
    StepCheckpointManager,
    atomic_write_json,
    capture_rng_state,
    configure_strict_determinism,
    dataset_order_sha256,
    exact_key_check,
    json_sha256,
    nested_state_sha256,
    parameter_sha256,
    restore_rng_state,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def target_epoch_range(completed_epoch: int, target_total_epochs: int):
    """Return the remaining 1-indexed epochs for target-total resume semantics."""
    if completed_epoch < 0:
        raise ValueError("completed_epoch must be non-negative")
    if target_total_epochs < 0:
        raise ValueError("target_total_epochs must be non-negative")
    return range(completed_epoch + 1, target_total_epochs + 1)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.step_mode = cfg.training.target_steps is not None
        self.strict_determinism = bool(cfg.training.strict_determinism)
        configure_strict_determinism(self.strict_determinism)
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            # # ==== /init ddp process group ====

        self.accelerator = Accelerator(log_with="wandb")
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0
        self.global_step = 0
        self.last_step_loss = None

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )
        if self.step_mode and self.accelerator.num_processes != 1:
            raise RuntimeError(
                "Deterministic step resume currently requires exactly one process. "
                "P3 cells are specified as one A100 per cell."
            )
        if (
            self.step_mode
            and self.strict_determinism
            and int(self.cfg.env.num_workers) != 0
        ):
            raise RuntimeError(
                "Strict step resume requires env.num_workers=0 so Python, NumPy, "
                "and Torch RNG state lives only in the checkpointed training process"
            )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project="dino_wm_debug",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                self.wandb_run = wandb.init(
                    project="dino_wm",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        phases = ["valid"] if self.step_mode else ["train", "valid"]
        self.dataloaders = {
            phase: torch.utils.data.DataLoader(
                self.datasets[phase],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False, # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
            )
            for phase in phases
        }

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        if self.step_mode:
            self.dataloaders["valid"] = self.accelerator.prepare(
                self.dataloaders["valid"]
            )
        else:
            self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
                self.dataloaders["train"], self.dataloaders["valid"]
            )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        self._keys_to_save = [
            "epoch",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        self._keys_to_save += (
            ["decoder", "decoder_optimizer"] if self.train_decoder else []
        )
        self._keys_to_save += ["action_encoder", "proprio_encoder"]

        self.init_models()
        self.init_optimizers()
        self.init_schedulers()

        self._resume_rng_state = None
        self._stop_requested = False
        self._stop_signal = None
        self._last_saved_step = None
        self._last_checkpoint_path = None
        self._last_checkpoint_sha256 = None
        if self.step_mode:
            self._initialize_step_resume()

        self.epoch_log = OrderedDict()

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            for k in self._keys_to_save:
                if hasattr(self.__dict__[k], "module"):
                    ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    ckpt[k] = self.__dict__[k]
            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        ckpt = torch.load(filename)
        for k, v in ckpt.items():
            self.__dict__[k] = v
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)

    def init_models(self):
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if not self.step_mode and model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

        # initialize encoder
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proprio_encoder = hydra.utils.instantiate(
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.cfg.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        self.action_encoder = hydra.utils.instantiate(
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.cfg.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # Token geometry is an encoder contract. It must not be inferred from
        # the world-model image size (DINOv2 has 196 tokens; DFormerv2 has 49).
        encoder_metadata = self.accelerator.unwrap_model(self.encoder)
        if not hasattr(encoder_metadata, "num_patches"):
            raise AttributeError(
                f"{type(encoder_metadata).__name__} must declare num_patches"
            )
        num_patches = int(encoder_metadata.num_patches)
        if num_patches <= 0:
            raise ValueError(f"Invalid encoder num_patches: {num_patches}")

        if self.cfg.concat_dim == 0:
            num_patches += 2

        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=self.encoder.emb_dim
                    + (
                        proprio_emb_dim * self.cfg.num_proprio_repeat
                        + action_emb_dim * self.cfg.num_action_repeat
                    )
                    * (self.cfg.concat_dim),
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,  # 384
                    )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
        self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder
        )
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        )

    def init_optimizers(self):
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.training.encoder_lr,
        )
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        if self.cfg.has_predictor:
            self.predictor_optimizer = torch.optim.AdamW(
                self.predictor.parameters(),
                lr=self.cfg.training.predictor_lr,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(), self.proprio_encoder.parameters()
                ),
                lr=self.cfg.training.action_encoder_lr,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

        if self.cfg.has_decoder:
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=self.cfg.training.decoder_lr
            )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)

    def _model_components(self):
        components = {
            "encoder": self.encoder,
            "action_encoder": self.action_encoder,
            "proprio_encoder": self.proprio_encoder,
        }
        if self.predictor is not None:
            components["predictor"] = self.predictor
        if self.decoder is not None:
            components["decoder"] = self.decoder
        return {
            name: self.accelerator.unwrap_model(component)
            for name, component in components.items()
        }

    def _optimizers(self):
        optimizers = {"encoder": self.encoder_optimizer}
        if self.cfg.has_predictor:
            optimizers["predictor"] = self.predictor_optimizer
            optimizers["action_encoder"] = self.action_encoder_optimizer
        if self.cfg.has_decoder:
            optimizers["decoder"] = self.decoder_optimizer
        return optimizers

    def init_schedulers(self):
        self.schedulers = {
            name: SerializableConstantScheduler(optimizer)
            for name, optimizer in self._optimizers().items()
        }

    def _semantic_resume_config(self):
        config = OmegaConf.to_container(self.cfg, resolve=True)
        for key in [
            "hydra",
            "ckpt_base_path",
            "saved_folder",
            "wandb_run_id",
            "effective_batch_size",
            "gpu_batch_size",
        ]:
            config.pop(key, None)
        training = config["training"]
        for key in [
            "target_steps",
            "segment_steps",
            "resume_from",
            "checkpoint_every_steps",
            "test_signal_after_step",
        ]:
            training.pop(key, None)
        return config

    def _sampler_state(self):
        sampler = StepBatchSampler(
            dataset_size=len(self.datasets["train"]),
            batch_size=int(self.cfg.gpu_batch_size),
            start_step=self.global_step,
            stop_step=self.global_step,
        )
        state = sampler.state_dict(self.global_step)
        state["dataset_order_sha256"] = self.dataset_order_sha256
        return state

    def _initialize_step_resume(self):
        if int(self.cfg.training.target_steps) <= 0:
            raise ValueError("training.target_steps must be positive")
        if self.cfg.training.segment_steps is not None and int(
            self.cfg.training.segment_steps
        ) <= 0:
            raise ValueError("training.segment_steps must be positive when set")
        if int(self.cfg.training.checkpoint_every_steps) < 0:
            raise ValueError("training.checkpoint_every_steps must be non-negative")

        self.dataset_order_sha256 = dataset_order_sha256(self.datasets["train"])
        self.resume_config_sha256 = json_sha256(self._semantic_resume_config())
        self.source_commit = subprocess.check_output(
            ["git", "-C", self.base_path, "rev-parse", "HEAD"], text=True
        ).strip()
        self.checkpoint_manager = StepCheckpointManager(
            Path(self.cfg.saved_folder) / "checkpoints" / "steps"
        )
        requested = self.cfg.training.resume_from
        checkpoint_path, checkpoint_digest = self.checkpoint_manager.resolve(requested)
        if checkpoint_path is not None:
            self._load_step_checkpoint(checkpoint_path, checkpoint_digest)
        self._install_signal_handlers()

    def _install_signal_handlers(self):
        def request_checkpoint(signum, _frame):
            self._stop_requested = True
            try:
                self._stop_signal = signal.Signals(signum).name
            except ValueError:
                self._stop_signal = str(signum)

        signal.signal(signal.SIGUSR1, request_checkpoint)
        signal.signal(signal.SIGTERM, request_checkpoint)

    def _load_step_checkpoint(self, path, digest):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError(f"Unknown step checkpoint schema in {path}")
        if checkpoint.get("source_commit") != self.source_commit:
            raise RuntimeError(
                "Source commit differs between checkpoint and resume: "
                f"{checkpoint.get('source_commit')} versus {self.source_commit}"
            )
        if checkpoint.get("resume_config_sha256") != self.resume_config_sha256:
            raise RuntimeError("Semantic training configuration differs from checkpoint")
        if checkpoint.get("dataset_order_sha256") != self.dataset_order_sha256:
            raise RuntimeError("Dataset order differs from checkpoint")

        components = self._model_components()
        exact_key_check("model component", components, checkpoint["models"])
        for name, component in components.items():
            component.load_state_dict(checkpoint["models"][name], strict=True)

        optimizers = self._optimizers()
        exact_key_check("optimizer", optimizers, checkpoint["optimizers"])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][name])

        exact_key_check("scheduler", self.schedulers, checkpoint["schedulers"])
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(checkpoint["schedulers"][name])

        self.global_step = int(checkpoint["global_step"])
        self.epoch = int(checkpoint["completed_epochs"])
        self.last_step_loss = checkpoint.get("last_step_loss")
        expected_sampler = self._sampler_state()
        if checkpoint["sampler"] != expected_sampler:
            raise RuntimeError(
                f"Sampler cursor differs: {checkpoint['sampler']} versus {expected_sampler}"
            )
        current_parameter_hash = parameter_sha256(components)
        if current_parameter_hash != checkpoint["parameter_sha256"]:
            raise RuntimeError("Model parameter hash changed while loading checkpoint")
        self._resume_rng_state = checkpoint["rng"]
        self._last_saved_step = self.global_step
        self._last_checkpoint_path = Path(path)
        self._last_checkpoint_sha256 = digest
        log.info(
            "Resuming at optimizer step %d from %s", self.global_step, path
        )

    def save_step_checkpoint(self):
        if self._last_saved_step == self.global_step:
            return self._last_checkpoint_path, self._last_checkpoint_sha256
        components = self._model_components()
        optimizers = self._optimizers()
        optimizer_states = {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        }
        scheduler_states = {
            name: scheduler.state_dict() for name, scheduler in self.schedulers.items()
        }
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "source_commit": self.source_commit,
            "resume_config_sha256": self.resume_config_sha256,
            "dataset_order_sha256": self.dataset_order_sha256,
            "global_step": self.global_step,
            "completed_epochs": self.global_step
            // self._sampler_state()["steps_per_epoch"],
            "last_step_loss": self.last_step_loss,
            "models": {
                name: component.state_dict() for name, component in components.items()
            },
            "optimizers": optimizer_states,
            "schedulers": scheduler_states,
            "sampler": self._sampler_state(),
            "rng": capture_rng_state(),
            "parameter_sha256": parameter_sha256(components),
            "optimizer_sha256": nested_state_sha256(optimizer_states),
            "scheduler_sha256": nested_state_sha256(scheduler_states),
        }
        path, digest = self.checkpoint_manager.save(payload, self.global_step)
        self._last_saved_step = self.global_step
        self._last_checkpoint_path = path
        self._last_checkpoint_sha256 = digest
        log.info("Saved deterministic step checkpoint %s", path)
        return path, digest

    def _write_step_progress(self, status, segment_start, segment_stop, elapsed_seconds):
        components = self._model_components()
        progress = {
            "schema": PROGRESS_SCHEMA,
            "status": status,
            "source_commit": self.source_commit,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "global_step": self.global_step,
            "target_steps": int(self.cfg.training.target_steps),
            "segment_start_step": segment_start,
            "segment_stop_step": segment_stop,
            "completed_segment_steps": self.global_step - segment_start,
            "completed_epochs": self.epoch,
            "last_step_loss": self.last_step_loss,
            "parameter_sha256": parameter_sha256(components),
            "optimizer_sha256": nested_state_sha256(
                {
                    name: optimizer.state_dict()
                    for name, optimizer in self._optimizers().items()
                }
            ),
            "scheduler_sha256": nested_state_sha256(
                {
                    name: scheduler.state_dict()
                    for name, scheduler in self.schedulers.items()
                }
            ),
            "sampler": self._sampler_state(),
            "checkpoint": str(self._last_checkpoint_path),
            "checkpoint_sha256": self._last_checkpoint_sha256,
            "stop_signal": self._stop_signal,
            "elapsed_seconds": elapsed_seconds,
        }
        atomic_write_json(Path(self.cfg.saved_folder) / "progress.json", progress)
        print(
            "STEP_PROGRESS "
            f"status={status} step={self.global_step} "
            f"loss={self.last_step_loss} hash={progress['parameter_sha256']}"
        )

    def _train_one_step(self, data):
        obs, act, _state = data
        self.model.train()
        self.encoder_optimizer.zero_grad()
        if self.cfg.has_decoder:
            self.decoder_optimizer.zero_grad()
        if self.cfg.has_predictor:
            self.predictor_optimizer.zero_grad()
            self.action_encoder_optimizer.zero_grad()

        _z_out, _visual_out, _visual_reconstructed, loss, _loss_components = self.model(
            obs, act
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Nonfinite loss at optimizer step {self.global_step}: {loss.item()}"
            )
        self.accelerator.backward(loss)

        if self.model.train_encoder:
            self.encoder_optimizer.step()
            self.schedulers["encoder"].step()
        if self.cfg.has_decoder and self.model.train_decoder:
            self.decoder_optimizer.step()
            self.schedulers["decoder"].step()
        if self.cfg.has_predictor and self.model.train_predictor:
            self.predictor_optimizer.step()
            self.action_encoder_optimizer.step()
            self.schedulers["predictor"].step()
            self.schedulers["action_encoder"].step()

        gathered_loss = self.accelerator.gather_for_metrics(loss.detach()).mean()
        return float(gathered_loss.cpu().item())

    def run_steps(self):
        target_steps = int(self.cfg.training.target_steps)
        if self.global_step > target_steps:
            raise RuntimeError(
                f"Checkpoint step {self.global_step} exceeds target {target_steps}"
            )
        segment_start = self.global_step
        configured_segment = self.cfg.training.segment_steps
        segment_stop = target_steps
        if configured_segment is not None:
            segment_stop = min(
                target_steps, segment_start + int(configured_segment)
            )
        sampler = StepBatchSampler(
            dataset_size=len(self.datasets["train"]),
            batch_size=int(self.cfg.gpu_batch_size),
            start_step=segment_start,
            stop_step=segment_stop,
        )
        generator = torch.Generator().manual_seed(int(self.cfg.training.seed))
        loader = torch.utils.data.DataLoader(
            self.datasets["train"],
            batch_sampler=sampler,
            num_workers=int(self.cfg.env.num_workers),
            collate_fn=None,
            generator=generator,
        )
        loader = self.accelerator.prepare(loader)
        iterator = iter(loader)
        if self._resume_rng_state is not None:
            restore_rng_state(self._resume_rng_state)
            self._resume_rng_state = None

        started = time.perf_counter()
        checkpoint_every = int(self.cfg.training.checkpoint_every_steps)
        test_signal_step = self.cfg.training.test_signal_after_step
        for data in tqdm(
            iterator,
            total=len(sampler),
            desc=f"Steps {segment_start + 1}-{segment_stop}",
        ):
            if self._stop_requested:
                break
            self.last_step_loss = self._train_one_step(data)
            self.global_step += 1
            self.epoch = self.global_step // sampler.steps_per_epoch

            if test_signal_step is not None and self.global_step == int(test_signal_step):
                os.kill(os.getpid(), signal.SIGUSR1)
            checkpoint_due = (
                checkpoint_every > 0 and self.global_step % checkpoint_every == 0
            )
            if checkpoint_due or self._stop_requested:
                self.save_step_checkpoint()
            if self._stop_requested:
                break

        self.save_step_checkpoint()
        if self.global_step == target_steps:
            status = "TARGET_REACHED"
        elif self._stop_requested:
            status = "SIGNAL_CHECKPOINTED"
        elif self.global_step == segment_stop:
            status = "SEGMENT_COMPLETE"
        else:
            raise RuntimeError(
                f"Step loader stopped at {self.global_step}, expected {segment_stop}"
            )
        self._write_step_progress(
            status,
            segment_start,
            segment_stop,
            time.perf_counter() - started,
        )

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if self.step_mode:
            self.run_steps()
            return
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        # training.epochs is a target-total epoch, including completed epochs.
        # A checkpoint at epoch 2 resumed with epochs=3 therefore runs only 3.
        for epoch in target_epoch_range(self.epoch, self.total_epochs):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                # main thread only: launch planning jobs on the saved ckpt
                if (
                    self.cfg.plan_settings.plan_cfg_path is not None
                    and ckpt_path is not None
                ):  # ckpt_path is only not None for main process
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data
            plot = i == 0  # only plot from the first batch
            self.model.train()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if self.cfg.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }
            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def val(self):
        self.model.eval()
        if len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            with torch.no_grad():
                train_rollout_logs = self.openloop_rollout(
                    self.train_traj_dset, mode="train"
                )
                train_rollout_logs = {
                    f"train_{k}": [v] for k, v in train_rollout_logs.items()
                }
                self.logs_update(train_rollout_logs)
                val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
                val_rollout_logs = {
                    f"val_{k}": [v] for k, v in val_rollout_logs.items()
                }
                self.logs_update(val_rollout_logs)

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )
            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start : 
                    start + horizon * self.cfg.frameskip + 1 : 
                    self.cfg.frameskip
                ]
            act = act[start : start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["epoch"] = step
        log.info(f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                Validation loss: {epoch_log['val_loss']:.4f}")

        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
