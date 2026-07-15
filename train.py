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
import json
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
from training_timing import StrictTimingWindow
from p3_completion import (
    FINAL_RECEIPT_SCHEMA,
    TRAINING_RECORD_SCHEMA,
    VALIDATION_RECORD_SCHEMA,
    P3CompletionError,
    append_training_record,
    append_validation_record,
    checkpoint_reference,
    initialize_training_tail_index,
    load_checkpoint_history,
    load_jsonl,
    percent_step_map,
    percents_at_step,
    require_job_id,
    runtime_slice_entries,
    sha256_file as completion_sha256_file,
    validate_checkpoint_evidence_bindings,
    validate_runtime_heldout_manifest,
    validate_training_records,
    validate_validation_records,
    write_final_receipt,
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
        self.p3_completion_enabled = bool(cfg.training.p3_completion_enabled)
        self.final_acceptance = bool(cfg.training.final_acceptance)
        if self.final_acceptance and not self.p3_completion_enabled:
            raise RuntimeError("final acceptance requires the P3 completion contract")
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
        self.source_commit = subprocess.check_output(
            ["git", "-C", self.base_path, "rev-parse", "HEAD"], text=True
        ).strip()

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

        self.p3_heldout_rows = None
        self.p3_heldout_metadata = None
        self.p3_validation_indices = None
        if self.p3_completion_enabled:
            self._initialize_p3_heldout_manifest()

        phases = ["valid"] if self.step_mode else ["train", "valid"]
        self.dataloaders = {
            phase: torch.utils.data.DataLoader(
                self.datasets[phase],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False,  # already shuffled in TrajSlicerDataset
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
            self.dataloaders["train"], self.dataloaders["valid"] = (
                self.accelerator.prepare(
                    self.dataloaders["train"], self.dataloaders["valid"]
                )
            )
        if self.p3_completion_enabled:
            heldout_subset = torch.utils.data.Subset(
                self.datasets["valid"], self.p3_validation_indices
            )
            heldout_loader = torch.utils.data.DataLoader(
                heldout_subset,
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=None,
            )
            self.p3_validation_loader = self.accelerator.prepare(heldout_loader)

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(
            f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}"
        )

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
        self._last_checkpoint_history_record = None
        self._last_checkpoint_state_hashes = None
        self._loaded_checkpoint_metadata = None
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

    def _initialize_p3_heldout_manifest(self):
        required = {
            "path": self.cfg.training.p3_heldout_manifest,
            "sha256": self.cfg.training.p3_heldout_manifest_sha256,
            "metadata_path": self.cfg.training.p3_heldout_metadata,
            "metadata_sha256": self.cfg.training.p3_heldout_metadata_sha256,
            "data_manifest_sha256": self.cfg.training.p3_data_manifest_sha256,
            "split_sha256": self.cfg.training.p3_split_sha256,
        }
        if any(
            value is None or str(value).lower() in {"", "none", "null"}
            for value in required.values()
        ):
            raise RuntimeError(
                "P3 completion requires a complete held-out manifest record"
            )
        environment = str(self.cfg.env.name)
        if environment == "deformable_env":
            environment = str(self.cfg.env.dataset.object_name)
        training_entries = runtime_slice_entries(
            self.datasets["train"], environment=environment, partition="train"
        )
        validation_entries = runtime_slice_entries(
            self.datasets["valid"], environment=environment, partition="valid"
        )
        try:
            rows, indices, metadata = validate_runtime_heldout_manifest(
                required,
                environment=environment,
                source_commit=self.source_commit,
                training_entries=training_entries,
                validation_entries=validation_entries,
            )
        except P3CompletionError as exc:
            raise RuntimeError(str(exc)) from exc
        self.p3_heldout_rows = rows
        self.p3_validation_indices = indices
        self.p3_heldout_metadata = metadata

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
            "timing_output",
            "timing_run_card",
            "timing_warmup_steps",
            "timing_measured_steps",
            "timing_projection_target_steps",
            "final_acceptance",
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
        if (
            self.cfg.training.segment_steps is not None
            and int(self.cfg.training.segment_steps) <= 0
        ):
            raise ValueError("training.segment_steps must be positive when set")
        if int(self.cfg.training.checkpoint_every_steps) < 0:
            raise ValueError("training.checkpoint_every_steps must be non-negative")

        self.dataset_order_sha256 = dataset_order_sha256(self.datasets["train"])
        self.resume_config_sha256 = json_sha256(self._semantic_resume_config())
        self.immutable_run_card_sha256 = os.environ.get(
            "STRICT_P2_IMMUTABLE_RUN_CARD_SHA256"
        )
        if self.immutable_run_card_sha256 is not None and (
            len(self.immutable_run_card_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.immutable_run_card_sha256
            )
        ):
            raise RuntimeError("immutable run-card SHA-256 environment is invalid")
        self.checkpoint_manager = StepCheckpointManager(
            Path(self.cfg.saved_folder) / "checkpoints" / "steps"
        )
        requested = self.cfg.training.resume_from
        checkpoint_path, checkpoint_digest = self.checkpoint_manager.resolve(requested)
        if checkpoint_path is not None:
            self._load_step_checkpoint(checkpoint_path, checkpoint_digest)
        if self.p3_completion_enabled:
            self._initialize_p3_completion_evidence()
        self._install_signal_handlers()

    def _initialize_p3_completion_evidence(self):
        if self.immutable_run_card_sha256 is None:
            raise RuntimeError("P3 completion requires an immutable run-card SHA-256")
        run_card_path = os.environ.get("STRICT_P2_IMMUTABLE_RUN_CARD")
        if not run_card_path:
            raise RuntimeError("P3 completion requires the immutable run-card path")
        run_card_path = Path(run_card_path).resolve()
        run_card = __import__("yaml").safe_load(
            run_card_path.read_text(encoding="utf-8")
        )
        if (
            not isinstance(run_card, dict)
            or run_card.get("kind") != "p3-training"
            or run_card.get("run_card_sha256") != self.immutable_run_card_sha256
            or run_card.get("source_commit") != self.source_commit
            or run_card.get("heldout_loss_manifest", {}).get("sha256")
            != str(self.cfg.training.p3_heldout_manifest_sha256)
        ):
            raise RuntimeError("P3 completion run card differs from the live process")
        container_sha256 = os.environ.get("STRICT_P2_CONTAINER_SHA256")
        if run_card.get("container", {}).get("sha256") != container_sha256:
            raise RuntimeError("P3 completion container differs from the run card")
        self.p3_run_card = run_card
        self.p3_training_ledger_path = (
            Path(self.cfg.saved_folder) / "training_steps.jsonl"
        )
        self.p3_validation_ledger_path = (
            Path(self.cfg.saved_folder) / "heldout_loss.jsonl"
        )
        training_rows = load_jsonl(self.p3_training_ledger_path, allow_missing=True)
        validation_rows = load_jsonl(self.p3_validation_ledger_path, allow_missing=True)
        try:
            validate_training_records(
                training_rows,
                source_commit=self.source_commit,
                immutable_run_card_sha256=self.immutable_run_card_sha256,
                dataset_order_sha256=self.dataset_order_sha256,
                target_steps=int(self.cfg.training.target_steps),
                expected_dataset_size=len(self.datasets["train"]),
                expected_batch_size=int(self.cfg.gpu_batch_size),
            )
            validate_validation_records(
                validation_rows,
                target_steps=int(self.cfg.training.target_steps),
                immutable_run_card_sha256=self.immutable_run_card_sha256,
                manifest_sha256=str(self.cfg.training.p3_heldout_manifest_sha256),
            )
            validate_checkpoint_evidence_bindings(
                load_checkpoint_history(self.checkpoint_manager.history_path),
                directory=self.checkpoint_manager.directory,
                source_commit=self.source_commit,
                immutable_run_card_sha256=self.immutable_run_card_sha256,
                dataset_order_sha256=self.dataset_order_sha256,
                training_rows=training_rows,
                validation_rows=validation_rows,
            )
            training_tail_index = initialize_training_tail_index(
                self.p3_training_ledger_path,
                training_rows,
                source_commit=self.source_commit,
                immutable_run_card_sha256=self.immutable_run_card_sha256,
                dataset_order_sha256=self.dataset_order_sha256,
                config_sha256=self.p3_run_card["config_sha256"],
                dataset_size=len(self.datasets["train"]),
                batch_size=int(self.cfg.gpu_batch_size),
                target_steps=int(self.cfg.training.target_steps),
            )
        except P3CompletionError as exc:
            raise RuntimeError(str(exc)) from exc
        self.p3_training_next_step = int(training_tail_index["next_step"])
        self.p3_training_rows_for_final = (
            training_rows if self.final_acceptance else None
        )

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
            raise RuntimeError(
                "Semantic training configuration differs from checkpoint"
            )
        if (
            checkpoint.get("immutable_run_card_sha256")
            != self.immutable_run_card_sha256
        ):
            raise RuntimeError("Immutable run card differs from checkpoint")
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
        self._last_checkpoint_history_record = self.checkpoint_manager.history_record(
            self.global_step
        )
        self._last_checkpoint_state_hashes = {
            "parameter_sha256": checkpoint["parameter_sha256"],
            "optimizer_sha256": checkpoint["optimizer_sha256"],
            "scheduler_sha256": checkpoint["scheduler_sha256"],
        }
        self._loaded_checkpoint_metadata = checkpoint
        log.info("Resuming at optimizer step %d from %s", self.global_step, path)

    def save_step_checkpoint(self, reasons):
        if self._last_saved_step == self.global_step:
            self._last_checkpoint_history_record = (
                self.checkpoint_manager.enrich_reasons(self.global_step, reasons)
            )
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
            "immutable_run_card_sha256": self.immutable_run_card_sha256,
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
        path, digest = self.checkpoint_manager.save(
            payload, self.global_step, reasons=reasons
        )
        self._last_saved_step = self.global_step
        self._last_checkpoint_path = path
        self._last_checkpoint_sha256 = digest
        self._last_checkpoint_history_record = (
            self.checkpoint_manager.last_history_record
        )
        self._last_checkpoint_state_hashes = {
            "parameter_sha256": payload["parameter_sha256"],
            "optimizer_sha256": payload["optimizer_sha256"],
            "scheduler_sha256": payload["scheduler_sha256"],
        }
        self._loaded_checkpoint_metadata = payload
        log.info(
            "Saved deterministic step checkpoint %s reasons=%s",
            path,
            sorted(set(reasons)),
        )
        return path, digest

    def _write_step_progress(
        self, status, segment_start, segment_stop, elapsed_seconds
    ):
        components = self._model_components()
        progress = {
            "schema": PROGRESS_SCHEMA,
            "status": status,
            "source_commit": self.source_commit,
            "immutable_run_card_sha256": self.immutable_run_card_sha256,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "training_process_id": os.getpid(),
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
        if self.p3_completion_enabled:
            progress["p3_completion"] = {
                "heldout_manifest_sha256": str(
                    self.cfg.training.p3_heldout_manifest_sha256
                ),
                "data_manifest_sha256": str(self.cfg.training.p3_data_manifest_sha256),
                "split_sha256": str(self.cfg.training.p3_split_sha256),
                "training_ledger": str(self.p3_training_ledger_path),
                "training_ledger_sha256": completion_sha256_file(
                    self.p3_training_ledger_path
                ),
                "validation_ledger": str(self.p3_validation_ledger_path),
                "validation_ledger_sha256": completion_sha256_file(
                    self.p3_validation_ledger_path
                ),
                "checkpoint_history": str(self.checkpoint_manager.history_path),
                "checkpoint_history_sha256": completion_sha256_file(
                    self.checkpoint_manager.history_path
                ),
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

    def _p3_checkpoint_reference(self):
        if self._last_checkpoint_history_record is None:
            raise RuntimeError("P3 step record has no checkpoint history reference")
        return checkpoint_reference(
            self._last_checkpoint_history_record,
            self.checkpoint_manager.directory,
        )

    def _p3_append_training_record(self):
        checkpoint = self._p3_checkpoint_reference()
        hashes = (
            self._last_checkpoint_state_hashes
            if int(checkpoint["step"]) == self.global_step
            else {}
        )
        record = {
            "schema": TRAINING_RECORD_SCHEMA,
            "source_commit": self.source_commit,
            "immutable_run_card_sha256": self.immutable_run_card_sha256,
            "dataset_order_sha256": self.dataset_order_sha256,
            "config_sha256": self.p3_run_card["config_sha256"],
            "global_step": self.global_step,
            "completed_epochs": self.epoch,
            "sampler": self._sampler_state(),
            "loss": self.last_step_loss,
            "parameter_sha256": hashes.get("parameter_sha256"),
            "optimizer_sha256": hashes.get("optimizer_sha256"),
            "scheduler_sha256": hashes.get("scheduler_sha256"),
            "slurm_job_id": require_job_id(os.environ.get("SLURM_JOB_ID")),
            "checkpoint": checkpoint,
        }
        try:
            appended = append_training_record(
                self.p3_training_ledger_path,
                record,
                target_steps=int(self.cfg.training.target_steps),
            )
        except P3CompletionError as exc:
            raise RuntimeError(str(exc)) from exc
        self.p3_training_next_step = int(appended["global_step"]) + 1

    def _p3_validation_loss(self, *, maximum_batches=None):
        module_training_modes = [
            (module, bool(module.training)) for module in self.model.modules()
        ]
        rng_state = capture_rng_state()
        before = {
            "parameter_sha256": parameter_sha256(self._model_components()),
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
            "rng_sha256": nested_state_sha256(rng_state),
        }
        numerator = 0.0
        element_count = 0
        try:
            self.model.eval()
            with torch.no_grad():
                for batch_index, data in enumerate(self.p3_validation_loader):
                    if maximum_batches is not None and batch_index >= maximum_batches:
                        break
                    obs, act, _state = data
                    z_pred, _visual, _reconstructed, loss, _components = self.model(
                        obs, act
                    )
                    if z_pred is None or not torch.isfinite(loss):
                        raise FloatingPointError(
                            "held-out validation produced nonfinite loss"
                        )
                    if self.model.concat_dim == 0:
                        count = int(z_pred[:, :, :-1, :].numel())
                    else:
                        count = int(z_pred[..., : -self.model.action_dim].numel())
                    if count <= 0:
                        raise RuntimeError(
                            "held-out validation produced no loss elements"
                        )
                    numerator += float(loss.detach().to(torch.float64).cpu()) * count
                    element_count += count
        finally:
            for module, was_training in module_training_modes:
                module.training = was_training
            restore_rng_state(rng_state)
        if any(
            module.training is not was_training
            for module, was_training in module_training_modes
        ):
            raise RuntimeError("held-out validation did not restore module modes")
        after = {
            "parameter_sha256": parameter_sha256(self._model_components()),
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
            "rng_sha256": nested_state_sha256(capture_rng_state()),
        }
        if before != after:
            raise RuntimeError(
                "held-out validation did not restore model/optimizer/scheduler/RNG state"
            )
        mean = numerator / element_count if element_count else float("nan")
        if not np.isfinite(numerator) or not np.isfinite(mean) or element_count <= 0:
            raise FloatingPointError(
                "held-out validation aggregate is nonfinite or empty"
            )
        return numerator, element_count, mean, before

    def _p3_depth_provenance(self):
        depth = self.p3_run_card.get("depth_inputs")
        if depth is None:
            return {
                "depth_producer_sha256": None,
                "depth_cache_manifest_sha256": None,
                "depth_native_contract_sha256": None,
                "depth_validation_sha256": None,
                "depth_checkpoint_sha256": None,
            }
        return {
            "depth_producer_sha256": depth["producer_sha256"],
            "depth_cache_manifest_sha256": depth["cache_manifest_sha256"],
            "depth_native_contract_sha256": depth["native_contract_sha256"],
            "depth_validation_sha256": depth["validation_sha256"],
            "depth_checkpoint_sha256": depth["checkpoint_sha256"],
        }

    def _p3_complete_percent(self, percent):
        mapping = percent_step_map(int(self.cfg.training.target_steps))
        if mapping[int(percent)] != self.global_step:
            raise RuntimeError("held-out percent is not at its exact mapped step")
        existing = load_jsonl(self.p3_validation_ledger_path, allow_missing=True)
        prior = next((row for row in existing if row.get("percent") == percent), None)
        if prior is not None:
            if (
                prior.get("global_step") != self.global_step
                or prior.get("checkpoint_sha256") != self._last_checkpoint_sha256
            ):
                raise RuntimeError("existing held-out percent differs from checkpoint")
            return
        if self._last_saved_step != self.global_step:
            raise RuntimeError("held-out percent lacks an exact step checkpoint")
        numerator, count, mean, hashes = self._p3_validation_loss()
        metadata = self.p3_heldout_metadata
        record = {
            "schema": VALIDATION_RECORD_SCHEMA,
            "percent": int(percent),
            "global_step": self.global_step,
            "target_steps": int(self.cfg.training.target_steps),
            "rounding_rule": "ceil(target_steps*percent/100)",
            "loss_numerator": numerator,
            "element_count": count,
            "mean_loss": mean,
            "source_commit": self.source_commit,
            "config_sha256": self.p3_run_card["config_sha256"],
            "container_sha256": self.p3_run_card["container"]["sha256"],
            "model_sha256": hashes["parameter_sha256"],
            "checkpoint_sha256": self._last_checkpoint_sha256,
            "checkpoint_history_record_sha256": self._last_checkpoint_history_record[
                "record_sha256"
            ],
            "manifest_sha256": str(self.cfg.training.p3_heldout_manifest_sha256),
            "data_manifest_sha256": metadata["data_manifest_sha256"],
            "split_sha256": metadata["split_sha256"],
            "immutable_run_card_sha256": self.immutable_run_card_sha256,
            "slurm_job_id": require_job_id(os.environ.get("SLURM_JOB_ID")),
            "state_restored": True,
            **self._p3_depth_provenance(),
        }
        try:
            append_validation_record(
                self.p3_validation_ledger_path,
                record,
                target_steps=int(self.cfg.training.target_steps),
            )
        except P3CompletionError as exc:
            raise RuntimeError(str(exc)) from exc

    def _p3_reconcile_loaded_step(self):
        recorded_steps = self.p3_training_next_step - 1
        if recorded_steps < max(0, self.global_step - 1):
            raise RuntimeError("training ledger is too short for the loaded checkpoint")
        if recorded_steps > self.global_step:
            raise RuntimeError("training ledger is ahead of the loaded checkpoint")
        if self.global_step > 0 and recorded_steps == self.global_step - 1:
            self._p3_append_training_record()
        for percent in percents_at_step(
            int(self.cfg.training.target_steps), self.global_step
        ):
            self._p3_complete_percent(percent)

    def _run_p3_final_acceptance(self):
        if os.environ.get("P3_FINAL_ACCEPTANCE_PROCESS") != "1":
            raise RuntimeError("final acceptance must run in the fresh wrapper process")
        target_steps = int(self.cfg.training.target_steps)
        if self.global_step != target_steps or self._loaded_checkpoint_metadata is None:
            raise RuntimeError(
                "final acceptance did not load the exact target checkpoint"
            )
        training_rows = self.p3_training_rows_for_final
        if training_rows is None:
            training_rows = load_jsonl(self.p3_training_ledger_path)
        validation_rows = load_jsonl(self.p3_validation_ledger_path)
        validate_training_records(
            training_rows,
            source_commit=self.source_commit,
            immutable_run_card_sha256=self.immutable_run_card_sha256,
            dataset_order_sha256=self.dataset_order_sha256,
            target_steps=target_steps,
            expected_dataset_size=len(self.datasets["train"]),
            expected_batch_size=int(self.cfg.gpu_batch_size),
            require_complete=True,
        )
        validate_validation_records(
            validation_rows,
            target_steps=target_steps,
            immutable_run_card_sha256=self.immutable_run_card_sha256,
            manifest_sha256=str(self.cfg.training.p3_heldout_manifest_sha256),
            require_complete=True,
        )
        progress_path = Path(self.cfg.saved_folder) / "progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        checkpoint = self._loaded_checkpoint_metadata
        training_process_id = progress.get("training_process_id")
        if (
            progress.get("status") != "TARGET_REACHED"
            or progress.get("global_step") != target_steps
            or progress.get("checkpoint_sha256") != self._last_checkpoint_sha256
            or progress.get("parameter_sha256") != checkpoint["parameter_sha256"]
            or progress.get("optimizer_sha256") != checkpoint["optimizer_sha256"]
            or progress.get("scheduler_sha256") != checkpoint["scheduler_sha256"]
            or checkpoint.get("sampler") != self._sampler_state()
            or not isinstance(training_process_id, int)
            or isinstance(training_process_id, bool)
            or training_process_id <= 0
            or training_process_id == os.getpid()
        ):
            raise RuntimeError(
                "final checkpoint metadata differs from progress/runtime"
            )
        history = load_checkpoint_history(self.checkpoint_manager.history_path)
        if not history or history[-1]["step"] != target_steps:
            raise RuntimeError("checkpoint history does not end at the exact target")
        restore_rng_state(checkpoint["rng"])
        numerator, count, mean, hashes = self._p3_validation_loss(maximum_batches=1)
        expected_fresh_hashes = {
            "parameter_sha256": checkpoint["parameter_sha256"],
            "optimizer_sha256": checkpoint["optimizer_sha256"],
            "scheduler_sha256": checkpoint["scheduler_sha256"],
            "rng_sha256": nested_state_sha256(checkpoint["rng"]),
        }
        if hashes != expected_fresh_hashes:
            raise RuntimeError(
                "fresh-load model/optimizer/scheduler/RNG differs from final checkpoint"
            )
        heldout = self.p3_run_card["heldout_loss_manifest"]
        receipt = {
            "schema": FINAL_RECEIPT_SCHEMA,
            "state": "PASS",
            "fresh_model_process": True,
            "process_id": os.getpid(),
            "training_process_id": training_process_id,
            "slurm_job_id": require_job_id(os.environ.get("SLURM_JOB_ID")),
            "source_commit": self.source_commit,
            "immutable_run_card_sha256": self.immutable_run_card_sha256,
            "config_sha256": self.p3_run_card["config_sha256"],
            "container_sha256": self.p3_run_card["container"]["sha256"],
            "target_steps": target_steps,
            "global_step": self.global_step,
            "sampler": self._sampler_state(),
            "checkpoint": str(self._last_checkpoint_path),
            "checkpoint_sha256": self._last_checkpoint_sha256,
            "checkpoint_history_record_sha256": history[-1]["record_sha256"],
            "parameter_sha256": checkpoint["parameter_sha256"],
            "optimizer_sha256": checkpoint["optimizer_sha256"],
            "scheduler_sha256": checkpoint["scheduler_sha256"],
            "rng_sha256": nested_state_sha256(checkpoint["rng"]),
            "manifest_sha256": heldout["sha256"],
            "data_manifest_sha256": heldout["data_manifest_sha256"],
            "split_sha256": heldout["split_sha256"],
            "training_ledger_sha256": completion_sha256_file(
                self.p3_training_ledger_path
            ),
            "validation_ledger_sha256": completion_sha256_file(
                self.p3_validation_ledger_path
            ),
            "checkpoint_history_sha256": completion_sha256_file(
                self.checkpoint_manager.history_path
            ),
            "dataset_order_sha256": self.dataset_order_sha256,
            "validation_batch": {
                "manifest_key": self.p3_heldout_rows[0]["key"],
                "loss_numerator": numerator,
                "element_count": count,
                "mean_loss": mean,
            },
            **self._p3_depth_provenance(),
        }
        validate_checkpoint_evidence_bindings(
            history,
            directory=self.checkpoint_manager.directory,
            source_commit=self.source_commit,
            immutable_run_card_sha256=self.immutable_run_card_sha256,
            dataset_order_sha256=self.dataset_order_sha256,
            training_rows=training_rows,
            validation_rows=validation_rows,
            final_receipt=receipt,
        )
        write_final_receipt(
            Path(self.cfg.saved_folder) / "final_acceptance.json", receipt
        )
        print("P3_FINAL_ACCEPTANCE=PASS")

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
            segment_stop = min(target_steps, segment_start + int(configured_segment))
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

        if self.p3_completion_enabled:
            if self._last_saved_step is None:
                self.save_step_checkpoint(("INITIAL_STATE",))
            self._p3_reconcile_loaded_step()

        started = time.perf_counter()
        checkpoint_every = int(self.cfg.training.checkpoint_every_steps)
        test_signal_step = self.cfg.training.test_signal_after_step
        timing = StrictTimingWindow.from_trainer(
            self,
            sampler=sampler,
            segment_start=segment_start,
            segment_stop=segment_stop,
            checkpoint_every=checkpoint_every,
        )
        try:
            for data in tqdm(
                iterator,
                total=len(sampler),
                desc=f"Steps {segment_start + 1}-{segment_stop}",
            ):
                if self._stop_requested:
                    break
                batch_samples = int(data[1].shape[0])
                self.last_step_loss = self._train_one_step(data)
                self.global_step += 1
                self.epoch = self.global_step // sampler.steps_per_epoch
                if timing is not None:
                    timing.after_step(
                        completed_step=self.global_step,
                        batch_samples=batch_samples,
                    )

                if test_signal_step is not None and self.global_step == int(
                    test_signal_step
                ):
                    os.kill(os.getpid(), signal.SIGUSR1)
                checkpoint_due = (
                    checkpoint_every > 0 and self.global_step % checkpoint_every == 0
                )
                epoch_complete = self.global_step % sampler.steps_per_epoch == 0
                reasons = []
                if checkpoint_due:
                    reasons.append("CONFIGURED_INTERVAL")
                if epoch_complete:
                    reasons.append("COMPLETE_EPOCH")
                if self._stop_requested:
                    reasons.append(
                        "USR1" if self._stop_signal == "SIGUSR1" else "SIGNAL_STOP"
                    )
                if self.global_step == segment_stop:
                    reasons.append("SEGMENT_BOUNDARY")
                if self.global_step == target_steps:
                    reasons.append("EXACT_TARGET")
                due_percents = (
                    percents_at_step(target_steps, self.global_step)
                    if self.p3_completion_enabled
                    else []
                )
                if due_percents:
                    reasons.append("INTEGER_PERCENT")
                if reasons:
                    self.save_step_checkpoint(tuple(reasons))
                if self.p3_completion_enabled:
                    self._p3_append_training_record()
                    for percent in due_percents:
                        self._p3_complete_percent(percent)
                if self._stop_requested:
                    break
        except BaseException as error:
            if timing is not None:
                timing.abort(error)
            raise

        final_reasons = []
        if self.global_step == segment_stop:
            final_reasons.append("SEGMENT_BOUNDARY")
        if self.global_step == target_steps:
            final_reasons.append("EXACT_TARGET")
        if self._stop_requested:
            final_reasons.append(
                "USR1" if self._stop_signal == "SIGUSR1" else "SIGNAL_STOP"
            )
        if self._last_saved_step != self.global_step:
            self.save_step_checkpoint(tuple(final_reasons or ["LOADER_STOP"]))
        elif final_reasons:
            self.save_step_checkpoint(tuple(final_reasons))
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
        if timing is not None:
            timing.finalize(self, sampler=sampler, status=status)

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
        if self.final_acceptance:
            self._run_p3_final_acceptance()
            return
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
                            obs["visual"].shape[0]
                            - min_horizon * self.cfg.frameskip
                            - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (
                        obs["visual"].shape[0] - start - 1
                    ) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start : start
                    + horizon * self.cfg.frameskip
                    + 1 : self.cfg.frameskip
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
                        logs[f"z_{k}_err_rollout{postfix}"].append(div_loss[k])
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [div_loss[k]]

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
        log.info(
            f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                Validation loss: {epoch_log['val_loss']:.4f}"
        )

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
