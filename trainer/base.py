import transformers
from transformers import HfArgumentParser, set_seed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from datasets import disable_caching, load_dataset
import accelerate
from accelerate import Accelerator
from tqdm.auto import tqdm
from accelerate.logging import get_logger

import os
import evaluate
from pprint import pprint
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForTokenClassification,
    TrainingArguments,
)
from models.TimeSeries.TCN.TCN import TCNModel
from models.TimeSeries.LSTM.LSTM import LSTMModel
from data.data_processing import Window_maker








@dataclass
class BaseTrainingArguments(TrainingArguments):
    save_epochs: int = 1
    dataset: Optional[str] = None
    project: str = "Time_sereis"
    model_type: str = "TCN"
    window_size : int = 90
    test_size : int = 30
    predict_size : int = 15
    sliding_size : int =1
    feature_size : int = 5
    batch_size : int =64
    fully_size : int =128
    feature_set_num : int = 1
    target : str = 'Close'

def collate_dictlist(dl):
    from collections import defaultdict
    out = defaultdict(list)
    for d in dl:
        for k, v in d.items():
            out[k].append(v)
    return out


class BaseTrainer:
    def __init__(self, accelerator: Accelerator, args: BaseTrainingArguments) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.args = args

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_model(self, args):
        model_cls = MODEL_TYPES[args.model_type]
        kwargs = {}
        print("load model", args.model_type, model_cls, args.model_name_or_path)
        if args.model_type == "sequence-classification":
            kwargs["num_labels"] = args.num_labels
        if args.config_name is not None:
            config = AutoConfig.from_pretrained(args.config_name)
            model = model_cls(config, **kwargs)
        elif args.model_name_or_path is not None:
            model = model_cls.from_pretrained(
                args.model_name_or_path,
                revision=args.revision,
                from_flax=args.from_flax,
                **kwargs,
            )
        else:
            raise Exception("config_name or model_name_or_path 가 지정되어야 합니다.")

        return model

    def get_tokenizer(self, args):
        if args.tokenizer_name is not None:
            return AutoTokenizer.from_pretrained(args.tokenizer_name)
        elif args.model_name_or_path is not None:
            return AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            raise Exception("config_name or model_name_or_path 가 지정되어야 합니다.")

    def setup(self):
        self.model = self.get_model(self.args)
        train_data, test_data = self.prepare_dataset()
        self.train_dataloader ,  self.eval_dataloader = self._create_dataloader(train_data,test_data)
        steps_per_epoch = len(train_data) / (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
        )
        total_steps = int(self.args.num_train_epochs * steps_per_epoch)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        lr_scheduler = None
        if self.args.lr_scheduler_type == "linear":
            lr_scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                self.args.warmup_steps
                if self.args.warmup_steps > 0
                else int(total_steps * self.args.warmup_ratio),
                total_steps,
            )
        elif self.args.lr_scheduler_type == "cosine":
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                self.args.warmup_steps
                if self.args.warmup_steps > 0
                else int(total_steps * self.args.warmup_ratio),
                total_steps,
            )

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            optimizer,
            self.train_dataloader,
            lr_scheduler,
            self.eval_dataloader,
        )

        self.accelerator.register_for_checkpointing(lr_scheduler)


    def prepare_dataset(self):
        dataname = self.args.dataset
        data= load_dataset('sehyun66/STOCKPRICE',dataname)
        data = data['train'].to_pandas()
        feature_set={
            1: ['Open', 'High','Low','Close', 'Volume'],
            2: ['Open', 'High','Low', 'Volume']
        }
        kwargs = {}
        kwargs['window_size'] = self.args.window_size
        kwargs['predict_size'] = self.args.predict_size
        kwargs['sliding_size'] = self.args.sliding_size
        kwargs['test_size'] = self.args.test_size
        kwargs['data'] =data
        # kwargs['feature'] = feature_set[self.args.feature_set_num]

        window_maker = Window_maker(**kwargs)
        original , scaled ,min_max_list= window_maker.preprocess_window()
        train_x,train_y,vaild_x,vaild_y = scaled
        trian_x_o,train_y_o,vaild_x_o,vaild_y_o = original
        train_data = self.make_dataset(train_x,train_y,train_y_o,self.args.target,window_maker)
        test_data =  self.make_dataset(vaild_x,vaild_y,vaild_y_o,self.args.target,window_maker,train=False)
        return train_data,test_data

    def make_dataset(self):
        """
        return Dataset(data)

        """
        return None


    def get_collator(self):
        return None
    def _shared_step(self, batch):
        """
        return loss
        """
        pass
    def training_step(self, batch):
        """
        return loss
        """
        pass

    def evaluation_step(self, batch):
        """
        return dict
        """
        pass

    def collate_evaluation(self, results: List[Dict]):
        """
        return dict(metric)
        """
        return None

    def _create_dataloader(self, train_data, test_data):
        if train_data is None:
            return None
        train_dataloader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True,drop_last=True)
        test_dataloader = DataLoader(test_data, batch_size=self.args.batch_size, shuffle=True,drop_last=True)
        return train_dataloader,test_dataloader

    def train(self):
        global_step = 0
        optimizer_step = 0

        for epoch in tqdm(
            range(int(self.args.num_train_epochs)),
            position=1,
            disable=not self.accelerator.is_local_main_process,
        ):
            self.model.train()
            epoch_tqdm = tqdm(
                self.train_dataloader,
                disable=not self.accelerator.is_local_main_process,
                position=0,
                leave=False,
            )

            for step, batch in enumerate(epoch_tqdm):
                with self.accelerator.accumulate(self.model):
                    step_output = self.training_step(batch)
                    if torch.is_tensor(step_output):
                        loss = step_output
                    else:
                        loss = step_output["loss"].mean()
                        acc = step_output["acc"].mean()
                        mae = step_output["mae"].mean()

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    optimizer_step += 1

                    if (
                        self.accelerator.is_main_process
                        and optimizer_step % self.args.logging_steps == 0
                    ):
                        if torch.is_tensor(step_output):
                            metrics = {"train/loss": step_output.item()}
                        else:
                            metrics = {
                                f"train/{k}": v.item() for k, v in step_output.items()
                            }

                        metrics["optimizer_step"] = optimizer_step
                        metrics[
                            "train/learning_rate"
                        ] = self.lr_scheduler.scheduler._last_lr[0]
                        metrics["train/loss"] = (
                            loss.item() * self.args.gradient_accumulation_steps
                        )
                        metrics["epoch"] = epoch
                        metrics["acc"] = acc
                        metrics['mae'] =mae
                        self.accelerator.log(metrics)
                        print()
                        pprint(metrics)
                        print()
                    epoch_tqdm.set_description(
                        f"loss: {loss.item() * self.args.gradient_accumulation_steps}"
                    )
                    if (
                        self.args.do_eval
                        and self.args.evaluation_strategy == "steps"
                        and optimizer_step % self.args.eval_steps == 0
                    ):
                        self.evaluate(epoch, optimizer_step)

                    global_step += 1

            if (
                self.args.save_strategy == "epoch"
                and (epoch + 1) % self.args.save_epochs == 0
            ):
                self.save_model(f"epoch-{epoch}")

            if self.args.do_eval and self.args.evaluation_strategy == "epoch":
                self.evaluate(epoch, optimizer_step)

        if self.args.save_strategy == "last":
            self.save_model(f"epoch-{epoch}-last")

    def save_model(self, name):
        run_name = self.args.run_name.replace("/", "__")
        path = f"{self.args.output_dir}/{run_name}/{name}"
        unwrapped_model = self.accelerator.unwrap_model(self.model).cpu()

        if self.accelerator.is_local_main_process:  # Save the model
            torch.save(unwrapped_model.state_dict(), path)
        unwrapped_model.to(self.accelerator.device)
        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def evaluate(self, epoch, optimizer_step):
        self.model.eval()

        epoch_tqdm = tqdm(
            self.eval_dataloader,
            disable=not self.accelerator.is_local_main_process,
            position=1,
            leave=False,
        )
        step_outputs = []
        for step, batch in enumerate(epoch_tqdm):
            outputs = self.evaluation_step(batch)
            if torch.is_tensor(outputs):
                outputs = {"loss": outputs}
            step_outputs.append(outputs)

        eval_outputs = self.accelerator.gather_for_metrics(step_outputs)

        if self.accelerator.is_local_main_process:
            eval_outputs = collate_dictlist(eval_outputs)
            eval_results = self.collate_evaluation(eval_outputs)
            eval_results = {f"eval/{k}": v for k, v in eval_results.items()}
            eval_results["epoch"] = epoch
            self.accelerator.log(eval_results)

        self.accelerator.wait_for_everyone()
        self.model.train()

    @classmethod
    def main(trainer_cls, arg_cls: BaseTrainingArguments):
        parser = HfArgumentParser((arg_cls,))
        args = parser.parse_args()

        set_seed(args.seed)

        os.environ["WANDB_NAME"] = args.project
        accelerator = Accelerator(
            log_with="wandb",
            kwargs_handlers=[
                accelerate.DistributedDataParallelKwargs(
                    broadcast_buffers=False,
                )
            ],
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        accelerator.init_trackers(args.project, config=args)
        trainer = trainer_cls(accelerator, args)
        trainer.setup()
        if args.do_train:
            trainer.train()
        elif args.do_eval:
            trainer.evaluate(0, 0)
        accelerator.end_training()
