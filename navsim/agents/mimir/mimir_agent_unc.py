from typing import Any, List, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
import numpy as np
import os

from navsim.agents.abstract_agent_mimir import AbstractAgent
from navsim.agents.mimir.mimir_config_unc import MimirConfig

from navsim.agents.mimir.mimir_model_unc import MimirModel as MimirModel

from navsim.agents.mimir.mimir_callback import MimirCallback 
from navsim.agents.mimir.mimir_loss_unc import mimir_loss
from navsim.agents.mimir.mimir_features import MimirFeatureBuilder, MimirTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.mimir.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf, open_dict
import torch.optim as optim
from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

class MimirAgent(AbstractAgent):
    """Agent interface for Mimir baseline."""

    def __init__(
        self,
        config: MimirConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes Mimir agent.
        :param config: global config of Mimir agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._mimir_model = MimirModel(config)
        self.init_from_pretrained()

    def init_from_pretrained(self):
        # import ipdb; ipdb.set_trace()
        if self._checkpoint_path:
            if torch.cuda.is_available():
                checkpoint = torch.load(self._checkpoint_path)
            else:
                checkpoint = torch.load(self._checkpoint_path, map_location=torch.device('cpu'))
            
            state_dict = checkpoint['state_dict']
            
            # Remove 'agent.' prefix from keys if present
            state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}
            
            # Load state dict and get info about missing and unexpected keys
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        else:
            print("No checkpoint path provided. Initializing from scratch.")
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})


    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [MimirTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [MimirFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None,goalpoint=None,token=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._mimir_model(features,targets=targets,goalpoint=goalpoint,token=token)



    def compute_trajectory(self, agent_input: AgentInput,token) -> Tuple[np.ndarray,np.ndarray]:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # targets: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        
        with torch.no_grad():
            features = {k: v.unsqueeze(0) for k, v in features.items()}
            predictions = self.forward(features,token=token)
    
            navi= predictions["navi"].squeeze()
            unc= predictions["unc"].squeeze()
            # # unc_path='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/socket/navsim_exp/1_goal_point_unc/navhard_noise'
            # # unc_path='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/socket/navsim_exp/1_goal_point_unc/navhard_lidar'
            # # unc_path='/lpai/volumes/ad-e2e-vol-ga/zhengyupeng/workspace/socket/navsim_exp/1_goal_point_unc/navhard_plus_revise'
            # unc_path=self._config.navi_unc_outputdir
            # os.makedirs(unc_path,exist_ok=True)
            # os.makedirs(os.path.join(unc_path,'navi'),exist_ok=True)
            # os.makedirs(os.path.join(unc_path,'unc'),exist_ok=True)
            # np.save(os.path.join(unc_path,'navi',f'{token}.npy'),navi.cpu().numpy())
            # np.save(os.path.join(unc_path,'unc',f'{token}.npy'),unc.cpu().numpy())

            return navi.cpu().numpy(), unc.cpu().numpy()

        return navi,unc

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return mimir_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_step_lr_optimizers(self):
        optimizer = torch.optim.Adam(self._mimir_model.parameters(), lr=self._lr, weight_decay=self._config.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._config.lr_steps, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_coslr_optimizers(self):
        # import ipdb; ipdb.set_trace()
        optimizer_cfg = dict(type=self._config.optimizer_type, 
                            lr=self._lr, 
                            weight_decay=self._config.weight_decay,
                            paramwise_cfg=self._config.opt_paramwise_cfg
                            )
        scheduler_cfg = dict(type=self._config.scheduler_type,
                            milestones=self._config.lr_steps,
                            gamma=0.1,
        )

        optimizer_cfg = DictConfig(optimizer_cfg)
        scheduler_cfg = DictConfig(scheduler_cfg)
        
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        
        if paramwise_cfg:
            params = []
            pgs = [[] for _ in paramwise_cfg['name']]

            for k, v in self._mimir_model.named_parameters():
                in_param_group = True
                for i, (pattern, pg_cfg) in enumerate(paramwise_cfg['name'].items()):
                    if pattern in k:
                        pgs[i].append(v)
                        in_param_group = False
                if in_param_group:
                    params.append(v)
        else:
            params = self._mimir_model.parameters()
        
        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        # import ipdb; ipdb.set_trace()
        if paramwise_cfg:
            for pg, (_, pg_cfg) in zip(pgs, paramwise_cfg['name'].items()):
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg['lr'] * pg_cfg['lr_mult']
                optimizer.add_param_group({'params': pg, **cfg})
        
        # scheduler = build_from_configs(optim.lr_scheduler, scheduler_cfg, optimizer=optimizer)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=1e-6,
            epochs=100,
            warmup_epochs=3,
        )
        
        if 'interval' in scheduler_cfg:
            scheduler = {'scheduler': scheduler, 'interval': scheduler_cfg['interval']}
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [MimirCallback(self._config),pl.callbacks.ModelCheckpoint(every_n_epochs=5, save_top_k=-1)]