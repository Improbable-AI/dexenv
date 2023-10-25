import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections.abc import Sequence
from dataclasses import dataclass
from loguru import logger
from omegaconf.dictconfig import DictConfig

from dexenv.utils.torch_utils import action_entropy
from dexenv.utils.torch_utils import action_from_dist
from dexenv.utils.torch_utils import action_log_prob
from dexenv.utils.torch_utils import clip_grad
from dexenv.utils.torch_utils import freeze_model
from dexenv.utils.torch_utils import load_ckpt_data
from dexenv.utils.torch_utils import load_state_dict
from dexenv.utils.torch_utils import move_to
from dexenv.utils.torch_utils import save_model
from dexenv.utils.torch_utils import torch_float


@dataclass
class RNNAgent:
    act_dim: int
    actor: nn.Module
    expert_actor: nn.Module
    cfg: DictConfig

    def __post_init__(self):
        move_to([self.actor, self.expert_actor],
                device=self.cfg.alg.device)
        self.optimizer = optim.AdamW(self.actor.parameters(),
                                     lr=self.cfg.alg.lr,
                                     amsgrad=True,
                                     fused=True)

        self.load_expert_model()
        if self.expert_actor is not None:
            self.expert_actor.eval()
            freeze_model(self.expert_actor)

    @torch.no_grad()
    def extract_obs(self, ob):
        actor_ob = ob['ob'].to(self.cfg.alg.device).unsqueeze(dim=1)
        if 'state' in ob:
            expert_actor_ob = ob['state'].to(self.cfg.alg.device)
            if self.cfg.vision.act_in:
                states = ob['state']
                prev_action = states[..., -self.act_dim:].to(self.cfg.alg.device).unsqueeze(dim=1)
                actor_ob = dict(ptd=actor_ob, prev_action=prev_action)
        else:
            expert_actor_ob = None

        return actor_ob, expert_actor_ob

    @torch.no_grad()
    def get_action(self, ob, sample=True, hidden_state=None, get_action_only=False, *args, **kwargs):
        self.eval_mode()
        actor_ob, expert_actor_ob = self.extract_obs(ob)

        actor_hidden_state = hidden_state[0] if not isinstance(hidden_state, torch.Tensor) and hidden_state is not None else hidden_state

        act_dist, body_out, out_hidden_state = self.actor(actor_ob, hidden_state=actor_hidden_state, done=None)
        action = action_from_dist(act_dist, sample=sample)

        if not get_action_only:
            entropy = action_entropy(act_dist)
            exp_act_dist, _ = self.expert_actor(expert_actor_ob)

            in_hidden_state = actor_hidden_state.detach() if actor_hidden_state is not None else actor_hidden_state
            action_info = dict(
                entropy=entropy.detach(),
                exp_act_loc=exp_act_dist.base_dist.loc,
                exp_act_scale=exp_act_dist.base_dist.scale,
                in_hidden_state=in_hidden_state
            )
        else:
            action_info = dict()
        if self.cfg.test and self.cfg.vision.pred_rot_dist:
            action_info['pred_rot_dist'] = body_out[3].detach()

        return action.squeeze(1).detach(), action_info, out_hidden_state

    def optim_preprocess(self, data):
        self.train_mode()
        for key, val in data.items():
            if key == 'ob':
                data[key] = data[key].clone().to(self.cfg.alg.device)
            else:
                data[key] = torch_float(val, device=self.cfg.alg.device)
        ob = data['ob']
        if self.cfg.vision.act_in:
            prev_actions = data['prev_actions']
            ob = dict(ptd=ob, prev_action=prev_actions)
        exp_act_loc = data['exp_act_loc']
        exp_act_scale = data['exp_act_scale']

        hidden_state = data['hidden_state']
        hidden_state = hidden_state.permute(1, 0, 2)
        done = data['done']
        act_dist, body_out, out_hidden_state = self.actor(x=ob,
                                                          hidden_state=hidden_state,
                                                          done=done)
        entropy = action_entropy(act_dist)
        processed_data = dict(
            act_dist=act_dist,
            exp_act_loc=exp_act_loc,
            exp_act_scale=exp_act_scale,
            entropy=entropy
        )
        if self.cfg.vision.pred_rot_dist:
            processed_data['rot_dist'] = data['rot_dist']
            processed_data['rot_dist_pred'] = body_out[-1]
        return processed_data

    def optimize(self, data, *args, **kwargs):
        pre_res = self.optim_preprocess(data)
        processed_data = pre_res

        loss = self.cal_loss(**processed_data)
        if isinstance(loss, Sequence):
            loss_info = loss[1]
            loss = loss[0]
        else:
            loss_info = dict()
        grad_norm = None
        loss.backward()
        grad_norm = clip_grad(self.actor.parameters(), self.cfg.alg.max_grad_norm)
        self.optimizer.step()
        self.clear_optim_grad()

        optim_info = dict(
            loss=loss.item(),
        )
        if grad_norm is not None:
            optim_info['grad_norm'] = grad_norm
        optim_info.update(loss_info)
        if 'entropy' in processed_data:
            optim_info['entropy'] = processed_data['entropy'].mean().item()
        return optim_info

    def clear_optim_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def cal_loss(self, act_dist, exp_act_loc, exp_act_scale, mask=None, **kwargs):
        loss = 0
        if self.cfg.vision.clip_action and not self.cfg.alg.loss == 'kl':
            exp_act_loc = exp_act_loc.clamp(-1. + self.cfg.vision.clip_eps,
                                            1. - self.cfg.vision.clip_eps)
        bc_loss = -action_log_prob(exp_act_loc, act_dist)
        if mask is None:
            bc_loss = bc_loss.mean()
        else:
            bc_loss = bc_loss[mask].mean()
        loss += bc_loss
        loss_info = dict(bc_loss=bc_loss.item())
        if self.cfg.vision.pred_rot_dist:
            rot_dist = kwargs['rot_dist']
            rot_dist_pred = kwargs['rot_dist_pred'].squeeze(-1)
            rot_dist_loss = F.mse_loss(rot_dist_pred, rot_dist, reduction='none')
            if mask is not None:
                rot_dist_loss = rot_dist_loss[mask]
            rot_dist_loss = rot_dist_loss.mean()
            loss += rot_dist_loss * self.cfg.vision.rot_dist_loss_coef
            loss_info['rot_dist_loss'] = rot_dist_loss.item()
        return loss, loss_info

    def train_mode(self):
        self.actor.train()

    def eval_mode(self):
        self.actor.eval()

    def save_model(self, wandb_run, is_best=False, step=None, eval=False):
        data_to_save = {
            'step': step,
            'actor_state_dict': self.actor.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
        }
        save_model(data_to_save,
                   wandb_run,
                   is_best=is_best,
                   step=step,
                   eval=eval)

    def load_model(self, pretrain_model=None, eval=False):
        ckpt_data = load_ckpt_data(self.cfg.resume_id,
                                   project_name=self.cfg.logging.wandb.project,
                                   pretrain_model=pretrain_model, eval=eval)
        load_state_dict(self.actor,
                        ckpt_data.get('actor_state_dict', dict()))
        if pretrain_model is not None:
            return ckpt_data['step']
        if self.cfg.resume_optim and not self.cfg.test:
            self.optimizer.load_state_dict(ckpt_data['optim_state_dict'])
        logger.info(f"Checkpoint step:{ckpt_data['step']}")
        return ckpt_data['step']

    def load_expert_model(self):
        if self.expert_actor is None:
            return
        ckpt_data = load_ckpt_data(wandb_run_id=None,
                                   pretrain_model=self.cfg.alg.expert_path,
                                   eval=False)
        load_state_dict(self.expert_actor,
                        ckpt_data.get('actor_state_dict', dict()))
        logger.info(f"Expert model checkpoint step:{ckpt_data['step']}")
