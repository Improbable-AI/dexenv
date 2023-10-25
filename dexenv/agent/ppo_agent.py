import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from loguru import logger
from omegaconf.dictconfig import DictConfig

from dexenv.utils.torch_utils import action_entropy
from dexenv.utils.torch_utils import action_from_dist
from dexenv.utils.torch_utils import action_log_prob
from dexenv.utils.torch_utils import clip_grad
from dexenv.utils.torch_utils import load_ckpt_data
from dexenv.utils.torch_utils import load_state_dict
from dexenv.utils.torch_utils import move_to
from dexenv.utils.torch_utils import save_model
from dexenv.utils.torch_utils import torch_float


@dataclass
class PPOAgent:
    actor: nn.Module
    critic: nn.Module
    cfg: DictConfig

    def __post_init__(self):
        move_to([self.actor, self.critic],
                device=self.cfg.alg.device)
        self.val_loss_criterion = nn.MSELoss().to(self.cfg.alg.device)

        self.optim_model = nn.ModuleList([self.actor, self.critic])

        optim_args = dict(
            lr=self.cfg.alg.policy_lr,
            amsgrad=True
        )

        optim_args['params'] = [{'params': self.actor.parameters(),
                                 'lr': self.cfg.alg.policy_lr},
                                {'params': self.critic.parameters(),
                                 'lr': self.cfg.alg.value_lr}
                                ]
        self.optimizer = optim.AdamW(**optim_args)

    @torch.no_grad()
    def get_action(self, ob, sample=True, get_action_only=False, *args, **kwargs):
        self.eval_mode()
        t_ob = torch_float(ob, device=self.cfg.alg.device)
        act_dist, val = self.get_act_val(t_ob, no_val=get_action_only)
        action = action_from_dist(act_dist,
                                  sample=sample)

        if not get_action_only:
            log_prob = action_log_prob(action, act_dist)
            action_info = dict(
                log_prob=log_prob.detach(),
                val=val.detach()
            )
            entropy = action_entropy(act_dist, log_prob)
            action_info['entropy'] = entropy.detach()
        else:
            action_info = dict()
        return action.detach(), action_info

    def get_act_val(self, ob, no_val=False, *args, **kwargs):
        act_dist, body_out = self.actor(ob)
        if no_val:
            val = None
        else:
            val, body_out = self.critic(x=ob)
            val = val.squeeze(-1)
        return act_dist, val

    @torch.no_grad()
    def get_val(self, ob, *args, **kwargs):
        self.eval_mode()
        ob = torch_float(ob, device=self.cfg.alg.device)
        val, body_out = self.critic(x=ob)
        val = val.squeeze(-1)
        return val

    def optimize(self, data, *args, **kwargs):
        pre_res = self.optim_preprocess(data)
        processed_data = pre_res

        self.optimizer.zero_grad(set_to_none=True)
        loss_res = self.cal_loss(**processed_data)
        loss, pg_loss, vf_loss, ratio, entropy, approx_kl, clip_frac = loss_res
        grad_norm = None
        loss.backward()
        grad_norm = clip_grad(self.optim_model.parameters(), self.cfg.alg.max_grad_norm)
        self.optimizer.step()
        optim_info = dict(
            pg_loss=pg_loss.item(),
            vf_loss=vf_loss.item(),
            total_loss=loss.item(),
            approx_kl=approx_kl.item(),
            clip_frac=clip_frac.item()
        )
        optim_info['entropy'] = entropy.item()
        if grad_norm is not None:
            optim_info['grad_norm'] = grad_norm
        return optim_info

    def optim_preprocess(self, data):
        self.train_mode()
        for key, val in data.items():
            data[key] = torch_float(val, device=self.cfg.alg.device)
        ob = data['ob']
        action = data['action']
        ret = data['ret']
        adv = data['adv']
        old_log_prob = data['log_prob']
        old_val = data['val']

        act_dist, val = self.get_act_val(ob)
        log_prob = action_log_prob(action, act_dist)
        entropy = action_entropy(act_dist, log_prob)
        if not all([x.ndim == 1 for x in [val, log_prob]]):
            raise ValueError('val, log_prob should be 1-dim!')
        processed_data = dict(
            val=val,
            old_val=old_val,
            ret=ret,
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            adv=adv,
            entropy=entropy
        )
        return processed_data

    def cal_loss(self, val, old_val, ret, log_prob, old_log_prob,
                 adv, entropy, *args, **kwargs):
        entropy = torch.mean(entropy)
        vf_loss = self.cal_val_loss(val=val, old_val=old_val, ret=ret)
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = adv * ratio
        surr2 = adv * torch.clamp(ratio,
                                  1 - self.cfg.alg.clip_range,
                                  1 + self.cfg.alg.clip_range)
        pg_loss = -torch.mean(torch.min(surr1, surr2))

        loss = pg_loss + vf_loss * self.cfg.alg.vf_coef

        loss = loss - entropy * self.cfg.alg.ent_coef

        with torch.no_grad():
            approx_kl = 0.5 * torch.mean(torch.pow(old_log_prob - log_prob, 2))
            clr = torch.abs(ratio - 1.0) > self.cfg.alg.clip_range
            clip_frac = torch.mean(clr.float())
        return loss, pg_loss, vf_loss, ratio, entropy, approx_kl, clip_frac

    def cal_val_loss(self, val, old_val, ret):
        vf_loss = self.val_loss_criterion(val, ret)
        return vf_loss

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def save_model(self, wandb_run, is_best=False, step=None, eval=False):
        data_to_save = {
            'step': step,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
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
        load_state_dict(self.critic,
                        ckpt_data.get('critic_state_dict', dict()))
        if pretrain_model is not None:
            return ckpt_data['step']
        if self.cfg.resume_optim and not self.cfg.test:
            self.optimizer.load_state_dict(ckpt_data['optim_state_dict'])
        logger.info(f"Checkpoint step:{ckpt_data['step']}")
        return ckpt_data['step']

    def print_param_grad_status(self):
        logger.info('Requires Grad?')
        logger.info('================== Actor ================== ')
        for name, param in self.actor.named_parameters():
            print(f'{name}: {param.requires_grad}')
        logger.info('================== Critic ================== ')
        for name, param in self.critic.named_parameters():
            print(f'{name}: {param.requires_grad}')
