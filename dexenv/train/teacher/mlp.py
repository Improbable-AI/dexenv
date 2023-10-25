import hydra
import isaacgym
import os
from omegaconf import DictConfig

import dexenv
from dexenv.agent.ppo_agent import PPOAgent
from dexenv.engine.ppo_engine import PPOEngine
from dexenv.models.state_model import get_mlp_models
from dexenv.runner.nstep_runner import NStepRunner
from dexenv.utils.common import make_dir
from dexenv.utils.common import set_print_formatting
from dexenv.utils.common import set_random_seed
from dexenv.utils.create_task_env import create_task_env


@hydra.main(config_path=dexenv.PROJECT_ROOT.joinpath('conf').as_posix(), config_name="debug_dclaw")
def main(cfg: DictConfig):
    make_dir(os.environ['WANDB_DIR'])
    set_random_seed(cfg.alg.seed)
    set_print_formatting()
    env = create_task_env(cfg)

    ob_size = env.observation_space['ob'].shape[0]
    actor, critic = get_mlp_models(ob_size, env, act=cfg.alg.act,
                                   init_log_std=cfg.alg.init_logstd)
    agent = PPOAgent(actor=actor, critic=critic, cfg=cfg)
    runner = NStepRunner(agent=agent, env=env, cfg=cfg, store_next_ob=False)
    engine = PPOEngine(agent=agent,
                       runner=runner,
                       cfg=cfg)
    if not (cfg.test or cfg.test_pretrain):
        engine.train()
    else:
        stat_info, raw_traj_info = engine.eval(render=cfg.render,
                                               eval_num=cfg.test_num,
                                               sleep_time=0.04)
        import pprint
        pprint.pprint(stat_info)


if __name__ == '__main__':
    main()
