import hydra
import isaacgym
import os
from loguru import logger
from omegaconf import DictConfig

import dexenv
from dexenv.agent.rnn_agent import RNNAgent
from dexenv.engine.rnn_engine import RNNEngine
from dexenv.models.sparse_cnn_rnn_models import get_voxel_rnn_model
from dexenv.models.state_model import get_mlp_actor
from dexenv.runner.rnn_runner import RNNRunner
from dexenv.utils.common import make_dir
from dexenv.utils.common import plot_ecff
from dexenv.utils.common import process_cfg
from dexenv.utils.common import set_print_formatting
from dexenv.utils.common import set_random_seed
from dexenv.utils.create_task_env import create_task_env


@hydra.main(config_path=dexenv.PROJECT_ROOT.joinpath('conf').as_posix(),
            config_name="debug_dclaw_fptd")
def main(cfg: DictConfig):
    process_cfg(cfg)
    make_dir(os.environ['WANDB_DIR'])
    set_random_seed(cfg.alg.seed)
    set_print_formatting()
    env = create_task_env(cfg, quantization_size=cfg.vision.quantization_size)

    if cfg.test:
        expert_actor = None
    else:
        expert_ob_size = env.observation_space['state'].shape[-1]
        logger.info(f'Creating expert actor')
        expert_actor = get_mlp_actor(expert_ob_size, env, act=cfg.alg.act)
    logger.info(f'Creating vision actor')
    act_dim = env.action_space.shape[0]

    actor = get_voxel_rnn_model(cfg, env, act_dim=act_dim)
    agent = RNNAgent(actor=actor, expert_actor=expert_actor, cfg=cfg, act_dim=act_dim)
    runner = RNNRunner(agent=agent, env=env, cfg=cfg, store_next_ob=False)
    engine = RNNEngine(agent=agent, runner=runner, cfg=cfg, act_dim=act_dim)
    if not (cfg.test or cfg.test_pretrain):
        engine.train()
    else:
        stat_info, raw_traj_info = engine.eval(eval_num=cfg.test_num,
                                               sleep_time=0.04,
                                               return_on_done=True, )
        import pprint
        pprint.pprint(stat_info)
        if 'abs_dist' in raw_traj_info:
            dist = raw_traj_info['abs_dist']
            save_distance_distribution(dist, engine)
            plot_ecff(dist)


if __name__ == '__main__':
    main()
