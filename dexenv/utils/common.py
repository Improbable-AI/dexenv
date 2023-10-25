import ast
import collections
import git
import importlib
import numbers
import numpy as np
import os
import pickle as pkl
import random
import shutil
import torch
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def stat_for_traj_data(traj_data, dones, op='min'):
    """
    traj_data: T x #of envs
    """
    op = getattr(np, op)
    stats = []
    for i in range(dones.shape[1]):
        stat = []
        di = dones[:, i]
        if not np.any(di):
            stat.append(op(traj_data[:, i]))
        else:
            done_idx = np.where(di)[0]
            t = 0
            for idx in done_idx:
                stat.append(op(traj_data[t: idx + 1, i]))
                t = idx + 1
        stats.append(stat)
    return np.concatenate(stats)


def check_torch_tensor(data):
    if isinstance(data, torch.Tensor):
        return True
    if not hasattr(data, '__len__'):  # then data is a scalar
        return False
    if len(data) > 0:
        return check_torch_tensor(data[0])
    return False


def stack_data(data, torch_to_numpy=False, dim=0):
    if isinstance(data[0], dict):
        out = dict()
        for key in data[0].keys():
            out[key] = stack_data([x[key] for x in data], dim=dim)
        return out
    if check_torch_tensor(data):
        try:
            ret = torch.stack(data, dim=dim)
            if torch_to_numpy:
                ret = ret.cpu().numpy()
        except:
            # if data is a list of arrays that do not have same shapes (such as point cloud)
            ret = data
    else:
        try:
            ret = np.stack(data, axis=dim)
        except:
            ret = data
    return ret


def get_module_path(module):
    modu = importlib.util.find_spec(module)
    return Path(list(modu.submodule_search_locations)[0])


def ask_before_remake_dir(directory):
    make_dir(directory, ask=True, delete=True)


def flatten_nested_dict(d, parent_key='', sep='_'):
    """
    Flatten nested dictionaries, compressing keys

    e.g.:

    Input:
       a = dict(b=dict(c=1,k=1), c=3)

    Output:
       {'b_c': 1, 'b_k': 1, 'c': 3}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_dir(directory, ask=False, delete=False):
    directory = pathlib_file(directory)
    if directory.exists():
        if delete:
            if ask:
                res = input(f'{directory} already exists, do u want to delete it?')
                if res == 'y':
                    shutil.rmtree(directory)
            else:
                shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def get_all_subdirs(directory, exclude_patterns=None, sort=True):
    directory = pathlib_file(directory)
    folders = list(directory.iterdir())
    folders = [x for x in folders if x.is_dir()]
    if exclude_patterns is not None:
        folders = filter_with_exclude_patterns(folders, exclude_patterns)
    if sort:
        folders = sorted(folders)
    return folders


def get_all_subfiles(directory, exclude_patterns=None, sort=True):
    directory = pathlib_file(directory)
    files = list(directory.iterdir())
    files = [x for x in files if x.is_file()]
    if exclude_patterns is not None:
        files = filter_with_exclude_patterns(files, exclude_patterns)
    if sort:
        files = sorted(files)
    return files


def get_all_files_with_suffix(directory, suffix,
                              exclude_patterns=None,
                              include_patterns=None,
                              sort=True):
    directory = pathlib_file(directory)
    if not suffix.startswith('.'):
        suffix = '.' + suffix
    files = directory.glob(f'**/*{suffix}')
    files = [x for x in files if x.is_file() and x.suffix == suffix]
    if exclude_patterns is not None:
        files = filter_with_exclude_patterns(files, exclude_patterns)
    if include_patterns is not None:
        files = filter_with_include_patterns(files, include_patterns)
    if sort:
        files = sorted(files)
    return files


def get_all_files_with_name(directory, name,
                            exclude_patterns=None,
                            include_patterns=None,
                            sort=True,
                            ):
    directory = pathlib_file(directory)
    files = directory.glob(f'**/{name}')
    files = [x for x in files if x.is_file() and x.name == name]
    if exclude_patterns is not None:
        files = filter_with_exclude_patterns(files, exclude_patterns)
    if include_patterns is not None:
        files = filter_with_include_patterns(files, include_patterns)
    if sort:
        files = sorted(files)
    return files


def filter_with_exclude_patterns(filenames, exclude_patterns):
    if isinstance(exclude_patterns, str):
        files = [x for x in filenames if exclude_patterns not in x.as_posix()]
    else:
        out = []
        for x in filenames:
            contain_patterns = [epat in x.as_posix() for epat in exclude_patterns]
            if not any(contain_patterns):
                out.append(x)
        files = out
    return files


def filter_with_include_patterns(filenames, include_patterns):
    if isinstance(include_patterns, str):
        files = [x for x in filenames if include_patterns in x.as_posix()]
    else:
        out = []
        for x in filenames:
            contain_patterns = [ipat in x.as_posix() for ipat in include_patterns]
            if any(contain_patterns):
                out.append(x)
        files = out
    return files


def set_random_seed(seed):
    rank = get_global_rank()
    seed += rank * 100000
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f'Setting random seed to:{seed}')


def chunker_list(seq_list, nchunks):
    # split the list into n parts/chunks
    return [seq_list[i::nchunks] for i in range(nchunks)]


def module_available(module_path: str) -> bool:
    """Testing if given module is avalaible in your env.

    Copied from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/__init__.py.

    >>> module_available('os')
    True
    >>> module_available('bla.bla')
    False
    """
    try:
        mods = module_path.split('.')
        assert mods, 'nothing given to test'
        # it has to be tested as per partets
        for i in range(len(mods)):
            module_path = '.'.join(mods[:i + 1])
            if importlib.util.find_spec(module_path) is None:
                return False
        return True
    except AttributeError:
        return False


def get_env_var(key, default=None):
    envrion = dict(os.environ)
    value = default
    if isinstance(key, list):
        for subkey in key:
            if subkey in envrion:
                value = envrion[subkey]
                break
    else:
        if key in envrion:
            value = envrion[key]
    return value


def list_to_numpy(data, expand_dims=None):
    if isinstance(data, numbers.Number):
        data = np.array([data])
    else:
        data = np.array(data)
    if expand_dims is not None:
        data = np.expand_dims(data, axis=expand_dims)
    return data


def save_to_pickle(data, file_name):
    file_name = pathlib_file(file_name)
    if not file_name.parent.exists():
        Path.mkdir(file_name.parent, parents=True)
    with file_name.open('wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


def load_from_pickle(file_name):
    file_name = pathlib_file(file_name)
    with file_name.open('rb') as f:
        data = pkl.load(f)
    return data


def pathlib_file(file_name):
    if isinstance(file_name, str):
        file_name = Path(file_name)
    elif not isinstance(file_name, Path):
        raise TypeError(f'Please check the type of the filename:{file_name}')
    return file_name


def linear_decay_percent(epoch, total_epochs):
    return 1 - epoch / float(total_epochs)


def smooth_value(current_value, past_value, tau):
    if past_value is None:
        return current_value
    else:
        return past_value * tau + current_value * (1 - tau)


def get_list_stats(data):
    if len(data) < 1:
        return dict()
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    min_data = np.amin(data)
    max_data = np.amax(data)
    mean_data = np.mean(data)
    median_data = np.median(data)
    stats = dict(
        min=min_data,
        max=max_data,
        mean=mean_data,
        median=median_data
    )
    return stats


def get_git_infos(path):
    git_info = None
    try:
        repo = git.Repo(path)
        try:
            branch_name = repo.active_branch.name
        except TypeError:
            branch_name = '[DETACHED]'
        git_info = dict(
            directory=str(path),
            code_diff=repo.git.diff(None),
            code_diff_staged=repo.git.diff('--staged'),
            commit_hash=repo.head.commit.hexsha,
            branch_name=branch_name,
        )
    except git.exc.InvalidGitRepositoryError as e:
        logger.error(f'Not a valid git repo: {path}')
    except git.exc.NoSuchPathError as e:
        logger.error(f'{path} does not exist.')
    return git_info


def generate_id(length=8):
    """
    https://github.com/wandb/client/blob/master/wandb/util.py#L677
    """
    import shortuuid

    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return run_gen.random(length)


def generate_id(length=8):
    """
    https://github.com/wandb/client/blob/master/wandb/util.py#L677
    """
    import shortuuid

    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return run_gen.random(length)


def pathlib_file(file_name):
    if isinstance(file_name, str):
        file_name = Path(file_name)
    elif not isinstance(file_name, Path):
        raise TypeError(f'Please check the type of the filename:{file_name}')
    return file_name


def make_dir(directory, ask=False, delete=False):
    directory = pathlib_file(directory)
    if directory.exists():
        if delete:
            if ask:
                res = input(f'{directory} already exists, do u want to delete it?')
                if res == 'y':
                    shutil.rmtree(directory)
            else:
                shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def restore_hydra_cfg(cfg):
    if cfg.resume_root is not None and (cfg.resume or cfg.test):
        root_dir = pathlib_file(cfg.resume_root)
        if not root_dir.is_absolute():
            base = pathlib_file(os.getcwd()).parent
            root_dir = base / root_dir
        output_dir = pathlib_file(root_dir)
        original_overrides = OmegaConf.load(output_dir.joinpath('.hydra/overrides.yaml'))
        current_overrides = HydraConfig.get().overrides.task
        config_name = HydraConfig.get().job.config_name
        overrides = original_overrides + current_overrides
        cfg = compose(config_name, overrides=overrides, return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
        os.chdir(output_dir)
    return cfg


def get_hydra_run_dir(pathlib=True):
    ## if run.dir is set via a lambda function for generating a random UUID
    ## HydraConfig.get().run.dir will return a random UUID every time this variable is called
    ## so to be safe, we use os.getcwd()
    # run_dir = HydraConfig.get().run.dir
    run_dir = os.getcwd()
    if pathlib:
        run_dir = pathlib_file(run_dir)
    return run_dir


def override_hydra_cfg(cfg, overrides):
    config_name = HydraConfig.get().job.config_name
    cfg = compose(config_name, overrides=overrides, return_hydra_config=True)
    return cfg


def set_print_formatting():
    """ formats numpy print """
    configs = dict(
        precision=6,
        edgeitems=30,
        linewidth=1000,
        threshold=5000,
    )
    np.set_printoptions(suppress=True,
                        formatter=None,
                        **configs)
    torch.set_printoptions(sci_mode=False, **configs)


def random_z_orientation():
    angle = np.random.uniform(-np.pi, np.pi, 1)
    r = R.from_rotvec(angle * np.array([0, 0, 1]))
    return r.as_matrix()


def list_class_names(dir_path):
    """
    Return the mapping of class names in all files
    in dir_path to their file path.
    Args:
        dir_path (str): absolute path of the folder.
    Returns:
        dict: mapping from the class names in all python files in the
        folder to their file path.
    """
    dir_path = pathlib_file(dir_path)
    py_files = list(dir_path.rglob('*.py'))
    py_files = [f for f in py_files if f.is_file() and f.name != '__init__.py']
    cls_name_to_path = dict()
    for py_file in py_files:
        with py_file.open() as f:
            node = ast.parse(f.read())
        classes_in_file = [n for n in node.body if isinstance(n, ast.ClassDef)]
        cls_names_in_file = [c.name for c in classes_in_file]
        for cls_name in cls_names_in_file:
            cls_name_to_path[cls_name] = py_file
    return cls_name_to_path


def load_class_from_path(cls_name, path):
    """
    Load a class from the file path.
    Args:
        cls_name (str): class name.
        path (str): python file path.
    Returns:
        Python Class: return the class A which is named as cls_name.
        You can call A() to create an instance of this class using
        the return value.
    """
    mod_name = 'MOD%s' % cls_name
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def chunker_list(seq_list, nchunks):
    # split the list into n parts/chunks
    return [seq_list[i::nchunks] for i in range(nchunks)]


def resolve_expert_model_path(model_path):
    model_path = pathlib_file(model_path)
    if not model_path.is_absolute():
        import dexenv
        model_path = dexenv.LIB_PATH / model_path
    return model_path


def string_list_to_list(list_in_str):
    out = list_in_str
    if isinstance(list_in_str, str):
        out = ast.literal_eval(list_in_str)
    return out


def process_cfg(cfg):
    if cfg.alg.expert_path is not None and 'wandb:' not in cfg.alg.expert_path:
        cfg.alg.expert_path = resolve_expert_model_path(cfg.alg.expert_path).as_posix()


def plot_ecff(dist):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(font_scale=2, rc={'figure.figsize': (8, 6)})
        sns.set_style("whitegrid")
        # ax = sns.histplot(dist, stat='probability')
        ax = sns.displot(dist, kind='ecdf')
        plt.tight_layout()
        plt.show()
    except:
        print(f'Seaborn/Matplotlib might have not been installed!')
        pass
