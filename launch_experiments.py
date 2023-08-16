from ray import tune
from typing import Dict, Tuple, Iterator
from pathlib import Path
from tqdm import tqdm
import argparse
import importlib
import os
import torch
import ray

def _is_str_float(val: str):
    try:
        float(val)
        return True
    except ValueError:
        return False


def _is_str_int(val: str):
    try:
        int(val)
        return True
    except ValueError:
        return False


def _check_ray_remote_args_dict(remote_args: Dict[str, object]):
    invalid_args = []
    for argname in remote_args.keys():
        if argname not in _VALID_RAY_REMOTE_ARGNAMES:
            invalid_args.append(argname)

    if len(invalid_args) > 0:
        raise ValueError((
            'The keys in the remote_args dict passed to a RemoteActorConfig ' +
            'constructor must be in the set %s.  However, got invalid '       +
            'argnames %s'
        ) % (_VALID_RAY_REMOTE_ARGNAMES, invalid_args))


def _get_actor_confs(obj: object) -> Iterator['RemoteActorConfig']:
    iterator = None
    if isinstance(obj, dict):
        iterator = obj.values()
    elif isinstance(obj, (list, set)):
        iterator = obj

    if iterator is None:
        if isinstance(obj, RemoteActorConfig):
            yield obj
    else:
        for val in iterator:
            yield from _get_actor_confs(val)


class RemoteActorConfig:
    def __init__(
        self,
        remote_args: Dict[str, object] = {},
        num_actors_to_launch: int = 1,
        needs_gpu: bool = False
    ):
        _check_ray_remote_args_dict(remote_args)
        self.remote_args = remote_args
        self.num_actors_to_launch = num_actors_to_launch
        self.needs_gpu = needs_gpu

        self._num_cpu_per_actor = None
        self._num_gpu_per_actor = None

    def inject_actor_resource_requirements(
        self,
        num_cpu_per_actor: float,
        num_gpu_per_actor: float
    ) -> None:
        '''
        This method must be called to set the appropriate resource per
        actor requirements before the resource per actor properties are
        accessed.
        '''
        self._num_cpu_per_actor = num_cpu_per_actor
        self._num_gpu_per_actor = num_gpu_per_actor

        if 'num_cpus' not in self.remote_args:
            self.remote_args['num_cpus'] = self._num_cpu_per_actor

        if 'num_gpus' not in self.remote_args:
            self.remote_args['num_gpus'] = self._num_gpu_per_actor

    @property
    def num_cpu_per_actor(self) -> float:
        if self._num_cpu_per_actor is None:
            raise RuntimeError(
                'Called RemoteActorConfig.num_cpu_per_actor before setting ' +
                'the number of CPUs per actor with '                         +
                'RemoteActorConfig.inject_actor_resource_requirements'
            )

        return self._num_cpu_per_actor

    @property
    def num_gpu_per_actor(self) -> float:
        if self._num_gpu_per_actor is None:
            raise RuntimeError(
                'Called RemoteActorConfig.num_gpu_per_actor before setting ' +
                'the number of GPUs per actor with '                         +
                'RemoteActorConfig.inject_actor_resource_requirements'
            )

        return self._num_gpu_per_actor


@dataclass
class ExperimentConfig:
    trainable: tune.Trainable
    param_space: Dict[str, object]

    use_tune: bool = True
    run_config: ray.air.config.RunConfig = None
    tune_config: tune.TuneConfig = None

    # Default to 1 if there are GPUs available to the run,
    # otherwise set to 0
    num_gpu_per_trial: float = float(torch.cuda.is_available())
    num_cpu_per_trial: float = 1.0

    placement_group_factory: tune.PlacementGroupFactory = None

    def __post_init__(self):
        if self.run_config is None:
            self.run_config = ray.air.config.RunConfig()

        if self.tune_config is None:
            self.tune_config = tune.TuneConfig()

        self.run_config.local_dir = LocalSettings.get_key('ray_results_dir')

        if self.use_tune:
            self.placement_group_factory = self._get_placement_group_factory()

    def inject_cli_args(self, cli_args: argparse.Namespace) -> None:
        self._inject_wandb(
            cli_args.experiment_name or cli_args.config_name,
            cli_args.entity_name,
            cli_args.wandb_log_dir
        )

        if self.run_config.name is None:
            self.run_config.name = cli_args.experiment_name

        if cli_args.num_training_iterations is not None:
            if self.run_config.stop is None:
                self.run_config.stop = {}

            self.run_config.stop['training_iteration'] = \
                cli_args.num_training_iterations

            if 'energy_model_config' in self.param_space:
                self.param_space[
                    'energy_model_config'
                ]['num_training_iterations'] = cli_args.num_training_iterations

        if cli_args.no_tune:
            self.use_tune = False

        if cli_args.verbosity is not None:
            self.run_config.verbose = cli_args.verbosity

        if cli_args.overrides is not None:
            self._inject_overrides(cli_args.overrides)

    def _inject_overrides(self, overrides: List[str]) -> None:
        for override in overrides:
            key_path, new_val = override.split('=')

            if _is_str_float(new_val):
                new_val = float(new_val)
            elif _is_str_int(new_val):
                new_val = int(new_val)

            key_path = key_path.split('.')
            try:
                dict_to_change = self.param_space
                for key in key_path[:-1]:
                    dict_to_change = dict_to_change[key]

                dict_to_change[key_path[-1]] = new_val

            except KeyError as e:
                print(f'Could not find key {key_path} in param_space')


    def _get_placement_group_factory(self) -> None:
        if not self.use_tune:
            return None

        actor_confs = list(_get_actor_confs(self.param_space))
        num_gpu_actors = sum(map(
            lambda x: x.needs_gpu * x.num_actors_to_launch,
            actor_confs
        ))

        cpu_per_actor = self.num_cpu_per_trial / (len(actor_confs) + 1)
        gpu_per_actor = self.num_gpu_per_trial / (num_gpu_actors + 1)

        def inject_rsrc_reqs(cfg: RemoteActorConfig) -> None:
            cfg.inject_actor_resource_requirements(cpu_per_actor, gpu_per_actor)

        list(map(inject_rsrc_reqs, actor_confs))

        gpu_actor_resource_reqs = [
            {'CPU': cpu_per_actor, 'GPU': gpu_per_actor}
            for _ in range(num_gpu_actors + 1)
        ]

        num_cpu_only_actors = len(actor_confs) - num_gpu_actors
        cpu_actor_resource_reqs = [
            {'CPU': cpu_per_actor}
            for _ in range(num_cpu_only_actors)
        ]

        return tune.PlacementGroupFactory(
            gpu_actor_resource_reqs + cpu_actor_resource_reqs
        )





_SLURM_JOB_CPUS_FILENAME = '/sys/fs/cgroup/cpuset/slurm/uid_%s/job_%s/cpuset.cpus'
_PY_FILE_SUFFIX = '.py'
_NOT_FOUND_ERROR_MSG_PRE = "No module named '%s'"

def parse_args() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", default=None)
    parser.add_argument("-t", "--entity_name", default=None)
    parser.add_argument("-c", "--config_name", default=None)
    parser.add_argument("-w", "--wandb_log_dir", default=None)
    parser.add_argument("-n", "--num_training_iterations", default=2500, type=int)
    parser.add_argument("-v", "--verbosity", choices=range(3), default=None, type=int)
    parser.add_argument("-o", "--overrides", nargs="*")
    parser.add_argument("--no_tune", action="store_true")

    return parser.parse_args(), parser


def _get_placement_group_factory(self) -> None:
    actor_confs = list(_get_actor_confs(self.param_space))
    num_gpu_actors = sum(map(
        lambda x: x.needs_gpu * x.num_actors_to_launch,
        actor_confs
    ))

    cpu_per_actor = self.num_cpu_per_trial / (len(actor_confs) + 1)
    gpu_per_actor = self.num_gpu_per_trial / (num_gpu_actors + 1)

    def inject_rsrc_reqs(cfg: RemoteActorConfig) -> None:
        cfg.inject_actor_resource_requirements(cpu_per_actor, gpu_per_actor)

    list(map(inject_rsrc_reqs, actor_confs))

    gpu_actor_resource_reqs = [
        {'CPU': cpu_per_actor, 'GPU': gpu_per_actor}
        for _ in range(num_gpu_actors + 1)
    ]

    num_cpu_only_actors = len(actor_confs) - num_gpu_actors
    cpu_actor_resource_reqs = [
        {'CPU': cpu_per_actor}
        for _ in range(num_cpu_only_actors)
    ]

    return tune.PlacementGroupFactory(
        gpu_actor_resource_reqs + cpu_actor_resource_reqs
    )
self.placement_group_factory = self._get_placement_group_factory()


    num_gpu_per_trial: float = float(torch.cuda.is_available())
    num_cpu_per_trial: float = 1.0


def _get_config_module_names_recurse(curr_path: Path) -> Iterator[str]:
    path_parts = curr_path.parts
    config_dir_idx = list(filter(
        lambda i: path_parts[i] == "configs",
        range(len(path_parts))
    ))[0]

    yield '.'.join(["gfn_exploration.configs", *path_parts[config_dir_idx + 1:]])

    for child in curr_path.iterdir():
        if child.is_dir() and child.name != '__pycache__':
            yield from _get_config_module_names_recurse(child)


def _get_config_module_names() -> Iterator[str]:
    package_dir = Path(__file__).parent
    yield from _get_config_module_names_recurse(package_dir / 'configs')


def get_config_module(args: argparse.Namespace) -> 'module':
    config_name = args.config_name
    if config_name is None:
        return None

    if config_name[-len(_PY_FILE_SUFFIX):] == _PY_FILE_SUFFIX:
        config_name = config_name[:-len(_PY_FILE_SUFFIX)]

    ex, conf_paths = None, list(_get_config_module_names())
    for conf_path in conf_paths:
        try:
            module_name = ".".join([conf_path, config_name])
            return importlib.import_module(module_name)
        except ImportError as e:
            not_found_msg = _NOT_FOUND_ERROR_MSG_PRE % module_name
            if e.msg == not_found_msg:
                ex = e
                continue
            else:
                raise

    if ex is not None:
        raise ex


def get_config(config_module: 'module') -> ExperimentConfig:
    try:
        return config_module.CONFIG
    except AttributeError:
        raise AttributeError(
            "The config in a config file should be stored in the "
            + "variable CONFIG, otherwise it cannot be found."
        )


def get_num_cpus() -> int:
    '''
    In a typical slurm allocation we only have access to a subset of the
    current node's CPUs.  Allocating ray with more CPUs than we actually
    have naturally leads to massive slowdowns in code performance, so
    we should avoid this.

    For any job allocation, Slurm writes a file to /sys listing the
    CPUs on the node which belong to the allocation.  If the code is
    being run on a slurm allocation this function reads the slurm CPU file
    and returns the number of CPUs allocated for the job.  If not on a slurm
    allocation, ray is initialized by default with the number of CPUs available
    on the system.
    '''
    if 'SLURM_JOB_ID' not in os.environ:
        return os.cpu_count()

    uid, slurm_job_id = os.getuid(), os.environ['SLURM_JOB_ID']
    fname = Path(_SLURM_JOB_CPUS_FILENAME % (uid, slurm_job_id))

    num_cpus = 0
    with open(fname, 'r') as f:
        line = f.read().replace('\n', '')
        for substr in line.split(','):
            if '-' not in substr:
                num_cpus += 1
                continue

            cpu_nums = list(map(int, substr.split('-')))
            num_cpus += cpu_nums[1] - cpu_nums[0] + 1

    return num_cpus


def launch_run_with_tune(config: ExperimentConfig) -> None:
    # If we're using tune and the trial is going to launch
    # other actors we need to give tune a placement group factory
    # to tell it how resource management will be done.  By default
    # tune allocates all resources to the trial's placement group,
    # and when the trial tries to launch new actors (which are necessarily
    # part of the trial's placement group) ray will say no resources
    # are available in the placement group.  By giving a placement group
    # factory, we avoid ray artificially running out of resources for an
    # already allocated trial.
    trainable = config.trainable
    if config.placement_group_factory is not None:
        trainable = tune.with_resources(
            trainable,
            resources=config.placement_group_factory
        )

    tuner = tune.Tuner(
        trainable,
        run_config=config.run_config,
        tune_config=config.tune_config,
        param_space=config.param_space
    )

    tuner.fit()



def main() -> None:
    cli_args, parser = parse_args()

    # config_module = get_config_module(cli_args)
    # if not config_module:
    #     print('Need config to be specified to run this script!')
    #     parser.print_help()

    #     return

    #config = get_config(config_module)
    #config.inject_cli_args(cli_args)

    ray.init(
        num_cpus=get_num_cpus(),
        num_gpus=torch.cuda.device_count(),
    )

    config = = ExperimentConfig(
    trainable=EnergyBasedModelLearner,
    param_space=_PARAM_SPACE,
    num_gpu_per_trial=0.5,
    tune_config=tune.TuneConfig(
        num_samples=-1,
        search_alg=OptunaSearch(metric='gflownet_negative_log_likelihood', mode='min'),
        scheduler=ASHAScheduler(
            time_attr='training_iteration',
            metric='gflownet_negative_log_likelihood',
            mode='min',
            grace_period=1000,
            max_t=10000
        )
    )
)

    #launch_run_with_tune(config)

    trainable = config.trainable
    if config.placement_group_factory is not None:
        trainable = tune.with_resources(
            trainable,
            resources=config.placement_group_factory
        )

    tuner = tune.Tuner(
        trainable,
        run_config=config.run_config,
        tune_config=config.tune_config,
        param_space=config.param_space
    )

    tuner.fit()

if __name__ == "__main__":
    main()