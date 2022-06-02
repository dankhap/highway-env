
import math
import tempfile
import numpy as np
import gym

from rlbit.common.logger import configure
from rlbit.envs.env_builder import load_envs_syncvectorenv
from rlbit.trainers.ppo_clean_rl.ppo_trainer import PPOTrainer
from rlbit.trainers.settings import ExperimentSettings
from rlbit.trainers.trainer import Trainer
from rlbit.common.torch import set_torch_config
from rlbit.trainers.trainer_controller import TrainerController

def check_environment_trains(env, config):

    # Create controller and begin training.
    with tempfile.TemporaryDirectory() as tmpdir:
        run_id = "id"
        print(f"using {tmpdir} for tmp dir")
        # seed = 1337 if training_seed is None else training_seed
        stats_manager = configure(tmpdir + "/rlbit_tests", ["stdout", "csv", "tensorboard"])
        trainer_controller = TrainerController(
                            config=config,
                            output_path=tmpdir,
                            load_model=False,
                            seed=config.seed,
                            logger=stats_manager,
                            run_id=run_id)

        # Begin training
        trainer_controller.start_learning(env)
        # stats should be taken from the logger
        assert  np.mean(np.array(env.envs[0].episode_returns[-10:])) == 500.0
        print("finished test!")
    


if __name__ == '__main__':
    with open("scripts/highway_rlbit_config.yaml") as f:
        content = f.read()
        config = ExperimentSettings.parse_raw(content)
    env = load_envs_syncvectorenv(config=config)
    set_torch_config(config.torch)
    check_environment_trains(env, config)
    
    pass
