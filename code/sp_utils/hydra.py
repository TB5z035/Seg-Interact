import pyrootutils
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


__all__ = ['init_config']


def init_config(config_name='train.yaml', overrides=[]):
    GlobalHydra.instance().clear()
    pyrootutils.setup_root(".", pythonpath=True)
    with initialize(version_base='1.2', config_path="../../sp_configs"):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg