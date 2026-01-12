from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import bxi_elf3_flat_env_cfg, bxi_elf3_rough_env_cfg
from .rl_cfg import bxi_elf3_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-BXI-ELF3",
  env_cfg=bxi_elf3_rough_env_cfg(),
  play_env_cfg=bxi_elf3_rough_env_cfg(play=True),
  rl_cfg=bxi_elf3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-BXI-ELF3",
  env_cfg=bxi_elf3_flat_env_cfg(),
  play_env_cfg=bxi_elf3_flat_env_cfg(play=True),
  rl_cfg=bxi_elf3_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
