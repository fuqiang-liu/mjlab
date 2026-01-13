"""BXI ELF3 velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  ELF3_ACTION_SCALE,
  get_elf3_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from .velocity_env_cfg import make_velocity_env_cfg


def bxi_elf3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create BXI ELF3 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 45

  cfg.scene.entities = {"robot": get_elf3_robot_cfg()}

  site_names = ("l_foot", "r_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("l", "r") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(l_ankle_x_link|r_ankle_x_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ELF3_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*hip_y.*": 0.3,
    r".*hip_x.*": 0.05,
    r".*hip_z.*": 0.05,
    r".*knee.*": 0.05,
    r".*ankle_y.*": 0.35,
    r".*ankle_x.*": 0.05,
    # Waist.
    r".*waist_z.*": 0.05,
    r".*waist_x.*": 0.05,
    r".*waist_y.*": 0.1,
    # Arms.
    r".*shoulder_y.*": 0.05,
    r".*shoulder_x.*": 0.05,
    r".*shoulder_z.*": 0.05,
    r".*elbow.*": 0.05,
    r".*wrist.*": 0.05,
    }
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_y.*": 0.3,
    r".*hip_x.*": 0.15,
    r".*hip_z.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_y.*": 0.25,
    r".*ankle_x.*": 0.1,
    # Waist.
    r".*waist_z.*": 0.2,
    r".*waist_x.*": 0.08,
    r".*waist_y.*": 0.1,
    # Arms.
    r".*shoulder_y.*": 0.15,
    r".*shoulder_x.*": 0.15,
    r".*shoulder_z.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_y.*": 0.5,
    r".*hip_x.*": 0.2,
    r".*hip_z.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_y.*": 0.35,
    r".*ankle_x.*": 0.15,
    # Waist.
    r".*waist_z.*": 0.3,
    r".*waist_x.*": 0.08,
    r".*waist_y.*": 0.2,
    # Arms.
    r".*shoulder_y.*": 0.5,
    r".*shoulder_x.*": 0.2,
    r".*shoulder_z.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.2

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def bxi_elf3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create BXI ELF3 flat terrain velocity configuration."""
  cfg = bxi_elf3_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
