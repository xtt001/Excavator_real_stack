"""
HDF5 schema constants for the real-excavator testbed.

All data access should go through hdf5_io.py and use the constants here
instead of hardcoding field names. This branch is real-only: metadata names
describe machine data and adapter boundaries.

Schema v1.2 layout
------------------
/
├── metadata/                 group, scalar attrs
│   ├── schema_version        "1.2"
│   ├── is_real               bool
│   ├── platform              "real_excavator"
│   ├── task_name             str
│   ├── seed                  int
│   ├── param_version         str
│   ├── timestamp             str, ISO 8601
│   ├── control_hz            int
│   ├── dt                    float
│   ├── action_semantics      "normalized_teleop_cmd_v1"
│   ├── camera_names          comma-separated str
│   ├── image_format          "raw_rgb"
│   ├── action_order          "swing,boom,stick,bucket"
│   ├── qpos_order            "swing,boom,stick,bucket"
│   └── qvel_order            "swing,boom,stick,bucket"
├── observations/
│   ├── qpos                  (T, 4) float32, rad
│   ├── qvel                  (T, 4) float32, rad/s
│   ├── env_state             (T, M) float32, optional task state
│   └── images/<camera>       (T, H, W, 3) uint8 RGB
├── action                    (T, 4) float32, guard-filtered normalized command
├── rewards                   (T,) float32, optional
├── timestamps/
│   ├── step_id               (T,) int64
│   └── step_ns               (T,) int64, optional
├── action_source/
│   ├── type                  (T,) str
│   └── id                    (T,) str
└── diagnostics/              optional per-step controller/guard data
"""

# ── Schema version ────────────────────────────────────────────────────────────
SCHEMA_VERSION = "1.2"

# ── Group paths ───────────────────────────────────────────────────────────────
GRP_METADATA      = "metadata"
GRP_OBS           = "observations"
GRP_IMAGES        = "observations/images"
GRP_TIMESTAMPS    = "timestamps"
GRP_ACTION_SOURCE = "action_source"
GRP_DIAGNOSTICS   = "diagnostics"

# ── Dataset paths ─────────────────────────────────────────────────────────────
DS_QPOS    = "observations/qpos"
DS_QVEL    = "observations/qvel"
DS_ACTION  = "action"
DS_REWARDS = "rewards"
DS_ENV_STATE       = "observations/env_state"
DS_STEP_ID         = "timestamps/step_id"
DS_STEP_NS         = "timestamps/step_ns"
DS_ACTION_SRC_TYPE = "action_source/type"
DS_ACTION_SRC_ID   = "action_source/id"

# ── Diagnostic dataset paths ─────────────────────────────────────────────────
DS_DIAG_RAW_ACTION = "diagnostics/raw_action"
DS_DIAG_GUARD_TRIGGERED = "diagnostics/guard_triggered"
DS_DIAG_GUARD_REASON = "diagnostics/guard_reason"
DS_DIAG_CONTROLLER_ACK = "diagnostics/controller_ack"
DS_DIAG_CONTROLLER_FAULT_CODE = "diagnostics/controller_fault_code"
DS_DIAG_CONTROLLER_TIMESTAMP_NS = "diagnostics/controller_timestamp_ns"
DS_DIAG_COMMANDED_ACTION = "diagnostics/commanded_action"

# ── Metadata attribute names ─────────────────────────────────────────────────
ATTR_SCHEMA_VERSION = "schema_version"
ATTR_IS_REAL        = "is_real"
ATTR_PLATFORM       = "platform"
ATTR_TASK_NAME      = "task_name"
ATTR_SEED           = "seed"
ATTR_PARAM_VERSION  = "param_version"
ATTR_TIMESTAMP      = "timestamp"
ATTR_CONTROL_HZ     = "control_hz"
ATTR_DT             = "dt"
ATTR_ACTION_SEMANTICS = "action_semantics"
ATTR_CAMERA_NAMES   = "camera_names"
ATTR_IMAGE_FORMAT   = "image_format"
ATTR_PROTOCOL_VERSION = "protocol_version"
ATTR_EPISODE_ID     = "episode_id"
ATTR_OPERATOR_ID    = "operator_id"
ATTR_SESSION_ID     = "session_id"
ATTR_NOTES          = "notes"
ATTR_RECORD_CONFIG_PATH = "record_config_path"
ATTR_RECORD_CONFIG_YAML = "record_config_yaml"
ATTR_CAMERA_WIDTH   = "camera_width"
ATTR_CAMERA_HEIGHT  = "camera_height"
ATTR_CAMERA_FPS     = "camera_fps"
ATTR_CAMERA_ROW_ORDER = "camera_row_order"
ATTR_ACTION_ORDER   = "action_order"
ATTR_QPOS_ORDER     = "qpos_order"
ATTR_QVEL_ORDER     = "qvel_order"
ATTR_ENV_STATE_ORDER = "env_state_order"
ATTR_TELEOP_INPUT   = "teleop_input"
ATTR_DEADZONE       = "deadzone"
ATTR_SCALE          = "scale"
ATTR_LIMIT          = "limit"
ATTR_AXIS_MAP       = "axis_map"
ATTR_JOYSTICK_IDS   = "joystick_ids"
ATTR_INVERT         = "invert"
ATTR_KEY_SPEED      = "key_speed"
ATTR_RESPONSE_PROFILE_ENABLED = "response_profile_enabled"
ATTR_RESPONSE_PROFILE_ATTACK_RATE = "response_profile_attack_rate"
ATTR_RESPONSE_PROFILE_RELEASE_RATE = "response_profile_release_rate"
ATTR_RESPONSE_PROFILE_RECENTER_RATE = "response_profile_recenter_rate"
ATTR_RESPONSE_PROFILE_EXPONENT = "response_profile_exponent"
ATTR_QPOS_UNITS = "qpos_units"
ATTR_QVEL_UNITS = "qvel_units"
ATTR_QPOS_SOURCE = "qpos_source"
ATTR_QVEL_SOURCE = "qvel_source"
ATTR_HYDRAULIC_CYLINDER_AVAILABLE = "hydraulic_cylinder_available"

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_CONTROL_HZ       = 50
DEFAULT_DT               = 0.02
DEFAULT_ACTION_SEMANTICS = "normalized_teleop_cmd_v1"
DEFAULT_IMAGE_FORMAT     = "raw_rgb"
DEFAULT_PLATFORM         = "real_excavator"


def image_ds(cam_name: str) -> str:
    """Return the HDF5 dataset path for a camera stream."""

    return f"observations/images/{cam_name}"
