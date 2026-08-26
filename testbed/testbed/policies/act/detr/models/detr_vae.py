# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from testbed.policies.act.camera_role_encoding import (
    CameraRoleEncoding,
    resolve_camera_role_encoding_config,
)

from ...goal_effect import GoalEffectHead
from .backbone import build_backbone
from .transformer import TransformerEncoder, TransformerEncoderLayer, build_transformer


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class TemporalFeatureMixer(nn.Module):
    """Causal, spatially-preserving mixer for a short feature history.

    The input history is ordered from oldest to newest.  A depthwise temporal
    convolution keeps the parameter count small (one kernel per feature
    channel) and is initialized to select the newest frame exactly.  This is
    intentional: an opt-in temporal checkpoint loaded from an old single-frame
    checkpoint starts with the old policy's image path rather than an
    uncalibrated temporal average.
    """

    def __init__(self, channels: int, history_steps: int):
        super().__init__()
        if int(channels) <= 0:
            raise ValueError("TemporalFeatureMixer channels must be positive")
        if int(history_steps) <= 0:
            raise ValueError("TemporalFeatureMixer history_steps must be positive")
        self.channels = int(channels)
        self.history_steps = int(history_steps)
        self.proj = nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=self.history_steps,
            groups=self.channels,
            bias=True,
        )
        with torch.no_grad():
            self.proj.weight.zero_()
            # Conv1d is a cross-correlation: the last kernel element consumes
            # the newest (right-most) frame in an oldest-to-newest history.
            self.proj.weight[:, 0, -1] = 1.0
            self.proj.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Mix ``(B, T, C, H, W)`` into ``(B, C, H, W)``."""

        if features.ndim != 5:
            raise ValueError(
                "TemporalFeatureMixer expects (batch, history, channels, height, width), "
                f"got shape {tuple(features.shape)}"
            )
        batch, history, channels, height, width = features.shape
        if history != self.history_steps:
            raise ValueError(
                "TemporalFeatureMixer history length mismatch: "
                f"expected {self.history_steps}, got {history}"
            )
        if channels != self.channels:
            raise ValueError(
                "TemporalFeatureMixer channel mismatch: "
                f"expected {self.channels}, got {channels}"
            )

        # Treat every spatial location as an independent temporal sequence.
        # The output has a single temporal position, then is restored to the
        # feature-map layout consumed by the existing DETR transformer.
        sequence = (
            features.permute(0, 3, 4, 2, 1)
            .contiguous()
            .reshape(batch * height * width, channels, history)
        )
        mixed = self.proj(sequence).squeeze(-1)
        return (
            mixed.reshape(batch, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        robot_state_dim,
        action_dim,
        num_queries,
        camera_names,
        vision_feature_scale=1.0,
        proprio_feature_scale=1.0,
        intent_dim=0,
        goal_effect_config=None,
        action_state_effort_config=None,
        effective_action_config=None,
        temporal_input_config=None,
        camera_role_encoding_config=None,
        condition_action_loss_config=None,
    ):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            robot_state_dim: low-dimensional robot state dimension fed to the policy
            action_dim: action dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.vision_feature_scale = float(vision_feature_scale)
        self.proprio_feature_scale = float(proprio_feature_scale)
        temporal_cfg = dict(temporal_input_config or {})
        self.temporal_input_enabled = bool(temporal_cfg.get("enabled", False))
        self.temporal_history_steps = int(temporal_cfg.get("history_steps", 1))
        if self.temporal_input_enabled and self.temporal_history_steps <= 0:
            raise ValueError("temporal_input.history_steps must be positive")
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        camera_role_cfg = resolve_camera_role_encoding_config(
            camera_role_encoding_config,
            camera_names=camera_names,
        )
        self.camera_role_encoding = (
            CameraRoleEncoding(
                hidden_dim=hidden_dim,
                camera_names=camera_names,
                config=camera_role_cfg,
            )
            if camera_role_cfg["enabled"]
            else None
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.intent_head = nn.Linear(hidden_dim, int(intent_dim)) if int(intent_dim) > 0 else None
        state_cfg = dict(action_state_effort_config or {})
        self.action_state_effort_enabled = bool(state_cfg.get("enabled", False))
        self.action_state_head = (
            nn.Linear(hidden_dim, 4 * int(state_cfg.get("state_count", 5)))
            if self.action_state_effort_enabled
            else None
        )
        effective_cfg = dict(effective_action_config or {})
        self.effective_action_enabled = bool(effective_cfg.get("enabled", False))
        self.effective_action_phase_head = (
            nn.Linear(hidden_dim, 4 * 3)
            if self.effective_action_enabled
            else None
        )
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(robot_state_dim, hidden_dim)
            self.temporal_feature_mixer = (
                TemporalFeatureMixer(hidden_dim, self.temporal_history_steps)
                if self.temporal_input_enabled
                else None
            )
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(robot_state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
            self.temporal_feature_mixer = None
            if self.temporal_input_enabled:
                raise ValueError("temporal_input requires image-backed ACT")

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(robot_state_dim, hidden_dim)  # project robot state to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

        # The goal/effect branch is strictly auxiliary.  Its context is built
        # from the current observation before the action-dependent ACT latent,
        # so future labels cannot leak through the training-time action
        # encoder.  The zero-initialised residual lets a checkpoint-initialised
        # continuous action head start with identical outputs and learn the
        # multi-task representation only through the opt-in objective.
        goal_cfg = dict(goal_effect_config or {})
        self.goal_effect_enabled = bool(goal_cfg.get("enabled", False))
        if self.goal_effect_enabled:
            horizons = tuple(int(item) for item in goal_cfg.get("horizons", (4, 8, 20)))
            context_input_dim = hidden_dim * (1 + len(camera_names))
            self.goal_context_proj = nn.Sequential(
                nn.Linear(context_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.goal_effect_head = GoalEffectHead(
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                num_queries=num_queries,
                horizons=horizons,
            )
            self.action_context_residual = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.action_context_residual.weight)
            nn.init.zeros_(self.action_context_residual.bias)
        else:
            self.goal_context_proj = None
            self.goal_effect_head = None
            self.action_context_residual = None

        condition_action_cfg = dict(condition_action_loss_config or {})
        self.condition_action_enabled = bool(condition_action_cfg.get("enabled", False))
        self.condition_action_head = (
            nn.Sequential(
                nn.Linear(num_queries * action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
            if self.condition_action_enabled
            else None
        )

    def _extract_camera_features(self, image, *, batch_cameras):
        """Run the shared backbone sequentially or once for all cameras."""
        if not batch_cameras:
            all_cam_features = []
            all_cam_pos = []
            for camera_index, _camera_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, camera_index])
                all_cam_features.append(
                    self.input_proj(features[0]) * self.vision_feature_scale
                )
                all_cam_pos.append(pos[0])
            if self.camera_role_encoding is not None:
                all_cam_pos = [
                    position
                    + self.camera_role_encoding(camera_index).to(
                        dtype=position.dtype,
                        device=position.device,
                    )
                    for camera_index, position in enumerate(all_cam_pos)
                ]
            return all_cam_features, all_cam_pos

        batch_size = image.shape[0]
        camera_count = len(self.camera_names)
        camera_images = image[:, :camera_count].reshape(
            batch_size * camera_count,
            *image.shape[2:],
        )
        features, pos = self.backbones[0](camera_images)
        projected = self.input_proj(features[0]) * self.vision_feature_scale
        projected = projected.reshape(
            batch_size,
            camera_count,
            *projected.shape[1:],
        )
        all_cam_features = [projected[:, cam_id] for cam_id in range(camera_count)]

        position = pos[0]
        if position.shape[0] == batch_size * camera_count:
            position = position.reshape(
                batch_size,
                camera_count,
                *position.shape[1:],
            )
            all_cam_pos = [position[:, cam_id] for cam_id in range(camera_count)]
        elif position.shape[0] == 1:
            all_cam_pos = [position for _ in range(camera_count)]
        else:
            raise ValueError(
                "Batched ACT backbone returned an unsupported position batch "
                f"dimension {position.shape[0]}; expected 1 or "
                f"{batch_size * camera_count}."
            )
        if self.camera_role_encoding is not None:
            all_cam_pos = [
                position
                + self.camera_role_encoding(camera_index).to(
                    dtype=position.dtype,
                    device=position.device,
                )
                for camera_index, position in enumerate(all_cam_pos)
            ]
        return all_cam_features, all_cam_pos

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width for single-frame ACT;
               batch, history, num_cam, channel, height, width when the
               opt-in temporal_input path is enabled.  History is causal and
               ordered oldest to newest.
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        goal_context = None
        if self.backbones is not None:
            if self.temporal_input_enabled:
                if image.ndim != 6:
                    raise ValueError(
                        "temporal_input requires causal image shape "
                        "(batch, history, num_cam, channel, height, width), "
                        f"got shape {tuple(image.shape)}"
                    )
                if image.shape[1] != self.temporal_history_steps:
                    raise ValueError(
                        "temporal_input history length mismatch: "
                        f"expected {self.temporal_history_steps}, got {image.shape[1]}"
                    )
                if image.shape[2] != len(self.camera_names):
                    raise ValueError(
                        "temporal_input camera count mismatch: "
                        f"expected {len(self.camera_names)}, got {image.shape[2]}"
                    )
            # Image observation features and position embeddings
            batch_single_frame_cameras = not self.temporal_input_enabled and not is_training
            if batch_single_frame_cameras:
                all_cam_features, all_cam_pos = self._extract_camera_features(
                    image,
                    batch_cameras=True,
                )
                camera_items = ()
            else:
                all_cam_features = []
                all_cam_pos = []
                camera_items = enumerate(self.camera_names)
            for cam_id, cam_name in camera_items:
                if self.temporal_input_enabled:
                    batch, history = image.shape[:2]
                    camera_image = image[:, :, cam_id]
                    flat_camera_image = camera_image.reshape(
                        batch * history, *camera_image.shape[2:]
                    )
                    features, pos = self.backbones[0](flat_camera_image) # HARDCODED
                    features = features[0] # take the last layer feature
                    feature_batch = features.shape[0]
                    if feature_batch != batch * history:
                        raise ValueError(
                            "backbone returned an unexpected temporal batch size: "
                            f"expected {batch * history}, got {feature_batch}"
                        )
                    projected = self.input_proj(features)
                    projected = projected.reshape(
                        batch,
                        history,
                        projected.shape[1],
                        projected.shape[2],
                        projected.shape[3],
                    )
                    features = self.temporal_feature_mixer(projected)
                    # Positional encoding is spatial-only in the existing
                    # DETR path.  Some encoders broadcast it as ``(1,C,H,W)``
                    # while learned encoders may return one copy per input
                    # image; in the latter case reshape the flattened
                    # ``B*T`` copies and select the newest frame.  The
                    # transformer consumes a broadcast spatial position map,
                    # so collapse an otherwise identical per-batch map to its
                    # first copy after selecting the newest frame.
                    pos = pos[0]
                    if pos.shape[0] == batch * history:
                        pos = pos.reshape(
                            batch,
                            history,
                            pos.shape[1],
                            pos.shape[2],
                            pos.shape[3],
                        )[:, -1]
                    if pos.shape[0] == batch and batch > 1:
                        pos = pos[:1]
                    elif pos.shape[0] not in (1, batch):
                        raise ValueError(
                            "backbone returned an unexpected temporal position batch size: "
                            f"got {pos.shape[0]} for batch={batch}, history={history}"
                        )
                else:
                    features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                    features = features[0] # take the last layer feature
                    pos = pos[0]
                    features = self.input_proj(features)
                if self.camera_role_encoding is not None:
                    role_encoding = self.camera_role_encoding(cam_id).to(
                        dtype=pos.dtype,
                        device=pos.device,
                    )
                    pos = pos + role_encoding
                all_cam_features.append(features * self.vision_feature_scale)
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos) * self.proprio_feature_scale
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]

            if self.goal_effect_enabled:
                camera_context = [feature.mean(dim=(2, 3)) for feature in all_cam_features]
                context_input = torch.cat([proprio_input, *camera_context], dim=1)
                goal_context = self.goal_context_proj(context_input)
                hs = hs + self.action_context_residual(goal_context).unsqueeze(1)
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        intent_logits = self.intent_head(hs) if self.intent_head is not None else None
        action_state_logits = (
            self.action_state_head(hs)
            if self.action_state_head is not None
            else None
        )
        effective_action_phase_logits = (
            self.effective_action_phase_head(hs)
            if self.effective_action_phase_head is not None
            else None
        )
        is_pad_hat = self.is_pad_head(hs)
        # Recompute the auxiliary effect head after the final action head so
        # it always consumes the exact continuous proposal returned to ACT.
        if self.goal_effect_enabled:
            if goal_context is None:
                raise RuntimeError("goal_effect requires image-backed ACT context")
            goal_effect_outputs = self.goal_effect_head(goal_context, a_hat)
        else:
            goal_effect_outputs = None
        condition_action_logits = (
            self.condition_action_head(a_hat.reshape(a_hat.shape[0], -1))
            if self.condition_action_head is not None
            else None
        )
        return (
            a_hat,
            is_pad_hat,
            [mu, logvar],
            intent_logits,
            goal_effect_outputs,
            action_state_logits,
            effective_action_phase_logits,
            condition_action_logits,
        )



# 未使用，没做单臂CNNMLP模型
class CNNMLP(nn.Module):
    def __init__(self, backbones, robot_state_dim, action_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            robot_state_dim: low-dimensional robot state dimension fed to the policy
            action_dim: action dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, action_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + robot_state_dim
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=action_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def _resolve_action_dim(equipment_model: str) -> int:
    return 4


def build(args):
    equipment_model = args.equipment_model if "equipment_model" in args else "real_excavator"
    explicit_state_dim = getattr(args, "state_dim", None)
    action_dim = _resolve_action_dim(equipment_model)
    robot_state_dim = int(explicit_state_dim) if explicit_state_dim is not None else action_dim

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        vision_feature_scale=getattr(args, "vision_feature_scale", 1.0),
        proprio_feature_scale=getattr(args, "proprio_feature_scale", 1.0),
        intent_dim=getattr(args, "intent_dim", 0),
        goal_effect_config=getattr(args, "goal_effect", None),
        action_state_effort_config=getattr(args, "action_state_effort", None),
        effective_action_config=getattr(args, "effective_action", None),
        temporal_input_config=getattr(args, "temporal_input", None),
        camera_role_encoding_config=getattr(args, "camera_role_encoding", None),
        condition_action_loss_config=getattr(args, "condition_action_loss", None),
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters / 1e6:.2f}M")

    return model

def build_cnnmlp(args):
    equipment_model = args.equipment_model if "equipment_model" in args else 'vx300s_bimanual'
    explicit_state_dim = getattr(args, "state_dim", None)
    action_dim = _resolve_action_dim(equipment_model)
    robot_state_dim = int(explicit_state_dim) if explicit_state_dim is not None else action_dim

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        robot_state_dim=robot_state_dim,
        action_dim=action_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters / 1e6:.2f}M")

    return model
