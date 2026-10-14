from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence
import os
import shutil
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .tokenizer import EOS, PAD, ByteFallbackTokenizer


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones((cfg.context, cfg.context), dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        self.norm_ff = nn.LayerNorm(cfg.dim)
        self.ff = nn.Sequential(
            nn.Linear(cfg.dim, cfg.ffn),
            nn.GELU(),
            nn.Linear(cfg.ffn, cfg.dim),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        n = x.size(1)
        causal = self.causal_mask[:n, :n]
        y = self.norm_attn(x)
        y, _ = self.attn(
            y,
            y,
            y,
            attn_mask=causal,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        y = x + y
        return y + self.ff(self.norm_ff(y))


CHECKPOINT_SCHEMA_VERSION = 10


class AletheiaModel(nn.Module):
    """A randomly initialized predictive learner with a trainable recurrent state.

    The model contains no downloaded weights, pretrained vocabulary, or external
    embedding system. Every parameter starts from a small random initialization
    and is shaped only by the training data supplied to Aletheia.
    """

    def __init__(self, cfg: ModelConfig, action_names: list[str], tokenizer=None):
        super().__init__()
        if not action_names:
            raise ValueError("at least one action is required")
        if cfg.dim % cfg.heads != 0:
            raise ValueError("model dim must be divisible by the attention head count")
        self.cfg = cfg
        self.tokenizer = tokenizer if tokenizer is not None else ByteFallbackTokenizer()
        if int(cfg.vocab_size) != int(self.tokenizer.vocab_size):
            raise ValueError(f"model vocab_size={cfg.vocab_size} does not match tokenizer vocab_size={self.tokenizer.vocab_size}")
        self.action_names = list(action_names)
        self.action_count = len(action_names)

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=PAD)
        self.position_embedding = nn.Embedding(cfg.context, cfg.dim)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.layers)])
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.action_embedding = nn.Embedding(self.action_count, cfg.dim)
        self.error_encoder = nn.Sequential(
            nn.Linear(1, cfg.dim),
            nn.SiLU(),
            nn.Linear(cfg.dim, cfg.dim),
        )
        self.state_input = nn.Linear(cfg.recurrent_dim, cfg.dim)
        self.recurrent = nn.GRUCell(cfg.dim * 3, cfg.recurrent_dim)
        self.state_norm = nn.LayerNorm(cfg.recurrent_dim)
        policy_mid = max(128, cfg.dim // 2)
        self.perception_vision = nn.Sequential(
            nn.Linear(cfg.perception_dim, cfg.recurrent_dim),
            nn.LayerNorm(cfg.recurrent_dim),
            nn.SiLU(),
        )
        self.perception_audio = nn.Sequential(
            nn.Linear(cfg.perception_dim, cfg.recurrent_dim),
            nn.LayerNorm(cfg.recurrent_dim),
            nn.SiLU(),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(cfg.recurrent_dim),
            nn.Linear(cfg.recurrent_dim, policy_mid),
            nn.SiLU(),
            nn.Linear(policy_mid, self.action_count),
        )
        if cfg.recurrent_dim % cfg.world_attention_heads != 0:
            raise ValueError("recurrent dim must be divisible by world attention head count")
        if cfg.world_attention_heads < 1:
            raise ValueError("world attention head count must be positive")
        # FINAL17 cognitive core: every modality is projected into the same learned
        # latent space, then selected by state-conditioned attention. There are no
        # semantic labels here; the model learns which stream is useful from data.
        self.text_to_world = nn.Linear(cfg.dim, cfg.recurrent_dim)
        self.modality_embedding = nn.Embedding(3, cfg.recurrent_dim)
        self.world_attention = nn.MultiheadAttention(
            cfg.recurrent_dim, cfg.world_attention_heads, dropout=cfg.dropout, batch_first=True
        )
        self.world_query = nn.Linear(cfg.recurrent_dim, cfg.recurrent_dim)
        self.world_fusion_gate = nn.Sequential(
            nn.Linear(cfg.recurrent_dim * 2, cfg.recurrent_dim),
            nn.SiLU(),
            nn.Linear(cfg.recurrent_dim, cfg.recurrent_dim),
            nn.Sigmoid(),
        )
        self.world_observation_head = nn.Sequential(
            nn.LayerNorm(cfg.recurrent_dim),
            nn.Linear(cfg.recurrent_dim, int(cfg.world_model_hidden)),
            nn.SiLU(),
            nn.Linear(int(cfg.world_model_hidden), cfg.dim),
        )
        transition_in = cfg.recurrent_dim + (2 * cfg.dim)
        self.world_transition = nn.Sequential(
            nn.LayerNorm(transition_in),
            nn.Linear(transition_in, int(cfg.world_model_hidden)),
            nn.SiLU(),
            nn.Linear(int(cfg.world_model_hidden), cfg.recurrent_dim),
        )
        self.world_uncertainty_head = nn.Sequential(
            nn.LayerNorm(cfg.recurrent_dim),
            nn.Linear(cfg.recurrent_dim, 1),
        )
        self._initialize_parameters()
        with torch.no_grad():
            nn.init.normal_(self.perception_vision[0].bias, mean=0.0, std=0.02)
            nn.init.normal_(self.perception_audio[0].bias, mean=0.0, std=0.02)

    def _initialize_parameters(self) -> None:
        """Use stable small random initialization; no pretrained weights are used."""
        with torch.no_grad():
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
            self.token_embedding.weight[PAD].zero_()

        for module in self.modules():
            if module is self or module in {
                self.token_embedding,
                self.position_embedding,
                self.action_embedding,
                self.lm_head,
            }:
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)
            elif isinstance(module, nn.GRUCell):
                nn.init.xavier_uniform_(module.weight_ih)
                nn.init.orthogonal_(module.weight_hh)
                nn.init.zeros_(module.bias_ih)
                nn.init.zeros_(module.bias_hh)

    def _modality_projector(self, modality: str) -> nn.Module:
        if modality == "vision":
            return self.perception_vision
        if modality == "audio":
            return self.perception_audio
        if modality == "text":
            return self.text_to_world
        raise ValueError(f"unknown modality: {modality}")

    def fuse_modalities(
        self,
        state: torch.Tensor,
        observations: dict[str, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse available sensory/text summaries into one learned world representation.

        Attention is conditioned on the current persistent state. The returned weights
        are useful as an empirical attention trace; they are not hand-coded importance
        values.
        """
        if state.ndim == 1:
            state = state.unsqueeze(0)
        tokens: list[torch.Tensor] = []
        keys = {"text": 0, "vision": 1, "audio": 2}
        for name, raw in observations.items():
            if raw is None:
                continue
            value = raw
            if value.ndim == 1:
                value = value.unsqueeze(0)
            projected = self._modality_projector(name)(value.to(device=state.device, dtype=state.dtype))
            if projected.size(0) == 1 and state.size(0) > 1:
                projected = projected.expand(state.size(0), -1)
            type_ids = torch.full(
                (projected.size(0),), keys[name], device=state.device, dtype=torch.long
            )
            projected = projected + self.modality_embedding(type_ids)
            tokens.append(projected)
        if not tokens:
            return state, torch.zeros(state.size(0), 0, device=state.device, dtype=state.dtype)
        sequence = torch.stack(tokens, dim=1)
        query = self.world_query(self.state_norm(state)).unsqueeze(1)
        attended, weights = self.world_attention(query, sequence, sequence, need_weights=True)
        attended = attended[:, 0]
        attention_trace = weights[:, 0, :]
        # MultiheadAttention applies dropout to returned weights during training,
        # so renormalize the trace before exposing it as an interpretable attention
        # signal. The learned computation itself is unchanged.
        attention_trace = attention_trace / attention_trace.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        gate = self.world_fusion_gate(torch.cat([state, attended], dim=-1))
        fused = self.state_norm(state + gate * attended)
        return fused, attention_trace

    def integrate_multimodal(
        self,
        state: torch.Tensor,
        *,
        text_summary: torch.Tensor | None = None,
        vision_embedding: torch.Tensor | None = None,
        audio_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fuse_modalities(
            state,
            {"text": text_summary, "vision": vision_embedding, "audio": audio_embedding},
        )

    def integrate_perception(
        self,
        embedding: torch.Tensor,
        state: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """Fold a learned sensory embedding into persistent recurrent state."""
        fused, _ = self.fuse_modalities(state, {modality: embedding})
        return fused

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        dtype = self.token_embedding.weight.dtype
        return torch.zeros(batch_size, self.cfg.recurrent_dim, device=device, dtype=dtype)

    def encode_sequence(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 2:
            raise ValueError("tokens must be a rank-1 or rank-2 tensor")
        if tokens.size(1) > self.cfg.context:
            tokens = tokens[:, -self.cfg.context:]
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        if state is not None:
            x = x + self.state_input(self.state_norm(state)).unsqueeze(1)
        padding_mask = tokens.eq(PAD)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled() and self.cfg.gradient_checkpointing:
                x = checkpoint(block, x, padding_mask, use_reentrant=False)
            else:
                x = block(x, padding_mask)
        x = self.final_norm(x)
        return self.lm_head(x), x

    def _action_error_vectors(
        self,
        state: torch.Tensor,
        previous_action: int | None,
        prediction_error: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous_action is None:
            action_vec = torch.zeros(state.size(0), self.cfg.dim, device=state.device, dtype=state.dtype)
        else:
            action = torch.full(
                (state.size(0),),
                int(previous_action),
                device=state.device,
                dtype=torch.long,
            )
            action_vec = self.action_embedding(action)
        if isinstance(prediction_error, torch.Tensor):
            values = prediction_error.to(device=state.device, dtype=state.dtype).reshape(-1, 1)
            if values.size(0) == 1 and state.size(0) > 1:
                values = values.expand(state.size(0), 1)
        else:
            values = torch.full((state.size(0), 1), float(prediction_error), device=state.device, dtype=state.dtype)
        return action_vec, self.error_encoder(values)

    def predict_next_state(
        self,
        state: torch.Tensor,
        previous_action: int | None,
        prediction_error: float | torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next latent state before the next observation is seen."""
        action_vec, error_vec = self._action_error_vectors(state, previous_action, prediction_error)
        delta = self.world_transition(torch.cat([self.state_norm(state), action_vec, error_vec], dim=-1))
        return self.state_norm(state + delta)

    def predict_next_observation(
        self,
        state: torch.Tensor,
        previous_action: int | None,
        prediction_error: float | torch.Tensor,
    ) -> torch.Tensor:
        return self.world_observation_head(
            self.predict_next_state(state, previous_action, prediction_error)
        )

    def world_prediction_loss(
        self, predicted_summary: torch.Tensor, target_summary: torch.Tensor
    ) -> torch.Tensor:
        predicted = F.normalize(predicted_summary, dim=-1)
        target = F.normalize(target_summary.detach(), dim=-1)
        return 0.5 * F.smooth_l1_loss(predicted, target) + 0.5 * (1.0 - (predicted * target).sum(dim=-1).mean())

    def transition_uncertainty(self, state: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.world_uncertainty_head(state)).squeeze(-1)

    def update_state(
        self,
        summary: torch.Tensor,
        state: torch.Tensor,
        previous_action: int | None,
        prediction_error: float | torch.Tensor,
    ) -> torch.Tensor:
        action_vec, error_vec = self._action_error_vectors(summary, previous_action, prediction_error)
        # Keep the persistent recurrent state in its stable/master dtype. CUDA
        # autocast may execute the GRUCell in FP16, but the state is carried
        # across trajectory steps and is stored in a FP32 tensor by default.
        # Casting here prevents mixed-precision assignment failures while
        # preserving AMP for the expensive model computation.
        updated_state = self.recurrent(
            torch.cat([summary, action_vec, error_vec], dim=-1), state
        )
        return updated_state.to(dtype=state.dtype)

    def step(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor,
        previous_action: int | None,
        prediction_error: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, hidden = self.encode_sequence(tokens, state)
        summary = hidden[:, -1]
        cognitive_state, _ = self.integrate_multimodal(state, text_summary=summary)
        next_state = self.update_state(summary, cognitive_state, previous_action, prediction_error)
        return logits, next_state, self.action_head(next_state), summary

    def action_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.state_norm(state))

    def language_loss(self, tokens: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.size(1) < 3:
            return torch.zeros((), device=tokens.device, requires_grad=True)
        logits, _ = self.encode_sequence(tokens[:, :-1], state)
        target = tokens[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size),
            target.reshape(-1),
            ignore_index=PAD,
        )

    def trajectory_loss(
        self,
        token_batches: list[torch.Tensor],
        previous_actions: list[int | None],
        previous_errors: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not token_batches:
            zero = torch.zeros((), device=self.token_embedding.weight.device, requires_grad=True)
            return zero, self.initial_state(1, zero.device)
        state = self.initial_state(1, token_batches[0].device)
        losses: list[torch.Tensor] = []
        world_losses: list[torch.Tensor] = []
        pending_prediction: torch.Tensor | None = None
        pending_uncertainty: torch.Tensor | None = None
        for idx, tokens in enumerate(token_batches):
            logits, hidden = self.encode_sequence(tokens, state)
            if tokens.size(1) >= 3:
                target_logits = logits[:, :-1]
                target = tokens[:, 1:]
                losses.append(
                    F.cross_entropy(
                        target_logits.reshape(-1, self.cfg.vocab_size),
                        target.reshape(-1),
                        ignore_index=PAD,
                    )
                )
            summary = hidden[:, -1]
            cognitive_state, _ = self.integrate_multimodal(state, text_summary=summary)
            if pending_prediction is not None:
                target = F.normalize(summary.detach(), dim=-1)
                predicted = F.normalize(pending_prediction, dim=-1)
                prediction_loss = self.world_prediction_loss(pending_prediction, summary)
                surprise_target = (1.0 - (predicted.detach() * target).sum(dim=-1)).clamp_min(0.0)
                uncertainty_loss = F.smooth_l1_loss(pending_uncertainty, surprise_target) if pending_uncertainty is not None else prediction_loss.new_zeros(())
                world_losses.append(prediction_loss + float(self.cfg.world_uncertainty_weight) * uncertainty_loss)
            action = previous_actions[idx]
            error = previous_errors[idx]
            pending_prediction = self.predict_next_observation(cognitive_state, action, error)
            pending_uncertainty = self.transition_uncertainty(cognitive_state)
            state = self.update_state(summary, cognitive_state, action, error)
        language = torch.stack(losses).mean() if losses else torch.zeros((), device=state.device, requires_grad=True)
        world = torch.stack(world_losses).mean() if world_losses else torch.zeros((), device=state.device)
        loss = language + float(self.cfg.world_prediction_weight) * world
        return loss, state

    @staticmethod
    def _pad_token_batch(tokens: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
        lengths = [int(item.size(1)) for item in tokens]
        width = max(lengths, default=2)
        result = torch.full((len(tokens), width), PAD, dtype=torch.long, device=device)
        for idx, item in enumerate(tokens):
            values = item.to(device=device, dtype=torch.long).reshape(1, -1)
            result[idx, : values.size(1)] = values[0]
        return result

    def trajectory_loss_batched(
        self,
        sequences: Sequence[tuple[list[torch.Tensor], list[int | None], list[float]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Train several recurrent trajectories together while preserving each trajectory's state."""
        if not sequences:
            zero = torch.zeros((), device=self.token_embedding.weight.device, requires_grad=True)
            return zero, self.initial_state(1, zero.device)
        device = sequences[0][0][0].device
        batch_size = len(sequences)
        state = self.initial_state(batch_size, device)
        losses: list[torch.Tensor] = []
        world_losses: list[torch.Tensor] = []
        pending_predictions: dict[int, torch.Tensor] = {}
        pending_uncertainties: dict[int, torch.Tensor] = {}
        max_steps = max((len(item[0]) for item in sequences), default=0)

        for step_index in range(max_steps):
            active_indices = [idx for idx, item in enumerate(sequences) if step_index < len(item[0])]
            if not active_indices:
                continue
            token_batch = self._pad_token_batch(
                [sequences[idx][0][step_index] for idx in active_indices], device
            )
            active_state = state[active_indices]
            logits, hidden = self.encode_sequence(token_batch, active_state)
            if token_batch.size(1) >= 3:
                target_logits = logits[:, :-1]
                target = token_batch[:, 1:]
                losses.append(
                    F.cross_entropy(
                        target_logits.reshape(-1, self.cfg.vocab_size),
                        target.reshape(-1),
                        ignore_index=PAD,
                    )
                )
            valid_lengths = torch.tensor(
                [int(sequences[idx][0][step_index].size(1)) for idx in active_indices],
                device=hidden.device,
                dtype=torch.long,
            ).clamp(min=1, max=hidden.size(1))
            row_indices = torch.arange(hidden.size(0), device=hidden.device)
            summaries = hidden[row_indices, valid_lengths - 1]
            cognitive_states, _ = self.integrate_multimodal(active_state, text_summary=summaries)
            for local_idx, sequence_idx in enumerate(active_indices):
                pending = pending_predictions.get(sequence_idx)
                if pending is not None:
                    target = F.normalize(summaries[local_idx : local_idx + 1].detach(), dim=-1)
                    predicted = F.normalize(pending, dim=-1)
                    prediction_loss = self.world_prediction_loss(pending, summaries[local_idx : local_idx + 1])
                    surprise_target = (1.0 - (predicted.detach() * target).sum(dim=-1)).clamp_min(0.0)
                    uncertainty = pending_uncertainties.get(sequence_idx)
                    uncertainty_loss = F.smooth_l1_loss(uncertainty, surprise_target) if uncertainty is not None else prediction_loss.new_zeros(())
                    world_losses.append(prediction_loss + float(self.cfg.world_uncertainty_weight) * uncertainty_loss)
            actions = [sequences[idx][1][step_index] for idx in active_indices]
            errors = torch.tensor(
                [sequences[idx][2][step_index] for idx in active_indices],
                device=device,
                dtype=summaries.dtype,
            )
            for local_idx, sequence_idx in enumerate(active_indices):
                action = actions[local_idx]
                error_value = errors[local_idx : local_idx + 1]
                prediction_state = cognitive_states[local_idx : local_idx + 1]
                pending_predictions[sequence_idx] = self.predict_next_observation(
                    prediction_state, action, error_value
                )
                pending_uncertainties[sequence_idx] = self.transition_uncertainty(prediction_state)
            next_active_state = active_state
            # The model's action embedding supports a single shared action per
            # batch. For the language-training path actions are normally None.
            if any(action is not None for action in actions):
                updated = active_state.clone()
                for local_idx, action in enumerate(actions):
                    updated[local_idx : local_idx + 1] = self.update_state(
                        summaries[local_idx : local_idx + 1],
                        cognitive_states[local_idx : local_idx + 1],
                        action,
                        errors[local_idx : local_idx + 1],
                    )
                next_active_state = updated
            else:
                next_active_state = self.update_state(summaries, cognitive_states, None, errors)
            state = state.clone()
            state[active_indices] = next_active_state

        language = torch.stack(losses).mean() if losses else torch.zeros((), device=state.device, requires_grad=True)
        world = torch.stack(world_losses).mean() if world_losses else torch.zeros((), device=state.device)
        loss = language + float(self.cfg.world_prediction_weight) * world
        return loss, state

    def response_loss(
        self,
        prompt: str,
        response: str,
        *,
        device: torch.device | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Train only on the response bytes while conditioning on the supplied prompt.

        This is used for online dialogue learning so an accepted conversational turn updates
        the continuation behavior without treating the user's prompt itself as a prediction
        target. The representation and all parameters remain fully from-scratch.
        """
        target_ids = self.tokenizer.encode(str(response), bos=False, eos=False)
        if not target_ids:
            return torch.zeros((), device=device or self.token_embedding.weight.device, requires_grad=True)
        target_device = device or self.token_embedding.weight.device
        prompt_ids = self.tokenizer.encode(str(prompt), bos=True, eos=False)
        context = max(3, int(self.cfg.context))
        if len(prompt_ids) + len(target_ids) > context:
            keep_prompt = max(1, context - len(target_ids))
            prompt_ids = prompt_ids[-keep_prompt:]
        if len(prompt_ids) + len(target_ids) > context:
            target_ids = target_ids[-max(1, context - 1):]
            prompt_ids = prompt_ids[-1:]
        full_ids = prompt_ids + target_ids
        if len(full_ids) < 3:
            return torch.zeros((), device=target_device, requires_grad=True)
        tokens = torch.tensor(full_ids, dtype=torch.long, device=target_device).unsqueeze(0)
        logits, _ = self.encode_sequence(tokens, state)
        start = max(0, len(prompt_ids) - 1)
        stop = min(logits.size(1) - 1, len(target_ids) + start)
        if stop <= start:
            return torch.zeros((), device=target_device, requires_grad=True)
        target = tokens[:, 1:][:, start:stop]
        selected = logits[:, start:stop]
        mask = target.ne(PAD) & target.ge(0)
        if not bool(mask.any()):
            return torch.zeros((), device=target_device, requires_grad=True)
        return F.cross_entropy(
            selected.reshape(-1, self.cfg.vocab_size),
            target.reshape(-1),
            ignore_index=PAD,
        ) if bool(mask.all()) else -F.log_softmax(selected, dim=-1).gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)[mask].mean()

    @torch.no_grad()
    def language_bpb_totals(self, tokens: torch.Tensor) -> tuple[float, float]:
        """Return (negative-log2-likelihood, represented UTF-8 bytes).

        Returning totals lets multi-window evaluation aggregate by bytes rather than
        incorrectly giving a short final window the same weight as a full window.
        """
        if tokens.size(1) < 3:
            return 0.0, 0.0
        self.eval()
        logits, _ = self.encode_sequence(tokens[:, :-1])
        target = tokens[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs.gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        numer = torch.zeros((), device=tokens.device, dtype=selected.dtype)
        denom = torch.zeros((), device=tokens.device, dtype=selected.dtype)
        log_two = torch.log(torch.tensor(2.0, device=tokens.device, dtype=selected.dtype))
        for batch_index in range(target.size(0)):
            ids = target[batch_index].detach().tolist()
            lengths = self.tokenizer.token_byte_lengths(ids)
            byte_lengths = torch.tensor(lengths, device=tokens.device, dtype=selected.dtype)
            valid = byte_lengths.gt(0) & target[batch_index].ne(PAD)
            if bool(valid.any()):
                numer = numer + (-selected[batch_index][valid] / log_two).sum()
                denom = denom + byte_lengths[valid].sum()
        return float(numer.detach().cpu()), float(denom.detach().cpu())

    @torch.no_grad()
    def language_bpb(self, tokens: torch.Tensor) -> float:
        """Compute byte-normalized bits per UTF-8 byte for the active tokenizer."""
        numerator, denominator = self.language_bpb_totals(tokens)
        return numerator / denominator if denominator > 0.0 else 8.0

    def pooled_representation(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a trainable normalized representation from Aletheia itself."""
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        _, hidden = self.encode_sequence(tokens, state)
        effective = tokens[:, -hidden.size(1) :]
        mask = effective.ne(PAD).unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1)
        pooled = summed / denom
        return F.normalize(pooled, dim=-1)

    def cross_modal_alignment_loss(
        self,
        text_tokens: torch.Tensor,
        *,
        vision_embeddings: torch.Tensor | None = None,
        audio_embeddings: torch.Tensor | None = None,
        temperature: float = 0.12,
    ) -> torch.Tensor:
        """Align explicitly paired text/vision/audio observations with in-batch negatives.

        The caller supplies the pairing; the model contains no semantic class labels or
        pretrained encoders. Missing modalities are simply skipped.
        """
        if text_tokens.ndim == 1:
            text_tokens = text_tokens.unsqueeze(0)
        text_vec = self.pooled_representation(text_tokens)
        scale = max(1e-3, float(temperature))
        losses = []
        targets = torch.arange(text_vec.size(0), device=text_vec.device)
        text_world = F.normalize(self.text_to_world(text_vec), dim=-1)
        for modality, raw in (("vision", vision_embeddings), ("audio", audio_embeddings)):
            if raw is None:
                continue
            value = raw
            if value.ndim == 1:
                value = value.unsqueeze(0)
            if value.size(0) != text_vec.size(0):
                raise ValueError(f"{modality} batch size must match text batch size")
            projected = F.normalize(self._modality_projector(modality)(value.to(device=text_vec.device, dtype=text_vec.dtype)), dim=-1)
            logits = torch.matmul(text_world, projected.transpose(0, 1)) / scale
            losses.append(0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets)))
        if not losses:
            return torch.zeros((), device=text_vec.device, requires_grad=True)
        return torch.stack(losses).mean()

    def response_alignment_loss(
        self,
        contexts: Sequence[torch.Tensor],
        responses: Sequence[torch.Tensor],
        *,
        temperature: float = 0.12,
    ) -> torch.Tensor:
        """Align observed dialogue contexts with their own responses using in-batch negatives.

        This is a self-supervised structural objective: the only supervision is which
        response actually followed which context in the supplied transcript. No semantic
        labels or pretrained embedding model are used.
        """
        if len(contexts) != len(responses) or len(contexts) < 2:
            return torch.zeros((), device=self.token_embedding.weight.device, requires_grad=True)
        context_vec = torch.cat([self.pooled_representation(item) for item in contexts], dim=0)
        response_vec = torch.cat([self.pooled_representation(item) for item in responses], dim=0)
        scale = max(1e-3, float(temperature))
        logits = torch.matmul(context_vec, response_vec.transpose(0, 1)) / scale
        labels = torch.arange(logits.size(0), device=logits.device)
        forward = F.cross_entropy(logits, labels)
        backward = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (forward + backward)

    @torch.no_grad()
    def text_representation(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a normalized representation from Aletheia's own current parameters."""
        was_training = self.training
        self.eval()
        result = self.pooled_representation(tokens, state)
        if was_training:
            self.train()
        return result

    @torch.no_grad()
    def generate_with_state(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        device: torch.device,
        state: torch.Tensor | None = None,
    ) -> tuple[str, torch.Tensor, dict]:
        seed = self.tokenizer.encode(prompt, bos=True, eos=False)
        x = torch.tensor(seed, dtype=torch.long, device=device).unsqueeze(0)
        running_state = (
            self.initial_state(1, device) if state is None else state.detach().to(device=device)
        )
        was_training = self.training
        self.eval()
        generated_ids: list[int] = []
        surprises: list[float] = []

        for _ in range(max(0, int(max_new_tokens))):
            context = x[:, -self.cfg.context :]
            logits, hidden = self.encode_sequence(context, state=running_state)
            scores = logits[:, -1] / max(float(temperature), 1e-3)
            if top_k > 0:
                values, indices = torch.topk(scores, min(int(top_k), scores.size(-1)))
                filtered = torch.full_like(scores, float("-inf"))
                filtered.scatter_(1, indices, values)
                scores = filtered
            probs = F.softmax(scores, dim=-1)
            if not torch.isfinite(probs).all():
                break
            nxt = torch.multinomial(probs, 1)
            token_id = int(nxt.item())
            generated_ids.append(token_id)
            x = torch.cat([x, nxt], dim=1)

            probability = float(probs[0, token_id].clamp_min(1e-8).item())
            surprise = min(12.0, -math_log2(probability))
            surprises.append(surprise)
            summary = hidden[:, -1]
            running_state = self.update_state(summary, running_state, None, surprise)
            if token_id == EOS:
                break

        if was_training:
            self.train()
        generated = self.tokenizer.decode(generated_ids)
        metrics = {
            "mean_surprise": sum(surprises) / len(surprises) if surprises else 0.0,
            "generated_tokens": len(generated_ids),
        }
        return generated, running_state, metrics

    @torch.no_grad()
    def continuation_bpb(self, prompt: str, continuation: str, device: torch.device) -> float:
        """Score a continuation using the active tokenizer over context-limited windows."""
        prompt_ids = self.tokenizer.encode(prompt, bos=True, eos=False)
        continuation_ids = self.tokenizer.encode(continuation, bos=False, eos=False)
        if not continuation_ids:
            return 8.0
        full_ids = prompt_ids + continuation_ids
        continuation_start = len(prompt_ids)
        context = max(2, int(self.cfg.context))
        segment_size = max(1, min(context - 1, context // 2))
        total_nll = 0.0
        total_count = 0

        for seg_start in range(0, len(continuation_ids), segment_size):
            seg_end = min(len(continuation_ids), seg_start + segment_size)
            target_start = continuation_start + seg_start
            target_end = continuation_start + seg_end
            input_start = max(0, target_end - context)
            window_ids = full_ids[input_start:target_end]
            tokens = torch.tensor(window_ids, dtype=torch.long, device=device).unsqueeze(0)
            if tokens.size(1) < 2:
                continue
            logits, _ = self.encode_sequence(tokens[:, :-1])
            local_start = target_start - input_start - 1
            local_end = target_end - input_start - 1
            target_logits = logits[:, local_start:local_end]
            targets = tokens[:, 1:][:, local_start:local_end]
            if target_logits.size(1) != targets.size(1) or target_logits.size(1) == 0:
                continue
            mask = targets.ne(PAD) & targets.ge(0)
            if not bool(mask.any()):
                continue
            log_probs = F.log_softmax(target_logits, dim=-1)
            selected = log_probs.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            ids = targets[0].detach().tolist()
            byte_lengths = self.tokenizer.token_byte_lengths(ids)
            lengths = torch.tensor(byte_lengths, device=device, dtype=selected.dtype)
            valid = mask[0] & lengths.gt(0)
            if bool(valid.any()):
                total_nll += float((-selected[0][valid]).sum().detach().cpu())
                total_count += int(lengths[valid].sum().item())

        if total_count <= 0:
            return 8.0
        return float(total_nll / total_count / torch.log(torch.tensor(2.0)).item())

    @staticmethod
    def _repeat_trigram_rate(text: str) -> float:
        raw = text.encode("utf-8", errors="replace")
        if len(raw) < 6:
            return 0.0
        grams = [bytes(raw[i:i + 3]) for i in range(len(raw) - 2)]
        return float((len(grams) - len(set(grams))) / max(1, len(grams)))

    @torch.no_grad()
    def generate_candidates(
        self,
        prompt: str,
        *,
        candidate_count: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        device: torch.device,
        state: torch.Tensor | None = None,
    ) -> tuple[str, torch.Tensor, dict]:
        count=max(1,int(candidate_count))
        candidates=[]
        for _ in range(count):
            text,new_state,metrics=self.generate_with_state(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device,
                state=state,
            )
            bpb=self.continuation_bpb(prompt,text,device) if text else 8.0
            repeat=self._repeat_trigram_rate(text)
            # Self-likelihood remains evidence, but repetitive continuations are penalized
            # so reranking is not equivalent to simply selecting the most familiar text.
            score=float(bpb + 1.0 * repeat)
            candidates.append((score,bpb,repeat,text,new_state,metrics))
        candidates.sort(key=lambda item:item[0])
        best=candidates[0]
        all_scores=[float(item[0]) for item in candidates]
        all_bpb=[float(item[1]) for item in candidates]
        all_repeat=[float(item[2]) for item in candidates]
        return best[3],best[4],{
            "candidate_count":count,
            "candidate_scores":all_scores,
            "candidate_bpb":all_bpb,
            "candidate_repeat_trigram_rate":all_repeat,
            "selected_bpb":float(best[1]),
            "selected_repeat_trigram_rate":float(best[2]),
            "candidate_score":float(best[0]),
            "candidate_score_gap":(all_scores[1]-all_scores[0]) if len(all_scores)>1 else 0.0,
            "candidate_bpb_gap":(all_bpb[1]-all_bpb[0]) if len(all_bpb)>1 else 0.0,
            **best[5],
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        device: torch.device,
        state: torch.Tensor | None = None,
    ) -> str:
        generated, _, _ = self.generate_with_state(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
            state=state,
        )
        return generated

    def save(self, path: Path, optimizer: torch.optim.Optimizer) -> None:
        """Atomically save a checkpoint while preserving the previous good copy.

        The temporary file is created beside the destination so os.replace stays on
        one filesystem. A crash during torch.save cannot truncate the active checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        backup = path.with_suffix(path.suffix + ".bak")
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        backup_temp = backup.with_suffix(backup.suffix + ".tmp")
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "config": asdict(self.cfg),
            "action_names": self.action_names,
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        try:
            torch.save(payload, temp_path)
            with temp_path.open("rb+") as handle:
                os.fsync(handle.fileno())
            if path.exists():
                shutil.copy2(path, backup_temp)
                os.replace(backup_temp, backup)
            os.replace(temp_path, path)
        finally:
            for candidate in (temp_path, backup_temp):
                try:
                    candidate.unlink(missing_ok=True)
                except OSError:
                    pass

    @staticmethod
    def _architecture_fields() -> tuple[str, ...]:
        return (
            "dim", "layers", "heads", "ffn", "recurrent_dim",
            "perception_dim", "world_model_hidden", "world_attention_heads",
        )

    @classmethod
    def _checkpoint_architecture(cls, payload: dict) -> dict:
        saved = payload.get("config")
        if not isinstance(saved, dict):
            raise ValueError("checkpoint is missing its model configuration")
        return {field: saved.get(field) for field in cls._architecture_fields()}

    @classmethod
    def _validate_checkpoint_architecture(cls, payload: dict, cfg: ModelConfig) -> None:
        saved = cls._checkpoint_architecture(payload)
        current = {field: getattr(cfg, field) for field in cls._architecture_fields()}
        # FINAL16 checkpoints may not contain the newly introduced cognitive-core
        # architecture fields. Missing fields are therefore treated as an upgrade path:
        # the new modules start from random initialization while all compatible weights
        # are retained. An explicitly stored value must still match exactly.
        mismatches = {
            field: (saved.get(field), current.get(field))
            for field in cls._architecture_fields()
            if saved.get(field) is not None and saved.get(field) != current.get(field)
        }
        if mismatches:
            details = ", ".join(f"{key}={old!r}->{new!r}" for key, (old, new) in mismatches.items())
            raise ValueError(
                "checkpoint architecture differs from the active model; refusing unsafe tensor migration "
                f"({details})"
            )

    @classmethod
    def load(
        cls,
        path: Path,
        device: torch.device,
        cfg: ModelConfig,
        action_names: list[str],
        optimizer: torch.optim.Optimizer,
        tokenizer=None,
    ) -> "AletheiaModel":
        payload = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint is not a valid Aletheia checkpoint dictionary")
        checkpoint_actions = payload.get("action_names")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint is missing its model state_dict")
        cls._validate_checkpoint_architecture(payload, cfg)
        model = cls(cfg, action_names, tokenizer=tokenizer).to(device)
        checkpoint_actions = list(checkpoint_actions or [])
        old_index = {name: idx for idx, name in enumerate(checkpoint_actions)}
        current_state = model.state_dict()
        special_resize = {"token_embedding.weight", "position_embedding.weight", "lm_head.weight"}
        base_state = {}
        for key, value in state_dict.items():
            if key in special_resize or key in {"action_embedding.weight", "action_head.3.weight", "action_head.3.bias"}:
                continue
            target = current_state.get(key)
            if target is not None and tuple(target.shape) == tuple(value.shape):
                base_state[key] = value
        model.load_state_dict(base_state, strict=False)
        with torch.no_grad():
            old_token = state_dict.get("token_embedding.weight")
            if old_token is not None:
                rows = min(old_token.size(0), model.token_embedding.weight.size(0))
                old_token_device = old_token.to(model.token_embedding.weight)
                model.token_embedding.weight[:rows].copy_(old_token_device[:rows])
                # Learned BPE tokens are byte sequences. Initialize each new row from
                # the compatible byte-token rows so tokenizer migration starts from a
                # meaningful representation instead of unrelated random vectors.
                token_bytes = getattr(model.tokenizer, "token_bytes", None)
                if token_bytes is not None and old_token.size(0) >= 256:
                    for token_id in range(max(256, rows), model.token_embedding.weight.size(0)):
                        raw = token_bytes[token_id] if token_id < len(token_bytes) else b""
                        if raw:
                            byte_ids = torch.tensor(list(raw), dtype=torch.long, device=model.token_embedding.weight.device)
                            model.token_embedding.weight[token_id].copy_(old_token_device.index_select(0, byte_ids).mean(dim=0))
            old_pos = state_dict.get("position_embedding.weight")
            if old_pos is not None:
                rows = min(old_pos.size(0), model.position_embedding.weight.size(0))
                model.position_embedding.weight[:rows].copy_(old_pos[:rows].to(model.position_embedding.weight))
            old_action_embedding = state_dict.get("action_embedding.weight")
            old_action_head_w = state_dict.get("action_head.3.weight")
            old_action_head_b = state_dict.get("action_head.3.bias")
            if old_action_embedding is not None and old_action_head_w is not None and old_action_head_b is not None:
                for new_idx, name in enumerate(action_names):
                    old_idx = old_index.get(name)
                    if old_idx is None:
                        continue
                    model.action_embedding.weight[new_idx].copy_(old_action_embedding[old_idx].to(model.action_embedding.weight))
                    model.action_head[3].weight[new_idx].copy_(old_action_head_w[old_idx].to(model.action_head[3].weight))
                    model.action_head[3].bias[new_idx].copy_(old_action_head_b[old_idx].to(model.action_head[3].bias))
        optimizer_state = payload.get("optimizer")
        if isinstance(optimizer_state, dict):
            try:
                optimizer.load_state_dict(optimizer_state)
            except Exception:
                pass
        return model



def math_log2(value: float) -> float:
    """Numerically safe log2 used by generation surprise tracking."""
    import math

    return math.log(max(value, 1e-12), 2.0)
