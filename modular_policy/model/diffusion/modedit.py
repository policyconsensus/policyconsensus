import logging
import math 
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from modular_policy.model.common.positional_embedding import SinusoidalPosEmb
from modular_policy.model.common.module_attr_mixin import ModuleAttrMixin

from typing import Union, Optional, Tuple

logger = logging.getLogger(__name__)


# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(ModuleAttrMixin):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g
    
    
# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(ModuleAttrMixin):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class Attention(ModuleAttrMixin):
    def __init__(
        self, 
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
        block_size: int,
        causal: bool = False,
        bias=False,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
        self.causal = causal
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash and causal:
            print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")
        # Dynamically compute causal mask instead of using a fixed bias buffer
        self.block_size = block_size
        self.qk_norm = qk_norm
        # init qk norm if enabled
        if self.qk_norm:
            self.q_norm = RMSNorm(n_embd//self.n_head, eps=1e-6)
            self.k_norm = RMSNorm(n_embd//self.n_head, eps=1e-6)
        else: 
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, x, context=None, custom_attn_mask=None):
        B, T, C = x.size()

        if context is not None:
            k = self.key(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(context).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        else:
            k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=custom_attn_mask, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=self.causal)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Optimize custom attention masking
            if custom_attn_mask is not None:
                att = att.masked_fill(custom_attn_mask == 0, float('-inf'))
            elif self.causal:
                # Dynamically compute causal mask based on current sequence length T
                causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
                att = att.masked_fill(causal_mask == 0, float('-inf'))

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
        logger.debug(f"Attention weights (after softmax): {att}")
        logger.debug(f"Query shape: {q.shape}, Key shape: {k.shape}, Value shape: {v.shape}")
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class CondRouterMLP(ModuleAttrMixin):

    def __init__(
            self, 
            n_embd: int,
            num_experts: int,
            use_swish: bool = True,
            use_relus: bool = False,
            dropout: float = 0,
            make_it_big: bool = False
        ):
        super().__init__()
        layers = []
        factor = 2 if make_it_big else 1  # Factor to double the hidden dimensions if make_it_big is True
        repeat = 2 if make_it_big else 1  # Repeat layers if make_it_big is True

        for i in range(repeat):  # Repeat constructing layers to double the number of layers if make_it_big
            if i == 0:
                curr_embed = n_embd
            else:
                curr_embed = factor * 2 * n_embd
            if use_swish:
                layers.append(SwishGLU(curr_embed, factor * 2 * n_embd))
            else:
                layers.append(nn.Linear(curr_embed, factor * 2 * n_embd))
                if use_relus:
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        # Ensure that the last layer always maps to the number of experts
        layers.append(nn.Linear(factor * 2 * n_embd, num_experts))
        
        self.mlp = nn.Sequential(*layers)

        # Initialize weights with zeros 
        self._init_weights()

    def forward(self, x):
        return self.mlp(x)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class Mlp(ModuleAttrMixin):
    def __init__(
            self, 
            n_embd: int,
            bias: bool,
            use_swish: bool = True,
            use_relus: bool = False,
            dropout: float = 0,
            identity_only: bool = False,
            output_dim: Optional[int] = None
        ):
        super().__init__()
        self.identity_only = identity_only
        layers = []

        if output_dim is not None:
            n_embed_final = output_dim
        else:
            n_embed_final = n_embd
        
        if identity_only:
            # Initialize as identity layers
            identity_layer = nn.Linear(n_embd, n_embd, bias=False)
            nn.init.eye_(identity_layer.weight)  # Set weights to identity matrix
            layers.append(identity_layer)
        else:
            if use_swish:
                layers.append(SwishGLU(n_embd, 4 * n_embd))
            else:
                layers.append(nn.Linear(n_embd, 4 * n_embd, bias=bias))
                if use_relus:
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(4 * n_embd, n_embed_final, bias=bias))
        
        self.mlp = nn.Sequential(*layers)
        
        if identity_only:
            # Freeze the parameters so they are not updated during training
            for param in self.mlp.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.mlp(x)


class RouterCond(ModuleAttrMixin):
    def __init__(
        self,
        hidden_states: int,
        cond_dim: int,
        num_experts: int,
        top_k: int,
        use_argmax: bool = False,
        normalize: bool = True,
        cond_router: bool = True,
        router_context_cond_only: bool = False,
        temperature: float = 1.0,
        use_shared_expert: bool = False,
    ):
        """Initialize the RouterCond module."""
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize = normalize
        self.temperature = temperature
        self.use_argmax = use_argmax
        self.use_shared_expert = use_shared_expert
        self.cond_router = cond_router
        self.router_context_cond_only = router_context_cond_only

        self.router = self._create_router(hidden_states, cond_dim)
        self.logits = None

    def _create_router(self, hidden_states: int, cond_dim: int) -> nn.Module:
        """Create the router MLP based on the configuration."""
        if self.cond_router:
            input_dim = cond_dim if self.router_context_cond_only else hidden_states + cond_dim
        else:
            input_dim = hidden_states

        return CondRouterMLP(
            input_dim,
            self.num_experts,
            use_swish=False,
            dropout=0,
            make_it_big=False
        )

    def forward(self, inputs: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """Forward pass of the router."""
        input_shape = inputs.size()
        logits = self._compute_logits(inputs, cond)
        probs = self._compute_probabilities(logits)
        router_mask, top_k_indices, router_probs = self._select_experts(probs, input_shape)
        return router_mask, top_k_indices, router_probs, probs.view(*input_shape[:-1], -1)

    def _compute_logits(self, inputs: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute logits based on inputs and conditional information."""
        if self.cond_router:
            return self._compute_cond_logits(inputs, cond)
        return self._compute_uncond_logits(inputs)

    def _compute_cond_logits(self, inputs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Compute logits for conditional routing."""
        if cond.shape[-2] != inputs.shape[-2]:
            cond = einops.repeat(cond, 'b t d -> b (t n) d', n=int(inputs.shape[-2] / cond.shape[-2]))

        if self.router_context_cond_only:
            router_inputs = cond.reshape(-1, cond.size(-1))
        else:
            router_inputs = torch.cat([inputs, cond], dim=-1).reshape(-1, inputs.size(-1) + cond.size(-1))

        logits = self.router(router_inputs)
        return logits

    def _compute_uncond_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute logits for unconditional routing."""
        return self.router(inputs.reshape(-1, inputs.size(-1)))

    def _compute_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute probabilities from logits."""
        logits = (logits - logits.max(dim=-1, keepdim=True).values) / self.temperature
        self.logits = logits

        probs = torch.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-9, max=1-1e-9)

        self._validate_probabilities(probs)
        return probs

    def _validate_probabilities(self, probs: torch.Tensor):
        """Validate the computed probabilities."""
        if not torch.isfinite(probs).all():
            logging.warning("Probabilities contain inf or NaN values")
        if not torch.allclose(probs.sum(dim=-1), torch.tensor(1.0, dtype=probs.dtype), atol=1e-5):
            logging.warning("Probabilities do not sum up to 1")

    def _select_experts(self, probs: torch.Tensor, input_shape: torch.Size):
        """Select experts based on computed probabilities."""
        if self.use_shared_expert and self.top_k == 2:
            return self._select_experts_with_shared(probs, input_shape)
        return self._select_experts_without_shared(probs, input_shape)

    def _select_experts_with_shared(self, probs: torch.Tensor, input_shape: torch.Size):
        """Select experts when using a shared expert."""
        shared_expert_index = self.num_experts - 1
        other_probs = probs[:, :3]
        other_expert_index = torch.multinomial(other_probs, 1) if self.training and not self.use_argmax else other_probs.topk(1, dim=-1).indices
        
        top_k_indices = torch.cat([other_expert_index, torch.full_like(other_expert_index, shared_expert_index)], dim=-1)
        router_mask = torch.zeros_like(probs).scatter_(1, top_k_indices, 1)
        
        router_probs = probs.clone()
        router_probs[:, 3:shared_expert_index] = 0
        router_probs = router_probs * router_mask

        return self._format_output(router_mask, top_k_indices, router_probs, input_shape)

    def _select_experts_without_shared(self, probs: torch.Tensor, input_shape: torch.Size):
        """Select experts when not using a shared expert."""
        
        # Flatten batch dimensions
        flat_probs = probs.view(-1, probs.size(-1))
        
        if self.training and not self.use_argmax:
            top_k_indices = torch.multinomial(flat_probs, self.top_k, replacement=False)
        else:
            top_k_indices = flat_probs.topk(self.top_k, dim=-1).indices
        
        try:
            router_mask = torch.zeros_like(flat_probs).scatter_(1, top_k_indices, 1)
            router_probs = torch.zeros_like(flat_probs).scatter_(1, top_k_indices, flat_probs.gather(1, top_k_indices))
            
            # Reshape back to original dimensions
            router_mask = router_mask.view(probs.shape)
            router_probs = router_probs.view(probs.shape)
            top_k_indices = top_k_indices.view(probs.shape[:-1] + (self.top_k,))
        except RuntimeError as e:
            print(f"Error in scatter_ operation: {e}")
            print(f"Debug - flat_probs shape: {flat_probs.shape}, top_k_indices shape: {top_k_indices.shape}")
            print(f"Debug - flat_probs device: {flat_probs.device}, top_k_indices device: {top_k_indices.device}")
            raise
        return self._format_output(router_mask, top_k_indices, router_probs, input_shape)

    def _format_output(self, router_mask: torch.Tensor, top_k_indices: torch.Tensor, router_probs: torch.Tensor, input_shape: torch.Size):
        """Format the output of the expert selection process."""
        router_mask = router_mask.view(*input_shape[:-1], -1)
        top_k_indices = top_k_indices.view(*input_shape[:-1], -1)
        router_probs = router_probs.view(*input_shape[:-1], -1)

        if self.normalize:
            router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True)

        return router_mask, top_k_indices, router_probs
    

class NoiseBlockMoE(ModuleAttrMixin):
    """
    Block with AdaLN-Zero conditioning and efficient expert caching.
    """
    def __init__(
            self, 
            n_embd: int, 
            n_heads: int, 
            attn_pdrop: float, 
            mlp_pdrop: float, 
            noise_in_cross_attention: bool = False,
            cond_router: bool = False,
            use_cross_attention: bool = False, 
            num_experts: int = 4,
            top_k: int = 2,
            router_normalize: bool = True,
            router_context_cond_only: bool = True,
            use_argmax: bool = False,
            use_shared_expert: bool = False,
            identity_expert: bool = False,
            attn_arg: str = 'causal',
        ):
        assert not identity_expert, "deprecated identity_expert"
        super().__init__()
        self.ln_1 = RMSNorm(n_embd, eps=1e-6)
        self.n_embd = n_embd
        self.attn = Attention(
            n_embd, 
            n_heads, 
            qk_norm=True,
            attn_pdrop=attn_pdrop,
            resid_pdrop=0,
            block_size=100,
            causal=True,
        )
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_att = Attention(
                n_embd, 
                n_heads, 
                qk_norm=True,
                attn_pdrop=attn_pdrop,
                resid_pdrop=0,
                causal=True,
            )
            self.ln_3 = RMSNorm(n_embd, eps=1e-6) 

        self.ln_2 = RMSNorm(n_embd, eps=1e-6) 
        self.logits = None
        
        self.cond_router = cond_router
        self.num_experts = num_experts
        self.use_shared_expert = use_shared_expert
        self.use_argmax = use_argmax
        self.router_normalize = router_normalize
        self.router_context_cond_only = router_context_cond_only
        self.mlp_pdrop = mlp_pdrop
        self.top_k = top_k

        if self.use_shared_expert:
            top_k_router = top_k - 1
            num_experts_router = num_experts - 1
        else:
            num_experts_router = num_experts
            top_k_router = top_k

        self.router = RouterCond(
            n_embd, 
            n_embd,
            num_experts_router, 
            top_k_router, 
            use_argmax=use_argmax,
            normalize=router_normalize,
            cond_router=cond_router,
            router_context_cond_only=router_context_cond_only,
        )

        self.experts = nn.ModuleDict(
            {
                f"expert_{i}": Mlp(
                    n_embd,  # in_features
                    bias=False,
                    dropout=mlp_pdrop
                )
                for i in range(num_experts_router - int(identity_expert))
            }
        )
        if self.use_shared_expert:
            self.shared_mlp = Mlp(n_embd, bias=False, dropout=mlp_pdrop)

        if identity_expert:
            self.experts[f"expert_{num_experts_router}"] = nn.Identity()

        self.noise_in_cross_attention = noise_in_cross_attention
        self.probs = None
        
        # To track the usage of each expert
        self.expert_usage = torch.zeros(num_experts_router)
        self.train_expert_usage = torch.zeros(num_experts_router)
        self.inference_expert_usage = torch.zeros(num_experts_router)
        self.total_tokens_processed = 0

    def forward(self, x, c, context=None, custom_attn_mask=None):
        # First apply attention
        x = x + self.attn(self.ln_1(x) + c, custom_attn_mask=custom_attn_mask)
        
        if self.use_cross_attention and context is not None:
            if self.noise_in_cross_attention:
                x = x + self.cross_att(self.ln_3(x) + c, context, custom_attn_mask=custom_attn_mask)
            else:
                x = x + self.cross_att(self.ln_3(x), context, custom_attn_mask=custom_attn_mask)
        x = self.ln_2(x)
            
        batch_tokens = x.size(0) * x.size(1)

        if self.cond_router:
            router_mask, top_k_indices, router_probs, true_probs = self.router(x, c)
        else:
            router_mask, top_k_indices, router_probs, true_probs = self.router(x, None)
        
        next_states = torch.zeros_like(x)

        num_balanced_experts = len(self.experts)
        
        # Process inputs through selected experts
        for idx in range(num_balanced_experts):
            token_indices = router_mask[:, :, idx].bool()
            if token_indices.any():
                expert = self.experts[f"expert_{idx}"]
                probs = router_probs[:, :, idx][token_indices].unsqueeze(-1)
                next_states[token_indices] += probs * expert(x[token_indices]).to(next_states.dtype)

                # Track expert usage statistics
                if self.experts[f"expert_{idx}"].training:
                    self.expert_usage[idx] += token_indices.sum().item()
                else:
                    self.inference_expert_usage[idx] += token_indices.sum().item()

        # Apply shared expert if enabled
        if self.use_shared_expert:
            shared_output = self.shared_mlp(x)
            if router_mask.size(-1) > num_balanced_experts:
                shared_prob = router_probs[:, :, -1].unsqueeze(-1)
                next_states += shared_prob * shared_output
            else:
                next_states = next_states + shared_output

        # Store routing information during training
        if (
            self.training or 
            any(
                self.experts[f"expert_{idx}"].training
                for idx in range(num_balanced_experts)
            )
        ):
            self.logits = self.router.logits
            self.probs = {
                "probs": true_probs,
                "top_k_hot": router_mask,
                "load_balancing_term": num_balanced_experts * (
                    router_probs[:, :, :num_balanced_experts].mean(dim=(0, 1)) * 
                    (router_mask[:, :, :num_balanced_experts].sum(dim=(0, 1)) / batch_tokens)
                ).sum()
            }
        self.total_tokens_processed += batch_tokens
        return x + next_states

    def get_expert_usage(self):
        """Get the combined expert usage statistics"""
        return self.inference_expert_usage

    def reset_expert_usage(self):
        """Reset all expert usage statistics"""
        self.expert_usage.zero_()
        self.inference_expert_usage.zero_()
        self.total_tokens_processed = 0


class MoDEDiT(ModuleAttrMixin):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        cond_router: bool = True,
        num_experts: int = 4,
        top_k: int = 2,
        router_normalize: bool = True,
        use_shared_expert: bool = False,
        use_argmax: bool = False,
    ):
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = n_obs_steps + horizon - 1

        # embedding stems
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.sigma_emb = nn.Sequential( # TODO: this is weird, but MoDE does this
            nn.Linear(1, n_emb),        #       we may make it 1-layer only
            nn.Linear(n_emb, n_emb, bias=False),
        )
        self.obs_emb = nn.Linear(cond_dim, n_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_processor = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.Mish(),
            nn.Linear(4 * n_emb, n_emb)
        )

        # MoE blocks
        self.moe_layers = nn.ModuleList([
            NoiseBlockMoE(
                n_embd=n_emb,
                n_heads=n_head,
                attn_pdrop=p_drop_attn,
                mlp_pdrop=p_drop_attn,
                cond_router=cond_router,
                num_experts=num_experts,
                top_k=top_k,
                router_normalize=router_normalize,
                use_shared_expert=use_shared_expert,
                use_argmax=use_argmax,
                attn_arg='causal',
            ) for _ in range(n_layer)
        ])

        # decoder head
        self.ln_f = RMSNorm(n_emb, eps=1e-6)
        self.head = nn.Linear(n_emb, output_dim)
            
        # constants
        self.T = T
        self.horizon = horizon
        self.num_experts = num_experts
        self.logits_per_layer = None
        self.probs_per_layer = None

        # init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, 'out_features') and module.out_features == self.num_experts:
                torch.nn.init.zeros_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
        elif isinstance(module, MoDEDiT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        def use_weight_decay(module_name):
            return all(x not in module_name for x in [
                'bias', 'LayerNorm', 'embedding', 
                'pos_emb', 'cond_pos_emb', '_dummy_variable'
            ])
        for pn, p in self.named_parameters():
            if use_weight_decay(pn):
                decay.add(pn)
            else:
                no_decay.add(pn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        sigmas: torch.Tensor,
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None, **kwargs):
        """
        x: (B,T,input_dim)
        sigmas: (B,)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)
        sigma_emb = self.sigma_emb(sigmas.unsqueeze(-1)).unsqueeze(1)
        obs_emb = self.obs_emb(cond)
        cond_emb = torch.cat([time_emb, obs_emb], dim=1)
        
        tc = cond_emb.shape[1]
        cond_emb = self.drop(cond_emb + self.pos_emb[:, :tc, :])
        cond_emb = self.cond_processor(cond_emb)
        
        t = input_emb.shape[1]
        input_emb = self.drop(input_emb + self.pos_emb[:, :t, :])

        x = torch.cat([cond_emb, input_emb], dim=1)
        x = self.forward_modedit(x, sigma_emb)
        x = x[:, tc:, :]  # remove the conditioning part

        # prediction head
        x = self.ln_f(x)
        x = self.head(x)
        return x
    
    def forward_modedit(self, x, sigma_emb):
        logits_per_layer = []
        probs_per_layer = []
        for layer in self.moe_layers:
            x = layer(x, sigma_emb, sigma_emb)
            logits_per_layer.append(layer.logits)
            probs_per_layer.append(layer.probs)
        self.logits_per_layer = logits_per_layer
        self.probs_per_layer = probs_per_layer
        return x
    
    def load_balancing_loss(self):
        """ 
        Compute the load balancing loss for MoE with separate control for entropy and KL divergence.
        
        Args:
            probs: List of dictionaries, each containing "probs" and "top_k_hot" tensors
            use_entropy: Boolean to include entropy term in the loss
            use_kl: Boolean to include KL divergence term in the loss
            entropy_weight: Weight for the entropy term
            kl_weight: Weight for the KL divergence term
            balance_weight: Weight for the original balance term
        Returns:
            Scalar loss value
        """
        total_loss = 0.0

        if self.probs_per_layer[0] is None:
            return total_loss       # if MoE is not used, return 0

        if 'load_balancing_loss' not in self.probs_per_layer[0]:
            self.probs_per_layer[0]['load_balancing_loss'] = []
            for layer in self.moe_layers:
                if hasattr(layer, 'probs') and 'load_balancing_term' in layer.probs:
                    self.probs_per_layer[0]['load_balancing_loss'].append(layer.probs['load_balancing_term'])

        list_of_losses = self.probs_per_layer[0]['load_balancing_loss']

        for block_loss in list_of_losses:
            total_loss += block_loss
        
        if len(list_of_losses) > 0:
            total_loss = total_loss / len(list_of_losses)

        return total_loss


# def test():

#     # GPT with time embedding and obs cond
#     transformer = MoDeDiT(
#         input_dim=16,
#         output_dim=16,
#         horizon=8,
#         n_obs_steps=4,
#         cond_dim=10,
#     )
#     opt = transformer.configure_optimizers()
    
#     timestep = torch.tensor(0)
#     sample = torch.zeros((4,8,16))
#     sigmas = torch.zeros((4,))
#     cond = torch.zeros((4,4,10))
#     out = transformer(sample, sigmas, timestep, cond)
#     print(f"{out.shape=}")
    
#     print(f"{sigmas.shape=}")
#     transformer.eval()
#     transformer.freeze_router()
#     transformer.unfreeze_router()
#     transformer.load_balancing_loss()
