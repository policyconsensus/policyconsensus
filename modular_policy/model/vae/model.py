"""
Modified from VQ-BeT https://github.com/jayLEE0301/vq_bet_official
Some code is adapted from Stable Diffusion https://github.com/CompVis/stable-diffusion
"""
import torch.nn
import einops
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer
from reactive_diffusion_policy.model.common.shape_util import get_output_shape
from reactive_diffusion_policy.model.vae.vector_quantize_pytorch.residual_vq import ResidualVQ
from reactive_diffusion_policy.model.vae.distributions import DiagonalGaussianDistribution
from reactive_diffusion_policy.model.vae.utils import *

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=16,
        hidden_dim=128,
        layer_num=1,
        last_activation=None,
    ):
        super(MLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layer_num):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        if last_activation is not None:
            self.last_layer = last_activation
        else:
            self.last_layer = None
        self.apply(weights_init_encoder)

    def forward(self, x):
        h = self.encoder(x)
        state = self.fc(h)
        if self.last_layer:
            state = self.last_layer(state)
        return state

class EncoderCNN(nn.Module):
    def __init__(self,
                    input_dim,
                    output_dim=16,
                    hidden_dim=128,
                    layer_num=1):
        super(EncoderCNN, self).__init__()

        self.action_dim = input_dim

        layers = []
        for i in range(layer_num):
            if i == 0:
                layers.append(nn.Conv1d(input_dim, hidden_dim, kernel_size=5, stride=2, padding=2))
            else:
                layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2))
            layers.append(nn.ReLU())
        layers.append(nn.Conv1d(hidden_dim, output_dim, kernel_size=5, stride=2, padding=2))

        self.encoder = nn.Sequential(*layers)
        self.apply(weights_init_encoder)

    def forward(self, x, flatten=False):
        x = einops.rearrange(x, "N (T A) -> N T A", A=self.action_dim)
        x = einops.rearrange(x, "N T A -> N A T")
        h = self.encoder(x)
        h = einops.rearrange(h, "N C T -> N T C")
        if flatten:
            h = einops.rearrange(h, "N T C -> N (T C)")
        return h

class DecoderRNN(nn.Module):
    def __init__(
        self,
        global_cond_dim,
        temporal_cond_dim,
        output_dim,
        hidden_dim,
        layer_num=1,
    ):
        super(DecoderRNN, self).__init__()
        self.rnn = nn.GRU(global_cond_dim + temporal_cond_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_encoder)

    def forward(self, global_cond, temporal_cond):
        global_cond = global_cond.unsqueeze(1).expand(-1, temporal_cond.size(1), -1)
        x = torch.cat([global_cond, temporal_cond], dim=-1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = einops.rearrange(x, "N T A -> N (T A)")
        return x

class VAE:
    def __init__(
        self,
        horizon=10, # length of action chunk
        shape_meta={},
        n_latent_dims=512,
        mlp_layer_num=1,
        use_conv_encoder=False,
        conv_latent_dims=512,
        conv_layer_num=1,
        use_rnn_decoder=False,
        rnn_latent_dims=512,
        rnn_layer_num=1,
        use_vq=False,
        n_embed=32,
        vqvae_groups=4,
        kl_multiplier=1e-6,
        eval=True,
        device="cuda",
        load_dir=None,
        encoder_loss_multiplier=1.0,
        act_scale=1.0,
    ):
        self.input_dim_h = horizon
        self.input_dim_w = shape_meta['action']['shape'][0]
        self.use_conv_encoder = use_conv_encoder
        self.use_rnn_decoder = use_rnn_decoder
        if self.use_rnn_decoder:
            all_extented_obs_keys = list(shape_meta['extended_obs'].keys())
            self.extented_obs_keys = sorted(all_extented_obs_keys)
            self.rnn_temporal_cond_dim = sum([shape_meta['extended_obs'][extented_obs_key]['shape'][0] for extented_obs_key in self.extented_obs_keys])
        self.use_vq = use_vq
        self.n_embed = n_embed
        self.vqvae_groups = vqvae_groups
        self.kl_multiplier = kl_multiplier
        self.device = device
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale

        self.normalizer = LinearNormalizer()

        if self.use_conv_encoder:
            self.encoder = EncoderCNN(
                input_dim=self.input_dim_w, output_dim=n_latent_dims, hidden_dim=conv_latent_dims, layer_num=conv_layer_num
            ).to(self.device)
        else:
            self.encoder = MLP(
                input_dim=self.input_dim_w * self.input_dim_h, output_dim=n_latent_dims, layer_num=mlp_layer_num
            ).to(self.device)

        output_shape = get_output_shape((self.input_dim_w * self.input_dim_h,), self.encoder)
        if len(output_shape) == 1:
            decoder_n_latent_dims = output_shape[0]
            self.downsampled_input_h = 1
        else:
            decoder_n_latent_dims = np.multiply(*output_shape)
            self.downsampled_input_h = output_shape[0]

        if self.use_rnn_decoder:
            self.decoder = DecoderRNN(global_cond_dim=decoder_n_latent_dims, temporal_cond_dim=self.rnn_temporal_cond_dim,
                                      output_dim=self.input_dim_w, hidden_dim=rnn_latent_dims,
                                      layer_num=rnn_layer_num).to(self.device)
        else:
            self.decoder = MLP(
                input_dim=decoder_n_latent_dims, output_dim=self.input_dim_w * self.input_dim_h, layer_num=mlp_layer_num
            ).to(self.device)
        self.n_latent_dims = n_latent_dims

        if self.use_vq:
            self.vq_layer = ResidualVQ(
                dim=self.n_latent_dims,
                num_quantizers=self.vqvae_groups,
                codebook_size=self.n_embed,
            ).to(self.device)
            self.vq_layer.device = device
        else:
            self.quant = torch.nn.Conv1d(self.n_latent_dims, 2*self.n_embed, 1).to(self.device)
            self.post_quant = torch.nn.Conv1d(self.n_embed, self.n_latent_dims, 1).to(self.device)
        self.embedding_dim = self.n_latent_dims

        self.optim_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )
        if self.use_vq:
            self.optim_params += list(self.vq_layer.parameters())
        else:
            self.optim_params += list(self.quant.parameters())
            self.optim_params += list(self.post_quant.parameters())

        if load_dir is not None:
            try:
                state_dict = torch.load(load_dir)
            except RuntimeError:
                state_dict = torch.load(load_dir, map_location=torch.device("cpu"))
            self.load_state_dict(state_dict)

        if self.use_vq:
            if eval:
                self.vq_layer.eval()
            else:
                self.vq_layer.train()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_action_from_latent(self, latent):
        output = self.decoder(latent) * self.act_scale
        if self.input_dim_h == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

    def get_action_from_latent_with_temporal_cond(self, latent, temporal_cond):
        output = self.decoder(latent, temporal_cond) * self.act_scale
        if self.input_dim_h == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

    def preprocess(self, state):
        if not torch.is_tensor(state):
            state = get_tensor(state, self.device)
        if self.input_dim_h == 1:
            state = state.squeeze(-2)  # state.squeeze(-1)
        else:
            state = einops.rearrange(state, "N T A -> N (T A)")
        return state.to(self.device)

    def quant_state_with_vq(self, state):
        batch_size = state.size(0)
        if len(state.shape) == 2:
            state = einops.rearrange(state, "N (T A) -> N T A", T=self.downsampled_input_h)

        state_vq, vq_code, vq_loss_state = self.vq_layer(state)
        state_vq = state_vq.reshape(batch_size, -1)
        vq_code = vq_code.reshape(batch_size, -1)
        vq_loss_state = torch.sum(vq_loss_state)

        return state_vq, vq_code, vq_loss_state

    def quant_state_without_vq(self, state):
        batch_size = state.size(0)
        if len(state.shape) == 2:
            state = einops.rearrange(state, "N (T A) -> N A T", T=self.downsampled_input_h)
        else:
            state = einops.rearrange(state, "N T A -> N A T")

        moments = self.quant(state)
        posterior = DiagonalGaussianDistribution(moments)
        state_vq = posterior.sample()
        state_vq = einops.rearrange(state_vq, "N A T -> N (T A)")

        return state_vq, posterior

    def postprocess_quant_state_without_vq(self, state_vq):
        state_vq = einops.rearrange(state_vq, "N (T A) -> N A T", T=self.downsampled_input_h)
        state_vq = self.post_quant(state_vq)
        state_vq = einops.rearrange(state_vq, "N A T -> N (T A)")

        return state_vq

    def get_temporal_cond(self, extended_obs_dict, extended_obs_last_step=None, extend_obs_pad_after_n=None):
        temporal_cond = []
        for extented_obs_key in self.extented_obs_keys:
            if extended_obs_last_step is not None:
                extented_obs = extended_obs_dict[extented_obs_key][..., -extended_obs_last_step:, :]
            else:
                extented_obs = extended_obs_dict[extented_obs_key]
            if extend_obs_pad_after_n is not None:
                padding_obs = extended_obs_dict[extented_obs_key][..., -1:, :].repeat(1, extend_obs_pad_after_n, 1)
                extented_obs = torch.cat([extented_obs, padding_obs], dim=-2)
            extented_obs = self.normalizer[extented_obs_key].normalize(extented_obs)
            temporal_cond.append(extented_obs)
        temporal_cond = torch.cat(temporal_cond, dim=-1)
        return temporal_cond

    def compute_loss_and_metric(self, batch):

        state = batch["action"]
        state = self.normalizer['action'].normalize(state)
        state = state / self.act_scale
        state = self.preprocess(state)

        state_rep = self.encoder(state)
        if self.use_vq:
            state_vq, vq_code, vq_loss_state = self.quant_state_with_vq(state_rep)
        else:
            state_vq, posterior = self.quant_state_without_vq(state_rep)
            state_vq = self.postprocess_quant_state_without_vq(state_vq)

        if self.use_rnn_decoder:
            temporal_cond = self.get_temporal_cond(batch["extended_obs"])
            temporal_cond = temporal_cond.to(self.device)
            dec_out = self.decoder(state_vq, temporal_cond)
        else:
            dec_out = self.decoder(state_vq)

        encoder_loss = (state - dec_out).abs().mean()
        vae_recon_loss = torch.nn.MSELoss()(state, dec_out)

        return_dict = {
            "loss": encoder_loss,
            "encoder_loss": encoder_loss.clone().detach().cpu().numpy(),
            "vae_recon_loss": vae_recon_loss.item(),
        }

        if self.use_vq:
            rep_loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)
            return_dict["loss"] = rep_loss
            return_dict.update({
                "vq_code": vq_code,
                "rep_loss": rep_loss.clone().detach().cpu().numpy(),
                "vq_loss_state": vq_loss_state.clone().detach().cpu().numpy(),
            })
        else:
            kl_loss = posterior.kl().mean()
            rep_loss = encoder_loss * self.encoder_loss_multiplier + (kl_loss * self.kl_multiplier)
            return_dict["loss"] = rep_loss
            return_dict.update({
                "kl_loss": kl_loss.clone().detach().cpu().numpy(),
                "rep_loss": rep_loss.clone().detach().cpu().numpy(),
            })

        return return_dict

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        if self.use_vq:
            self.vq_layer.eval()
        else:
            self.quant.eval()
            self.post_quant.eval()

    def train(self):
        self.encoder.train()
        self.decoder.train()
        if self.use_vq:
            self.vq_layer.train()
        else:
            self.quant.train()
            self.post_quant.train()

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)
        if self.use_vq:
            self.vq_layer.to(device)
        else:
            self.quant.to(device)
            self.post_quant.to(device)
        self.device = device

    def state_dict(self):
        state_dict = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
        }
        if self.use_vq:
            state_dict["vq_embedding"] = self.vq_layer.state_dict()
        else:
            state_dict["quant"] = self.quant.state_dict()
            state_dict["post_quant"] = self.post_quant.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        # for compatibility
        if 'state_dicts' in state_dict:
            state_dict = state_dict['state_dicts']['model']
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        if self.use_vq:
            self.vq_layer.load_state_dict(state_dict["vq_embedding"])
            self.vq_layer.eval()
        else:
            self.quant.load_state_dict(state_dict["quant"])
            self.post_quant.load_state_dict(state_dict["post_quant"])
