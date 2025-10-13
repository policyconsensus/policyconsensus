import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from modular_policy.policy.base_policy import BasePolicy
from typing import List


class DiffusionUnetsBase(BasePolicy):

    """
    Adaptation related methods
    """
    def adapt(self, method):
        # child class should implement this method
        # when called, policy.adapt(), should act as policy.train()
        # but with only a few portion of the parameters being updated
        raise NotImplementedError
    

    def augment_unets(self, num):
        # child class should implement this method
        # when called, policy.augment_unets(), should add unets to the policy
        # and update instance variable `self.num_new_unets` accordingly
        raise NotImplementedError
    

    """
    Inference-time flow control related methods
    """
    inference_time_mask: List[bool] = None
    def set_inference_time_mask(self, mask: List[bool]):
        # during inference, mask out a set of unets' score during sampling
        # mask: a list of bool, True means the unet is used, False means the unet is masked out
        # mask should be the same length as the number of unets
        self.inference_time_mask = mask
    

    """
    Score similarity logging and visualization related methods
    """
    num_cosine_similarity_log: int = 0
    cosine_similarity_log: np.ndarray = None
    @torch.inference_mode()
    def log_score_cosine_similarity(self, scores: List[torch.Tensor]):
        # scores: a list of tensors, each tensor is the score of one unet
        # log the cosine similarity between the scores of different unets
        # each score has shape (batch_size, horizon, action_dim)
        _, _, action_dim = scores[0].shape
    
        flats = torch.stack([   # (n_models, batch_size * horizon, action_dim)
            s.reshape(-1, action_dim) 
            for s in scores], 
            dim=0)
        norm = F.normalize(flats, p=2, dim=-1)
        sim = torch.einsum('mid,nid->mn', norm, norm) / flats.size(1)   # (n_models, n_models)
        sim.fill_diagonal_(1.0)

        if len(self.inference_time_mask) > len(scores):
            # some unets are masked out during inference
            original_sim = torch.zeros(
                (len(self.inference_time_mask), len(self.inference_time_mask)), 
                device=sim.device)
            original_sim.fill_diagonal_(1.0)
            mask = torch.tensor(self.inference_time_mask, device=sim.device).bool()
            idx = mask.nonzero(as_tuple=True)[0]
            original_sim[idx[:, None], idx] = sim
            sim = original_sim
        
        sim = sim.cpu().numpy()

        # running average
        if self.cosine_similarity_log is None:
            self.cosine_similarity_log = sim
        else:
            self.cosine_similarity_log = (
                self.cosine_similarity_log * self.num_cosine_similarity_log + sim
            ) / (self.num_cosine_similarity_log + 1)
        self.num_cosine_similarity_log += 1


    def plot_cosine_similarity_log(self, save_path: str):
        assert self.cosine_similarity_log is not None, \
            "cosine_similarity_log is not initialized"
        text_size = 40
        sim = self.cosine_similarity_log

        # make it 5x5
        n = sim.shape[0]
        new_sim = np.zeros((5, 5))
        new_sim[:n, :n] = sim
        sim = new_sim

        # mask out the upper triangle (i < j)
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        mask[n:] = True
        masked_sim = np.ma.array(sim, mask=mask)

        # create a copy of Viridis where masked values are white
        cmap = plt.get_cmap('GnBu').copy()
        cmap.set_bad(color='white')

        fig, ax = plt.subplots(figsize=(12, 12))
        cax = ax.matshow(masked_sim, cmap=cmap, vmin=sim.min(), vmax=sim.max())

        # only bottom & left ticks
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(top=False, right=False, labelsize=text_size)
        # hide the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel("Component index", fontsize=text_size)
        ax.set_ylabel("Component index", fontsize=text_size)

        # annotate only the visible cells (i >= j)
        for i in range(n):
            for j in range(n):
                if i >= j:
                    val = sim[i, j]
                    bg = cax.cmap(cax.norm(val))[:3]
                    brightness = np.dot(bg, [0.299, 0.587, 0.114])
                    text_color = 'white' if brightness < 0.5 else 'black'
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha='center', va='center',
                        color=text_color, fontsize=text_size,
                    )

        cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=text_size)

        plt.tight_layout()
        plt.savefig(save_path, 
            bbox_inches='tight',
            pad_inches=0.1,
            format=save_path.split('.')[-1])
        plt.close(fig)

    
    """
    Weight logging and visualization related methods
    """
    # num_weights_log: int = 0
    # weights_log: np.ndarray = None
    # @torch.inference_mode()
    # def log_weights(self, weights: torch.Tensor):
    #     # weights: (num_unets, batch_size)
    #     assert len(weights) == len(self.inference_time_mask), \
    #         "weights should have the same length as the number of unets"
        
    #     this_weights = weights.cpu().numpy().mean(axis=-1)

    #     # running average
    #     if self.weights_log is None:
    #         self.weights_log = this_weights
    #     else:
    #         self.weights_log = (
    #             self.weights_log * self.num_weights_log + this_weights
    #         ) / (self.num_weights_log + 1)
    #     self.num_weights_log += 1
    max_log_size: int = 100_000
    num_weights_log: int = 0
    weights_log: List[List[float]] = None               # TODO: this can be problematic in size, but
    def log_weights(self, weights: torch.Tensor):       # we need to log everything atm
        # weights: (num_unets, batch_size)
        if self.weights_log is None:
            self.weights_log = [[] for _ in range(len(weights))]
        for i in range(len(self.weights_log)):
            self.weights_log[i].extend(weights[i].detach().cpu().numpy().tolist())
            self.weights_log[i] = self.weights_log[i][-self.max_log_size:]
        self.num_weights_log += 1
        

    def plot_weights_log(self, save_path: str, method: str = "hist"):
        assert self.weights_log is not None, "weights_log is not initialized"
        n_unets = len(self.weights_log)

        if method == 'hist':
            # histogram, x-axis is the weight value in each unet, range [-1, 1]
            # and y-axis is the density of the weight value there will be 
            # `n_unets` subplots vertically
            fig, axs = plt.subplots(n_unets, 1, figsize=(10, 5 * n_unets))
            for i in range(n_unets):
                if len(self.weights_log[i]) == 0:
                    continue
                axs[i].hist(self.weights_log[i], bins=50, density=True)
                axs[i].set_title(f"Component {i} weights")
                axs[i].set_xlabel("Weight value")
                axs[i].set_ylabel("Density")
                axs[i].set_xlim(-1, 1)
        
        elif method == 'median_iqr':
            # in one plot, x-axis is the unet index, y-axis is the median 
            # of the weights of each unet, and the error bar is the IQR of the weights
            # there will be `n_unets` points in the plot
            medians = []
            iqr = []
            for i in range(n_unets):
                if len(self.weights_log[i]) == 0:
                    continue
                medians.append(np.median(self.weights_log[i]))
                q1 = np.quantile(self.weights_log[i], 0.25)
                q3 = np.quantile(self.weights_log[i], 0.75)
                iqr.append(q3 - q1)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.errorbar(range(n_unets), medians, yerr=iqr, fmt='o', capsize=5)
            ax.set_title("Component weights median and IQR")
            ax.set_xlabel("Component index")
            ax.set_ylabel("Weight value")
            ax.set_xlim(-1, n_unets)
            ax.set_ylim(-1, 1)

        elif method == 'lineplot':
            # in one plot, x-axis is the timestep, y-axis is the weight value
            # and there will be `n_unets` lines in the plot
            fig, ax = plt.subplots(figsize=(10, 5))
            for i in range(n_unets):
                if len(self.weights_log[i]) == 0:
                    continue
                ax.plot(self.weights_log[i], label=f"Component {i}")
            ax.set_title("Component weights over time")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Weight value")
            ax.set_xlim(-1, len(self.weights_log[0]))
            ax.set_ylim(-1, 1)
            ax.legend()
        
        else:
            raise ValueError(f"Unknown method: {method}")

        plt.tight_layout()
        plt.savefig(save_path, 
            bbox_inches='tight',
            pad_inches=0.1,
            format=save_path.split('.')[-1])
        plt.close(fig)
