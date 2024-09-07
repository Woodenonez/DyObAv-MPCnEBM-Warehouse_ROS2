import os
import pathlib
from typing import Optional

import numpy as np
from PIL import Image # type: ignore
import torch # type: ignore

from .network_loader import NetworkLoader


PathNode = tuple[float, float]

root_dir = pathlib.Path(__file__).resolve().parents[1]


class MotionPredictor:
    def __init__(self, config_file_path: str, model_suffix: str, ref_image_path:Optional[str]=None) -> None:
        self.network_loader = NetworkLoader.from_config(config_file_path=config_file_path, project_dir=str(root_dir), verbose=True)
        self.network_loader.quick_setup_inference(model_suffix=model_suffix)
        if ref_image_path is not None:
            self.load_ref_image(ref_img_path=ref_image_path)

    def load_ref_image(self, ref_img_path: str) -> None:
        self.ref_image = torch.tensor(np.array(Image.open(ref_img_path).convert('L')))

    def get_motion_prediction(self, input_traj: list[PathNode], rescale:Optional[float]=1.0, debug:bool=False):
        """Get motion prediction

        Args:
            input_traj: Input trajectory.
            rescale: Scale from real world to image world. Defaults to 1.0.

        Returns:
            clusters_list: A list of clusters, each cluster is a list of points.
            mu_list_list: A list of means of the clusters.
            std_list_list: A list of standard deviations of the clusters.
            conf_list_list: A list of confidence of the clusters.
            pred (if debug): The logits prediction, [CxHxW].
            prob_map (if debug): The probability map, [CxHxW]
        """
        if rescale is not None:
            input_traj = [(x[0]*rescale, x[1]*rescale) for x in input_traj]
        pred = self.network_loader.inference(input_traj=input_traj, ref_image=self.ref_image)
        prob_map = self.network_loader.net_manager.to_prob_map(pred.unsqueeze(0))[0, :]
        clusters_list, mu_list_list, std_list_list, conf_list_list = self.network_loader.clustering_and_fitting(prob_map)
        if debug:
            return clusters_list, mu_list_list, std_list_list, conf_list_list, pred, prob_map
        return clusters_list, mu_list_list, std_list_list, conf_list_list
    
    def get_motion_prediction_samples(self, input_traj: list[PathNode], rescale:Optional[float]=1.0, num_samples:int=100, replacement=True):
        """Get motion prediction

        Args:
            input_traj: Input trajectory.
            rescale: Scale from real world to image world. Defaults to 1.0.
            num_samples: The number of samples to generate on each probability map. Defaults to 500.
            replacement: Whether to sample with replacement (if False, each sample index appears only once). Defaults to True.

        Returns:
            prediction_samples: numpy array [T*x*y], meaning (x, y) at time T
        """
        if rescale is not None:
            input_traj = [(x[0]*rescale, x[1]*rescale) for x in input_traj]
        pred = self.network_loader.inference(input_traj=input_traj, ref_image=self.ref_image)
        prob_map = self.network_loader.net_manager.to_prob_map(pred.unsqueeze(0))
        prediction_samples = self.network_loader.net_manager.gen_samples(prob_map, num_samples=num_samples, replacement=replacement)[0, :].numpy()
        return prediction_samples
    
    def clustering_and_fitting_from_samples(self, traj_samples: np.ndarray, eps=10, min_sample=5, enlarge=1.0, extra_margin=0.0):
        """Inference the network and then do clustering.

        Args:
            traj_samples: numpy array [T*x*y], meaning (x, y) at time T
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 10.
            min_sample: The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 5.

        Raises:
            ValueError: If the input probability maps are not [CxHxW].

        Returns:
            clusters_list: A list of clusters, each cluster is a list of points.
            mu_list_list: A list of means of the clusters.
            std_list_list: A list of standard deviations of the clusters.
            conf_list_list: A list of confidence of the clusters.
        """
        clusters_list, mu_list_list, std_list_list, conf_list_list = self.network_loader.clustering_and_fitting_from_samples(traj_samples, eps=eps, min_sample=min_sample, enlarge=enlarge, extra_margin=extra_margin)
        return clusters_list, mu_list_list, std_list_list, conf_list_list

