import torch
import torchvision as tv
import onnx
import onnxruntime as ort
import numpy as np
from torch.utils.data import DataLoader
from onnxruntime import quantization
from typing import Dict, Optional

class CustomCalibrationDataReader(quantization.CalibrationDataReader):
    def __init__(self, torch_dl: DataLoader, input_name: str="input") -> None:
        """
        Constructor to create a CustomCalibrationDataReader for ONNX Quantization

        Parameters
        ----------
        - torch_dl: torch.utils.data.DataLoader
          DataLoader containing the calibration dataset used for static quantization
        - input_name: str, optional
          Name of the input node in the ONNX model (default is "input")
        """
        self.torch_dl = torch_dl        
        self.input_name = input_name
        self.enum_data = iter(self.torch_dl)

    def to_numpy(self, pt_tensor: torch.Tensor) -> np.ndarray:
        """
        Function that converts a PyTorch tensor to a NumPy array

        Parameters
        ----------
        - pt_tensor: torch.Tensor
          The input PyTorch tensor to convert

        Returns
        -------
        - numpy_array: numpy.ndarray
          The converted NumPy array on CPU
        """
        return pt_tensor.detach().cpu.numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Function that returns the next batch of data formatted for the ONNX Runtime calibrator

        Returns
        -------
        - input_feed: dict or None
          A dictionary mapping the input name to the data numpy array, or None if the dataset is exhausted
        """

        batch = next(self.enum_data, None)

        batch = next(self.enum_data, None)
        if batch is not None:

            image_tensor = batch[0]

            return {self.input_name: self.to_numpy(image_tensor)}
        else:
            return None
    
    def rewind(self) -> None:
        """
        Function that resets the data iterator to the beginning of the dataset
        """
        self.enum_data = iter(self.torch_dl)
