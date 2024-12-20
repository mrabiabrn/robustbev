"""
    Adapted from https://github.com/Daniel-xsy/RoboBEV
"""
from copy import deepcopy
import functools
import PIL
from PIL import Image

import torch
import numpy as np

from imagecorruptions import corrupt


class Clean:
    def __init__(self, severity, norm_config):
        """
        No corruption
        """
        self.severity = severity
        assert severity >= 1 and severity <= 5, f"Corruption Severity should between (1, 5), now {severity}"
        self.norm_config = norm_config
        self.corruption = 'clean'
    def __call__(self, x):
        return x


class BaseCorruption:
    def __init__(self, severity, norm_config, corruption):
        """
        Base Corruption Class
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        self.severity = severity
        assert severity >= 1 and severity <= 5, f"Corruption Severity should between (1, 5), now {severity}"
        self.norm_config = norm_config
        self.corruption = corruption
        print(f"Corruption: {corruption} with severity {severity}")
        try:
            self.corrupt_func = self._get_corrupt_func()
        except:
            self.corrupt_func = None

        self.pass_through = False

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): [B, M, C, H, W]
        """

        if self.corruption in ["CameraCrash", "FrameLost", "LowLight", "ColorQuant"]:
            return self.corrupt_func(img)
        
        mean = self.norm_config['mean']
        std = self.norm_config['std']
        
        img = deepcopy(img)
        B, M, C, H, W = img.size()
        img = img.permute(0, 1, 3, 4, 2) # [B, M, C, H, W] => [B, M, H, W, C]

        # pixel value [0, 255]
        assert img.min() >= 0 and img.max() <= 255, "Image pixel out of range"
        new_img = np.zeros_like(img)
        for b in range(B):
            for m in range(M):
                new_img[b, m] = self.corrupt_func(np.uint8(img[b, m].numpy()))

        return new_img

    def _get_corrupt_func(self):
        if self.corruption == 'CameraCrash': 
            return CameraCrash(self.severity, self.norm_config)
        elif self.corruption == 'FrameLost':
            return FrameLost(self.severity, self.norm_config)
        elif self.corruption == 'LowLight':
            return LowLight(self.severity, self.norm_config)
        elif self.corruption == 'ColorQuant':
            return ColorQuant(self.severity, self.norm_config)
        elif self.corruption == 'MotionBlur':
            return functools.partial(corrupt, corruption_name='motion_blur', severity=self.severity)
        elif self.corruption == 'Brightness':
            return functools.partial(corrupt, corruption_name='brightness', severity=self.severity)
        elif self.corruption == 'Fog':
            return functools.partial(corrupt, corruption_name='fog', severity=self.severity)
        elif self.corruption == 'Snow':
            return functools.partial(corrupt, corruption_name='snow', severity=self.severity)
        elif self.corruption == 'Pixelate':
            return functools.partial(corrupt, corruption_name='pixelate', severity=self.severity)
        elif self.corruption == 'GlassBlur':
            return functools.partial(corrupt, corruption_name='glass_blur', severity=self.severity)
        elif self.corruption == 'ZoomBlur':
            return functools.partial(corrupt, corruption_name='zoom_blur', severity=self.severity)
        elif self.corruption == 'DefocusBlur':
            return functools.partial(corrupt, corruption_name='defocus_blur', severity=self.severity)


class DefocusBlur(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Defocus Blur'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'defocus_blur')


class GlassBlur(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Glass Blur'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'glass_blur')

class MotionBlur(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Motion Blur'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'motion_blur')


class ZoomBlur(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Zoom Blur'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'zoom_blur')


class GaussianNoise(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Gaussian Noise'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'gaussian_noise')


class ImpulseNoise(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Impulse Noise'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'impulse_noise')


class ShotNoise(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'ISO Noise'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'shot_noise')


class ISONoise(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Shot Noise'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'iso_noise')
        self.corrupt_func = self._get_corrupt_func()

    def _get_corrupt_func(self):
        return functools.partial(self.iso_noise, severity=self.severity)

    def iso_noise(self, x, severity):
        c_poisson = 25
        x = np.array(x) / 255.
        x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255.
        c_gauss = 0.7 * [.08, .12, 0.18, 0.26, 0.38][severity]
        x = np.array(x) / 255.
        x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255.
        return np.uint8(x)


class Brightness(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Brightness'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'brightness')


class LowLight(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Dark'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'dark')
        self.corrupt_func = self._get_corrupt_func()

    def _get_corrupt_func(self):
        return functools.partial(self.low_light, severity=self.severity)

    def imadjust(self, x, a, b, c, d, gamma=1):
        y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
        return y

    def poisson_gaussian_noise(self, x, severity):
        c_poisson = 10 * [60, 25, 12, 5, 3][severity]
        x = np.array(x) / 255.
        x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255
        c_gauss = 0.1 * [.08, .12, 0.18, 0.26, 0.38][severity]
        x = np.array(x) / 255.
        x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255
        return np.uint8(x)

    def low_light(self, x, severity):
        c = [0.60, 0.50, 0.40, 0.30, 0.20][severity]
        # c = [0.50, 0.40, 0.30, 0.20, 0.10][severity-1]
        x = np.array(x) / 255.
        x_scaled = self.imadjust(x, x.min(), x.max(), 0, c, gamma=2.) * 255
        x_scaled = self.poisson_gaussian_noise(x_scaled, severity=severity)
        return x_scaled


class Fog(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Fog'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'fog')

class Snow(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Snow'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'snow')

class CameraCrash:
    def __init__(self, severity, norm_config):
        """ 
        Create corruptions: 'Camera Crash'.
        """
        self.severity = severity
        assert severity >= 1 and severity <= 5, f"Corruption Severity should between (1, 5), now {severity}"
        self.norm_config = norm_config
        self.corruption = 'cam_crash'
        self.crash_camera = self.get_crash_camera()

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): [B, M, C, H, W]
        """
        mean = self.norm_config['mean']
        std = self.norm_config['std']

        # if img is numpy array, to tensor 
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        img = deepcopy(img)
        B, M, C, H, W = img.size()
        img = img.permute(0, 1, 3, 4, 2) # [B, M, C, H, W] => [B, M, H, W, C]

        # pixel value [0, 255]
        assert img.min() >= 0 and img.max() <= 255, "Image pixel out of range"

        for b in range(B):
            for m in self.crash_camera:
                img[b, m] = 0

        assert img.min() >= 0 and img.max() <= 255, "Image pixel out of range"

        return img.numpy()

    def get_crash_camera(self):
        crash_camera = np.random.choice([0, 1, 2, 3, 4, 5], size=self.severity)
        return list(crash_camera)


class FrameLost():
    def __init__(self, severity, norm_config):
        """ 
        Create corruptions: 'Frame Lost'.
        """
        self.severity = severity
        assert severity >= 1 and severity <= 5, f"Corruption Severity should between (1, 5), now {severity}"
        self.norm_config = norm_config
        self.corruption = 'frame_lost'

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): [B, M, C, H, W]
        """
        mean = self.norm_config['mean']
        std = self.norm_config['std']

        # if img is numpy array, to tensor 
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        
        img = deepcopy(img)
        B, M, C, H, W = img.size()
        img = img.permute(0, 1, 3, 4, 2) # [B, M, C, H, W] => [B, M, H, W, C]
        
        # pixel value [0, 255]
        assert img.min() >= 0 and img.max() <= 255, "Image pixel out of range"

        for b in range(B):
            for m in range(M):
                if np.random.rand() < (self.severity * 1. / 6.):
                    img[b, m] = 0

        assert img.min() >= 0 and img.max() <= 255, "Image pixel out of range"

        return img.numpy()


class ColorQuant(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Color Quantization'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'color_quant')

    def _get_corrupt_func(self):
        return functools.partial(self.color_quant, severity=self.severity)

    def color_quant(self, x, severity):
        bits = 5 - severity
        x = Image.fromarray(np.uint8(x))
        x = PIL.ImageOps.posterize(x, bits)
        return np.asarray(x)


class Pixlate(BaseCorruption):
    def __init__(self, severity, norm_config):
        """
        Create corruptions: 'Pixelate'.
        Args: 
            severity (int): severity of corruption, range (1, 5)
        """
        super().__init__(severity, norm_config, 'pixelate')