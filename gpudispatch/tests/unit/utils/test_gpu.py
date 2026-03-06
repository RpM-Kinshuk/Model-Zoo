"""Tests for GPU detection utilities."""

import pytest
from unittest.mock import patch, MagicMock
from gpudispatch.utils.gpu import (
    detect_gpus,
    get_gpu_memory_usage,
    is_gpu_available,
    GPUInfo,
)


class TestGPUInfo:
    def test_gpu_info_creation(self):
        info = GPUInfo(
            index=0,
            name="NVIDIA A100",
            memory_total_mb=81920,
            memory_used_mb=1024,
            utilization_percent=45,
        )
        assert info.index == 0
        assert info.name == "NVIDIA A100"
        assert info.memory_free_mb == 81920 - 1024

    def test_gpu_info_memory_free(self):
        info = GPUInfo(
            index=0,
            name="Test GPU",
            memory_total_mb=16000,
            memory_used_mb=4000,
            utilization_percent=0,
        )
        assert info.memory_free_mb == 12000


class TestDetectGPUs:
    @patch('gpudispatch.utils.gpu.gpustat')
    def test_detect_gpus_success(self, mock_gpustat):
        mock_gpu = MagicMock()
        mock_gpu.index = 0
        mock_gpu.name = "NVIDIA A100"
        mock_gpu.memory_total = 81920
        mock_gpu.memory_used = 1024
        mock_gpu.utilization = 45

        mock_collection = MagicMock()
        mock_collection.gpus = [mock_gpu]
        mock_gpustat.GPUStatCollection.new_query.return_value = mock_collection

        gpus = detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].index == 0
        assert gpus[0].name == "NVIDIA A100"

    @patch('gpudispatch.utils.gpu.gpustat')
    def test_detect_gpus_empty(self, mock_gpustat):
        mock_collection = MagicMock()
        mock_collection.gpus = []
        mock_gpustat.GPUStatCollection.new_query.return_value = mock_collection

        gpus = detect_gpus()
        assert len(gpus) == 0

    @patch('gpudispatch.utils.gpu.gpustat')
    def test_detect_gpus_error(self, mock_gpustat):
        mock_gpustat.GPUStatCollection.new_query.side_effect = Exception("No GPU")

        gpus = detect_gpus()
        assert len(gpus) == 0


class TestIsGPUAvailable:
    @patch('gpudispatch.utils.gpu.detect_gpus')
    def test_gpu_available_below_threshold(self, mock_detect):
        mock_detect.return_value = [
            GPUInfo(0, "GPU0", 16000, 400, 0),
        ]
        assert is_gpu_available(0, memory_threshold_mb=500)

    @patch('gpudispatch.utils.gpu.detect_gpus')
    def test_gpu_not_available_above_threshold(self, mock_detect):
        mock_detect.return_value = [
            GPUInfo(0, "GPU0", 16000, 8000, 50),
        ]
        assert not is_gpu_available(0, memory_threshold_mb=500)

    @patch('gpudispatch.utils.gpu.detect_gpus')
    def test_gpu_not_found(self, mock_detect):
        mock_detect.return_value = [
            GPUInfo(0, "GPU0", 16000, 0, 0),
        ]
        assert not is_gpu_available(1)
