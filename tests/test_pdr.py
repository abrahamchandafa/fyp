"""Tests for pdr module (PDRNet model)."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Load pdr/model.py by file path
pdr_path = Path(__file__).parent.parent / "pdr" / "model.py"
spec = importlib.util.spec_from_file_location("pdr_module", str(pdr_path))
pdr_model = importlib.util.module_from_spec(spec)
sys.modules["pdr_module"] = pdr_model
spec.loader.exec_module(pdr_model)

PDRNet = pdr_model.PDRNet
_ConvBlock = pdr_model._ConvBlock


class TestConvBlock:
    """Test _ConvBlock module."""

    def test_conv_block_creation(self):
        """Test creating a _ConvBlock."""
        block = _ConvBlock(in_ch=2, out_ch=32)
        assert isinstance(block, nn.Module)

    def test_conv_block_forward_pass(self):
        """Test _ConvBlock forward pass."""
        block = _ConvBlock(in_ch=2, out_ch=32)
        x = torch.randn(1, 2, 64, 64)
        y = block(x)
        assert y.shape == (1, 32, 64, 64)

    def test_conv_block_output_channels(self):
        """Test _ConvBlock output channels."""
        block = _ConvBlock(in_ch=16, out_ch=64)
        x = torch.randn(1, 16, 32, 32)
        y = block(x)
        assert y.shape[1] == 64


class TestPDRNetCreation:
    """Test PDRNet model creation."""

    def test_pdrnet_creation_default(self):
        """Test creating PDRNet with default parameters."""
        model = PDRNet()
        assert isinstance(model, nn.Module)

    def test_pdrnet_creation_custom_channels(self):
        """Test creating PDRNet with custom channels."""
        model = PDRNet(in_channels=3, out_channels=2)
        assert isinstance(model, nn.Module)

    def test_pdrnet_creation_custom_features(self):
        """Test creating PDRNet with custom feature dimensions."""
        model = PDRNet(features=(16, 32, 64, 128))
        assert isinstance(model, nn.Module)


class TestPDRNetForwardPass:
    """Test PDRNet forward pass."""

    def test_pdrnet_forward_pass_basic(self):
        """Test PDRNet forward pass with basic input."""
        model = PDRNet()
        x = torch.randn(1, 2, 640, 480)
        y = model(x)
        assert y.shape == (1, 1, 640, 480)

    def test_pdrnet_output_dtype(self):
        """Test PDRNet output is float32."""
        model = PDRNet()
        x = torch.randn(1, 2, 640, 480, dtype=torch.float32)
        y = model(x)
        assert y.dtype == torch.float32

    def test_pdrnet_output_range(self):
        """Test PDRNet output is in [0, 1] due to sigmoid."""
        model = PDRNet()
        x = torch.randn(1, 2, 640, 480)
        y = model(x)
        assert torch.all(y >= 0) and torch.all(y <= 1)

    def test_pdrnet_forward_pass_batch(self):
        """Test PDRNet forward pass with batch input."""
        model = PDRNet()
        x = torch.randn(4, 2, 640, 480)
        y = model(x)
        assert y.shape == (4, 1, 640, 480)

    def test_pdrnet_forward_pass_small_input(self):
        """Test PDRNet forward pass with smaller input."""
        model = PDRNet()
        x = torch.randn(1, 2, 160, 120)
        y = model(x)
        assert y.shape == (1, 1, 160, 120)


class TestPDRNetGradients:
    """Test PDRNet gradient computation."""

    def test_pdrnet_backward_pass(self):
        """Test PDRNet backward pass computes gradients."""
        model = PDRNet()
        x = torch.randn(1, 2, 160, 120, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_pdrnet_parameters_have_gradients(self):
        """Test that model parameters can accumulate gradients."""
        model = PDRNet()
        x = torch.randn(1, 2, 160, 120)
        y = model(x)
        loss = y.sum()
        loss.backward()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestPDRNetArchitecture:
    """Test PDRNet architecture properties."""

    def test_pdrnet_has_encoders(self):
        """Test PDRNet has encoder layers."""
        model = PDRNet()
        assert hasattr(model, "encoders")
        assert len(model.encoders) > 0

    def test_pdrnet_has_decoders(self):
        """Test PDRNet has decoder layers."""
        model = PDRNet()
        assert hasattr(model, "decoders")
        assert len(model.decoders) > 0

    def test_pdrnet_has_pooling(self):
        """Test PDRNet has pooling layers."""
        model = PDRNet()
        assert hasattr(model, "pools")
        assert len(model.pools) > 0

    def test_pdrnet_has_upconv(self):
        """Test PDRNet has up-convolution layers."""
        model = PDRNet()
        assert hasattr(model, "upconvs")
        assert len(model.upconvs) > 0

    def test_pdrnet_has_bottleneck(self):
        """Test PDRNet has bottleneck layer."""
        model = PDRNet()
        assert hasattr(model, "bottleneck")

    def test_pdrnet_has_head(self):
        """Test PDRNet has final head layer."""
        model = PDRNet()
        assert hasattr(model, "head")


class TestPDRNetFeatureDimensions:
    """Test PDRNet with different feature dimensions."""

    def test_pdrnet_shallow_features(self):
        """Test PDRNet with shallow feature dimensions."""
        model = PDRNet(features=(8, 16, 32))
        x = torch.randn(1, 2, 160, 120)
        y = model(x)
        assert y.shape == (1, 1, 160, 120)

    def test_pdrnet_deep_features(self):
        """Test PDRNet with deep feature dimensions."""
        model = PDRNet(features=(32, 64, 128, 256, 512))
        x = torch.randn(1, 2, 160, 120)
        y = model(x)
        assert y.shape == (1, 1, 160, 120)


class TestPDRNetOddDimensions:
    """Test PDRNet handles odd dimensions gracefully."""

    def test_pdrnet_odd_height(self):
        """Test PDRNet with odd height input."""
        model = PDRNet()
        x = torch.randn(1, 2, 159, 480)
        y = model(x)
        assert y.shape[2] == 159  # height preserved

    def test_pdrnet_odd_width(self):
        """Test PDRNet with odd width input."""
        model = PDRNet()
        x = torch.randn(1, 2, 640, 479)
        y = model(x)
        assert y.shape[3] == 479  # width preserved

    def test_pdrnet_both_odd(self):
        """Test PDRNet with both odd dimensions."""
        model = PDRNet()
        x = torch.randn(1, 2, 159, 479)
        y = model(x)
        assert y.shape[2] == 159
        assert y.shape[3] == 479


class TestPDRNetEval:
    """Test PDRNet evaluation mode."""

    def test_pdrnet_eval_mode(self):
        """Test switching PDRNet to eval mode."""
        model = PDRNet()
        model.eval()
        x = torch.randn(1, 2, 160, 120)
        y = model(x)
        assert y.shape == (1, 1, 160, 120)

    def test_pdrnet_train_mode(self):
        """Test switching PDRNet to train mode."""
        model = PDRNet()
        model.train()
        x = torch.randn(1, 2, 160, 120)
        y = model(x)
        assert y.shape == (1, 1, 160, 120)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
