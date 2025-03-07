# Quantised MobileNetV2_5 Model (BETA)

WARNING: CURRENTLY NOT COMPLETELY FUNCTIONAL- Quantisation Observer has to be implemented

A custom implementation of MobileNet architecture (between v2 and v3) with quantization support for efficient inference with minimal cost to accuracy. 

## Overview

A PyTorch implementation of a modified MobileNetV2 architecture with selected features from MobileNetV3. The model includes:

* Squeeze and Excitation layers for improved feature representation
* Quantization support for reduced model size and faster inference
* Configurable width multiplier for model scaling

## Features

- **Hybrid Architecture**: Combines MobileNetV2's inverted residual blocks with MobileNetV3's squeeze-excitation mechanism
- **Quantization-Aware Training**: Built-in support for QAT to optimize model size and inference speed
- **Customizable**: Adjustable width multiplier and other parameters for different deployment scenarios

## Model Architecture

The model follows the standard MobileNet architecture with the following modifications:

1. Uses inverted residual blocks as the main building component
2. Incorporates squeeze-excitation layers for improved channel attention
3. Uses Leaky ReLU activation instead of hardswish (Matter of personal preference)
4. Includes quantization stubs for easy conversion to quantized models

## Usage

### Basic Usage

```python
from MobileNetV3 import MobileNetV2_5

# Create model
model = MobileNetV2_5(class_num=2)

# Forward pass
output = model(input_tensor)
```

### Quantization (QAT)

```python
from MobileNetV3 import QuantizableMobileNetV2_5

# Create quantizable model
model = QuantizableMobileNetV2_5(class_num=2)

# Set quantization configuration
model.qconfig = torch.ao.quantization.default_qconfig

# Prepare for quantization-aware training
torch.ao.quantization.prepare_qat(model, inplace=True)

# Train the model
# ...

# Convert to fully quantized model
quantized_model = torch.ao.quantization.convert(model.eval(), inplace=False)
```

## Training

The repository includes a complete training pipeline with:

- Data loading using `CrosswalkDataset`, using tools shown in sightLinks-Dev
- Epoch-based training with quantization-aware adjustments
- Progressive freezing of batch normalization statistics and quantization parameters
- Automatic model saving and conversion

## Model Size

The quantized model achieves significant size reduction compared to the full-precision version:

```
# Check model size
print_size_of_model(quantized_model)  # 2.81MB for oru MobileNetV2_5 mdoel
```

## Requirements

- PyTorch 1.13+
- torchvision
- matplotlib (for visualization)

## Notes

- This implementation uses `qnnpack` as the quantization backend.
- The model is designed for binary classification by default but can be adjusted for multi-class problems
