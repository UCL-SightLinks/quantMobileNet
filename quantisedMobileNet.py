# Read these to catch up on what is (trying to at least) being done here
# https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
# https://pytorch.org/docs/stable/quantization.html#model-preparation-for-eager-mode-static-quantization

# Torch implementation of these models - mine is heavily based on these with some minor adjustments
#
# I've added squeeze and excitation layers to the MobileNetV2, a feature of MobileNetV3, but I did not put in 
# NAS (unnecessary since we're not optimising for mobile) or hardswish (because I prefer ReLU/ think it is better)
# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py#L117
# https://github.com/pytorch/vision/blob/11bf27e37190b320216c349e39b085fb33aefed1/torchvision/models/mobilenetv3.py#L56

# This is an adapted version of MobileNet, somewhere between versions 2/3, as some features of 3 were not required. There are
# also some additions for our particular use case from miscallaneous sources

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torch.ao.quantization import QuantStub, DeQuantStub
from torchvision.models.mobilenetv2 import _make_divisible

import time
import os

# Squeeze: summarising global context by pooling feature maps into a single value
# Excitation: Learning attention weights for each channel to prioritise the most relevant ones
class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels:int, squeeze_factor: int = 4):
        super().__init__()
        # If channels are a multiple of 8, they're optimised by the hardware
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.squeeze = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.unsqueeze = nn.Conv2d(squeeze_channels, input_channels, 1)
    
    # Scale returns the feature attention map, how much attention should be payed to each input layer, in range [0, 1]
    # Inplace is used to save memory on operations - it might not be necessary in our case since we aren't using edge devices
    def _scale(self, input: Tensor, inplace=bool) -> Tensor:
        # Squeeze
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.squeeze(scale)
        # Excite
        scale = self.relu(scale)
        scale = self.unsqueeze(scale)
        return F.hardsigmoid(scale, inplace=inplace)
    
    def forward(self, input: Tensor) -> Tensor:
        return nn.quantized.FloatFunctional().mul(self._scale(input, True), input)


# The basic building block of our convolutional neural network
# - qconfig should automatically insert fakeQuantisation operations during training, so there is no need to manually place them now
class ConvBNReLu(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            # No point applying a bias (constant addative term) if the next layer is a batch normalisation layer
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=True)
        )
 

# Like typical residual blocks but uses inverse narrow->wide->narrow, with Depth-wise convolutions instead of normal,
# to reduce the number of parameters required compared to the usual residual blocks
class InvertedResidual(nn.Module):
    def __init__(self, inpt, oupt, stride, expnd_ratio, kernel_size=3, se_layer=None):
        super().__init__()

        self.stride = stride
        assert stride in [1, 2]

        intermediate_channels = int(round(inpt * expnd_ratio))
        # If the stride != 1, downsampling occurs so cannot be true. 
        self.use_residual = (stride==1) and (inpt==oupt)
        # Squeeze and excitation layer - applied after the dw and pw convolutions, but before the residual
        self.se_layer = se_layer if se_layer else None

        layers = []

        if expnd_ratio != 1:
            # Pointwise convolution to increase the channels
            layers.append(ConvBNReLu(inpt, intermediate_channels, kernel_size=1))

        layers.extend([
            # Depthwise convolution - each channel is convoled on an independent basis 
            ConvBNReLu(intermediate_channels, intermediate_channels, stride=stride, groups=intermediate_channels),
            # point-wise convolution - linear combination to reduce layers back to the expected number
            nn.Conv2d(intermediate_channels, oupt, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oupt, momentum=0.25)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        outpt = self.conv(x)
        if self.se_layer is not None:
            outpt = self.se_layer(outpt)

        if self.use_residual:
            return x + outpt
        else:
            return outpt
        

# Same as the inverted residual, but replaces addition with a quantizable friendly operation 
class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, inpt, outpt, stride, expnd_ratio, se_layer=None):
        super().__init__(inpt, outpt, stride, expnd_ratio, se_layer=se_layer)
        self.skip_add = nn.quantized.FloatFunctional()

    # Overwrites the forwarding to use a quantizable friendly version of the addition
    def forward(self, x):
        outpt = self.conv(x)
        if self.se_layer is not None:
            outpt = self.se_layer(outpt)

        if self.use_residual:
            return self.skip_add.add(x, outpt)
        else:
            return outpt

# The MobileNetV2 Architecture + some features from V3 (squeeze and excitation) but I didn't add NAS since we aren't running this on mobile
# And I prefer ReLU over hardswish
class MobileNetV2_5(nn.Module):
    def __init__(self, class_num=2, width_mult=1.0, round_nearest=8):
        super().__init__()

        layers = []

        input_channel = 32
        last_channel = 1280

        # Just straight up copying this from the torchvision implementation
        self.residual_params =  [
                # expnd_ratio, outpt_channels, num_blocks, stride
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        
        first_conv_output_channels = _make_divisible(self.residual_params[0][1] *width_mult, round_nearest)
        layers.append(
            ConvBNReLu(3,
                first_conv_output_channels,
                kernel_size=3,
                stride=2,
                )
        )
        prev_input_channels = first_conv_output_channels

        # Main body of feature extraction
        for expnd, oupt_c, num_blocks, strd in self.residual_params:
            # output channels must be a multiple of 8 for hardware optimisation
            output_channel = _make_divisible(oupt_c * width_mult, round_nearest)
            
            for i in range(num_blocks):
                stride = strd if i == 0 else 1
                se_layer = SqueezeExcitation(oupt_c) if i == 0 else None
                layers.append(QuantizableInvertedResidual(prev_input_channels, output_channel, stride, expnd_ratio=expnd, se_layer=se_layer))
                prev_input_channels = output_channel
            
        
        self.last_channel = _make_divisible(last_channel * max(width_mult, 1.0), round_nearest)

        # We could put this in the classifier, but I want that to be lightweight so that we could do transfer learning only on the head and 
        # the feature extraction part of the model.
        layers.append(
            ConvBNReLu(prev_input_channels, self.last_channel, kernel_size=1)
        )

        self.feature_extraction = nn.Sequential(*layers)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.125),
            nn.Linear(last_channel, class_num)
        )

        # This bit is also just straight up copied from torch's implementation - I'm not touching it in case it gets messed up
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extraction(x)
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        print("eyo")
        x = self.classifier(x)

        return x

class QuantizableMobileNetV2_5(MobileNetV2_5):
    def __init__(self, class_num=2, width_mult=1.0, round_nearest=8):
        super().__init__(class_num=class_num, width_mult=width_mult, round_nearest=round_nearest)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.feature_extraction(x)

        # This was for debugging errors in shape of feature maps as they pass through - not deleting incase useful later
        # for idx, layer in enumerate(self.feature_extraction):
        #     x = layer(x)
        #     print(f"Feature extraction layer {idx}, output shape: {x.shape}")
    
        x = self.avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x= self.dequant(x)
        return x


def train_single_epoch(model, loss_fnc, optimiser, data_loader, device):
    model.train()
    running_loss = 0

    for images, labels in data_loader:
        start_time = time.time()
        print(".", end=" ")
        
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = loss_fnc(preds, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        print(f"{(time.time() - start_time):.2f}", end=" ")

    print(f"loss of {running_loss}")
    return

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def adjust_quantisation_engine():
    # Adjust according to what your device supports
    print(torch.backends.quantized.supported_engines)
    torch.backends.quantized.engine = 'qnnpack'

def train_model(model, dataloader, loss_function, optimiser, epoch_number=25):
    for epoch in range(epoch_number):
        train_single_epoch(model, loss_function, optimiser, dataloader, torch.device('cpu'))
        if epoch > 3:
            # Freeze quantizer parameters
            model.apply(torch.ao.quantization.disable_observer)
        if epoch > 2:
            # Freeze batch norm mean and variance estimates
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # Check the accuracy after each epoch
        quantized_model = torch.ao.quantization.convert(model.eval(), inplace=False)
        quantized_model.eval()

        # Saving each intermediary model since they're so small, and this lets load up any of them for performace difference examples later
        torch.save(quantized_model.state_dict(), "quantStateDict"+str(epoch)+".pth")

        print(f"the above was Epoch {epoch} of {epoch_num} \nThe model has a size of", end=" ")
        print_size_of_model(quantized_model)
    
    return model
