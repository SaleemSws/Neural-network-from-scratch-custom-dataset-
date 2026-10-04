# Custom U-Net architecture

All convolution weights are Kaiming-random initialized; all layers are trained.
There is no pretrained backbone. Total trainable parameters: **122,465**.

```mermaid
flowchart TD
    A["RGB input: N x 3 x 36 x 64"] --> E1["Encoder 1: 3 -> 8; 36 x 64"]
    E1 --> P1["MaxPool: 18 x 32"] --> E2["Encoder 2: 8 -> 16; 18 x 32"]
    E2 --> P2["MaxPool: 9 x 16"] --> E3["Encoder 3: 16 -> 32; 9 x 16"]
    E3 --> P3["MaxPool: 4 x 8"] --> B["Bottleneck: 32 -> 64; 4 x 8"]
    B --> U3["Bilinear resize to 9 x 16 + concatenate"]
    E3 --> U3 --> D3["Decoder 3: 96 -> 32; 9 x 16"]
    D3 --> U2["Bilinear resize to 18 x 32 + concatenate"]
    E2 --> U2 --> D2["Decoder 2: 48 -> 16; 18 x 32"]
    D2 --> U1["Bilinear resize to 36 x 64 + concatenate"]
    E1 --> U1 --> D1["Decoder 1: 24 -> 8; 36 x 64"]
    D1 --> H["1 x 1 Conv: 8 -> 1 logits; 36 x 64"]
    H --> S["Sigmoid + probability >= 0.5: binary lane mask"]
```

Each encoder/decoder block has two 3x3 convolutions with padding=1,
each followed by GroupNorm (4 groups) and ReLU. The head has a bias.
GroupNorm supports small batches without running batch statistics. Three
pooling levels provide context with a small footprint. Skip connections preserve
spatial detail. Bilinear resize to the exact skip shape avoids mismatches for
36x64, 48x48 and odd dimensions, and adds no transpose-convolution parameters.
The one-channel logit head fits binary semantic segmentation and BCEWithLogits.

The model can accept larger spatial dimensions >=8, but this experiment trains
and infers at 36x64. Original 1280x720 images are resized before the network;
binary predictions are restored to source size with nearest-neighbor interpolation.
This restoration does not recover detail lost at the low input resolution.
