# Changelog

## 2026-10-27

### Added

- Introduced `LoRALayerBase` as a shared base for LoRA-related settings and weight-loading behavior.
- Added constructor options for the adapter rank, alpha, dropout probability, and rsLoRA scaling mode, along with logic intended to calculate the scaling factor from alpha and rank.
- Added optional dropout handling: the code creates a dropout module for a positive probability and otherwise uses an identity function.
- Added `load_pretrained_weights` to copy a pretrained linear layer's weight and, when present, its bias into the layer.
- Introduced `LoRALinearLayer` as an `nn.Linear` subclass. It accepts the standard linear-layer dimensions and bias option, as well as LoRA configuration options.
- Froze the base linear layer's weight so it is excluded from gradient updates.
- Added trainable `lora_A` and `lora_B` parameters with dimensions `in_features × rank` and `rank × out_features`, respectively. `lora_A` is initialized with Kaiming uniform initialization, while `lora_B` remains zero-initialized.
- Implemented `LoRALinearLayer.forward` to compute the frozen linear layer output, apply dropout to the input for the adapter path, transform it with the low-rank A/B matrices and scaling factor, and add the adapter output to the base output.
- Imported PyTorch's neural-network functional API for the base linear operation in the forward pass.

### Current scope

- The base linear layer's bias remains trainable when enabled; only its weight is frozen.

## 2026-10-28

### Added

- Added `_merge_weights` to combine the frozen linear weight with the scaled LoRA update and load the merged parameters into a standard `nn.Linear` layer.
- Updated the script example to instantiate the adapter with rank 2, run a random input through its forward pass, and print the output.

### Known limitation

- `_merge_weights` currently creates and returns the merged layer only when the original layer has a bias. For a bias-free layer, the method returns `None`.

## 2026-10-30

### Added

- Added `LoRAEmbedding`, which freezes the base embedding weights and adds a trainable low-rank adapter to embedding outputs.

### Changed

- Initialized `LoRALinearLayer.lora_B` with Kaiming uniform values instead of zeros.
- Updated the script example to instantiate a `LoRAEmbedding` layer.
