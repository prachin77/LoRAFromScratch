# Changelog

## 2026-10-27

### Added

- Introduced `LoRALayerBase` as a shared base for LoRA-related settings and weight-loading behavior.
- Added constructor options for the adapter rank, alpha, dropout probability, and rsLoRA scaling mode, along with logic intended to calculate the scaling factor from alpha and rank.
- Added optional dropout handling: the code creates a dropout module for a positive probability and otherwise uses an identity function.
- Added `load_pretrained_weights` to copy a pretrained linear layer's weight and, when present, its bias into the layer.
- Introduced `LoRALinearLayer` as an `nn.Linear` subclass. It accepts the standard linear-layer dimensions and bias option, as well as LoRA configuration options.
- Froze the base linear layer's weight so it is excluded from gradient updates.
- Added trainable `lora_A` and `lora_B` parameters with dimensions `in_features × rank` and `rank × out_features`, respectively. Both tensors are created from zeros; the code then attempts to initialize `lora_A` with Kaiming uniform initialization, while `lora_B` remains zero-initialized.

### Current scope

- This remains an implementation scaffold: the adapter matrices are not yet used to compute or add a LoRA update in a `forward` method. The base-layer bias, when enabled, is not frozen.

### Known issues

- LoRA configuration initialization is currently broken: `LoRALayerBase` assigns `self.lora_alpha` to itself instead of using the `lora_alpha` argument, and `LoRALinearLayer` calls the base constructor as `LoRALayerBase(self, ...)` rather than invoking its initializer.
- The Kaiming initialization call uses `nn.init.kaiming_uniform`; PyTorch's initializer is named `nn.init.kaiming_uniform_`.
