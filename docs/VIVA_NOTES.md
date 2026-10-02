# Viva notes: what each part does and why

**Runtime corruption** (`corruptions.py`, `data_pets.py`): the dataset class draws a corruption type and severity every time an image is loaded, so the model never sees the same noisy image twice. Validation/test use stored manifests so every run compares on identical inputs.

**UDAE** (`models.py: UDAE`): 4 stride-2 conv stages shrink 128x128 to 8x8 with more channels, a 1x1 conv squeezes to `latent_ch` channels (the bottleneck, 16x8x8 = 1024 numbers vs 49,152 input values), the decoder upsamples back. No skip connections, otherwise the decoder could copy the noisy input. Loss = alpha*L1 + (1-alpha)*(1-SSIM): L1 for pixel accuracy, SSIM for structure.

**Why the Optuna objective is L1 + (1 - SSIM) and not the training loss**: the training loss contains alpha, which Optuna tunes. If the objective used alpha too, Optuna would just pick whatever alpha makes the number small. A fixed objective compares models fairly.

**Classifier + specialists** (`task2.py`): classifier sees balanced batches (exactly 1/4 per class) so it is not biased to one class. Each specialist trains only on its corruption. At inference: clean -> identity bypass, else call the predicted specialist. Oracle routing uses the true label (upper bound for the specialists); predicted routing is the real system. The gap between them is the cost of classifier mistakes.

**Soft MoE** (`task3.py`, `models.py: SoftMoE`): gate = classifier logits divided by temperature tau, softmax gives 4 weights, output = weighted sum of identity + 3 experts. Low tau = near-hard routing, high tau = blended. Gate/experts start from Task 2 (random start tends to collapse). Warm-up trains only the gate, then everything fine-tunes at a lower LR. Loss adds cross-entropy (keeps the gate tied to the true corruption) and a balance term (stops one expert taking everything). Trials are pruned if routing collapses.

**cGAN** (`models.py`, `task4.py`): U-Net generator takes photo + style; style is a learned embedding fed in as input channels and as FiLM (scale/shift) in every decoder block. PatchGAN discriminator judges local patches of (photo, sketch, style). D learns real=1, fake=0; G tries to make D say 1 and stay close to the real sketch (L1 weight lambda, 100 as the starting point from pix2pix). Augmentation is applied identically to photo and sketch, otherwise the pair no longer matches.

**ONNX parity** (`export.py`): the same random inputs go through PyTorch and ONNX Runtime; the max absolute difference must be tiny (about 1e-6). The MoE export bakes tau in as a constant.

**Likely questions**: why no skip connections? why SSIM? what is oracle vs predicted routing? why balance loss? what happens if tau is large? why identical augmentation for both images? why is the GAN objective only a proxy? what does PSNR hide for occlusion?
