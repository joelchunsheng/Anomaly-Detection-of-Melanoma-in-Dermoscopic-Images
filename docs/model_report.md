# Model Architectures, Training Iterations, and Evaluation

This section details the development, rationale, and performance of various deep learning architectures for binary melanoma classification on the HAM10000 dataset.

---

## 1. Custom CNN Experiments (`notebooks/cnn/`)

### Architecture & Strategy
This series explores custom CNN architectures trained from scratch as a baseline before any transfer learning. Four variants were tested: a simple 3-conv baseline, the same with class-weighted loss, a BatchNorm variant, a deeper version, and a residual variant. All use Adam + BCEWithLogitsLoss with `pos_weight`.

### Results Summary
| # | Notebook | Architecture | Key Change | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|
| 01 | `01.cnn_baseline` | Simple CNN | No weighting, no BN | 0.5676 | 0.8246 | 0.2527 | 0.8449 |
| 02 | `02.cnn_baseline_weighted` | Simple CNN | + pos_weight | 0.5670 | **0.8713** | 0.2365 | 0.8471 |
| 03 | `03.cnn_batchnorm_weighted` | CNN + BatchNorm | + BatchNorm | **0.5900** | 0.7895 | **0.2935** | 0.8520 |
| 04 | `04.cnn_deeper_batchnorm_weighted` | Deeper CNN + BN | + depth | 0.5525 | 0.7076 | 0.2944 | 0.8538 |
| 05 | `05.cnn_residual_batchnorm_weighted` | Residual CNN | + skip connections | 0.5814 | 0.8187 | 0.2692 | **0.8672** |

### Iteration Intuition & Reasoning
- **Class weighting (01 → 02)**: Adding `pos_weight` shifted the model toward higher recall (0.8246 → 0.8713) but produced no F2 gain — it adjusts the operating point without improving the underlying discriminative ability.
- **BatchNorm (02 → 03)**: Normalising activations stabilised training and improved precision-recall balance, yielding the best F2 (0.5900). BatchNorm acts as implicit regularization on a small dataset, reducing internal covariate shift.
- **Depth alone (03 → 04)**: Adding layers without skip connections hurt recall and F2 despite a marginal AUC gain — the vanishing gradient problem limits deeper plain CNNs on small datasets.
- **Residual connections (04 → 05)**: Skip connections resolved gradient flow, producing the best AUC (0.8672) and strong recall (0.8187). However precision fell, giving a lower F2 than the simpler BatchNorm model — the residual network's higher capacity introduces more aggressive positive predictions.

### Cross-Architecture Context
The CNN series establishes the from-scratch baseline: AUC peaks at 0.8672, well below what transfer learning achieves. The progression confirms that **architectural improvements** (BatchNorm, residual connections) provide meaningful gains even without pretraining, but the capacity ceiling of small custom CNNs is clear. The jump to transfer learning (DenseNet, ResNet) is motivated by this ceiling.

---

## 2. DenseNet-121 Experiments (`notebooks/densenet/`)

### Architecture & Strategy
DenseNet-121 trained from scratch, exploring the effect of class weighting, head architecture (BatchNorm vs Dropout), and pretrained frozen backbones. All experiments use AdamW + CosineAnnealingLR, 30 epochs.

### Results Summary
| # | Notebook | Pretrained | Frozen | Head Structure | Dropout | Pos Weight | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | `01.densenet121_baseline` | No | No | Linear | 0.0 | No | 0.5997 | 0.5771 | 0.8012 | 0.2724 | 0.8570 |
| 02 | `02.densenet121_baseline_weighted` | No | No | Linear | 0.0 | Yes | 0.5945 | 0.5837 | **0.8889** | 0.2460 | 0.8531 |
| 03 | `03.densenet121_batchnorm_weighted` | No | No | Linear → BN → ReLU → Linear | 0.0 | Yes | 0.5978 | **0.5964** | 0.8246 | **0.2831** | **0.8580** |
| 04 | `04.densenet121_dropout_weighted` | No | No | Linear → ReLU → Dropout → Linear | 0.5 | Yes | 0.6122 | 0.5882 | **0.8889** | 0.2500 | 0.8543 |
| 05 | `05.densenet121_pretrained_weighted` | Yes | Yes | Linear → ReLU → Dropout → Linear | 0.4 | Yes | 0.5814 | 0.5516 | 0.8070 | 0.2434 | 0.8449 |

### Iteration Intuition & Reasoning
- **Unweighted baseline (01)**: AUC of 0.857 shows DenseNet-121 features have reasonable transferability to dermoscopy even from scratch, but without `pos_weight` the model under-prioritises the minority class — a very low threshold (0.17) is needed to recover recall.
- **Adding pos_weight (02)**: Loss reweighting meaningfully improves recall (0.8012 → 0.8889) and balanced accuracy, confirming class imbalance was a key bottleneck. AUC dips marginally due to the loss landscape shift.
- **BatchNorm head (03)**: Best overall result — highest Test F2 (0.5964), highest AUC (0.8580), and best balanced accuracy. The hidden layer with BatchNorm adds capacity and stabilises activations without overfitting.
- **Dropout head (04)**: Matches recall (0.8889) but achieves lower F2 and AUC than the BatchNorm variant. Stochastic regularisation is less effective than normalisation for this dataset size when training from scratch.
- **Pretrained frozen backbone (05)**: Counterintuitively the worst configuration. Freezing the ImageNet backbone limits adaptation to dermoscopy-specific features, and the small trainable head lacks capacity to compensate. This challenges the assumption that pretrained features always help.

### Cross-Architecture Context
DenseNet-121 from scratch (best AUC 0.858) slightly outperforms the best custom CNN (0.867) but the gap is small — the dense connectivity pattern gives moderate feature reuse benefits. The pretrained-frozen experiment foreshadowed a finding later repeated in ResNet-50 and EfficientNet-B4: **frozen pretrained features underperform fine-tuned ones** on dermoscopy, where texture patterns differ meaningfully from ImageNet.

---

## 4. ResNet-18 Experiments (`notebooks/resnet/`)

### Architecture & Strategy
The ResNet-18 series focuses on adapting a lightweight residual network to dermoscopic images. For metadata experiments, a custom `ResNet18WithMetadata` architecture was used, which fuses 512-dim image features with a 32-dim encoding of patient metadata (age, sex, anatomical site) before the final classification head.

### Results Summary
| # | Notebook | Unfrozen | Trainable Params | Regularisation | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|
| 01 | `01.resnet_naive` | None (FC only) | 513 | None | 0.7068 | 0.6409 | 0.7076 | 0.4654 | 0.8900 |
| 02 | `02.resnet_unfreeze` | Full backbone | 11.2M | None | 0.6893 | 0.6280 | 0.7544 | 0.3761 | 0.8997 |
| 03 | `03.resnet_augmented_v1` | Full backbone | 11.2M | Strong Aug + WD=1e-4 + Scheduler | 0.6680 | 0.6660 | **0.8304** | 0.3717 | 0.9068 |
| 04 | `04.resnet_augmented_v2` | Full backbone | 11.2M | Dropout=0.4 + AdamW + TTA | 0.6835 | **0.6704** | 0.7778 | **0.4318** | **0.9133** |
| 05 | `05.resnet_l1_dropout_tta` | Full backbone | 11.2M | L1 (λ=1e-3) + Dropout=0.4 + AdamW + TTA | 0.6505 | 0.6507 | 0.7602 | 0.4127 | 0.9090 |
| 06 | `06.resnet_metadata` | Full backbone | 11.2M | ResNet18WithMetadata + Metadata + L1 | 0.6642 | 0.6442 | 0.7602 | 0.4000 | 0.9104 |

### Iteration Intuition & Reasoning
- **Naive Transferability (01)**: The high baseline AUC (0.89) confirms that ResNet-18's ImageNet features are remarkably descriptive for dermatological textures, but the model lacks the specificity to distinguish hard negatives (nevus) from melanoma without further tuning.
- **The Overfitting Challenge (02)**: Unfreezing the backbone without constraints led to a drop in Test F2. Without regularization, the 11.2M parameters became poorly calibrated — the model grew more aggressive in its positive predictions, increasing recall but at the cost of precision and overall F2.
- **Impact of Augmentation (03)**: Since skin lesions have no natural "up" or "down" orientation, introducing strong rotation and flip augmentation forced the model to become orientation-invariant. Augmentation also acts as implicit regularization by increasing the effective training set size, helping the model generalize better. This significantly boosted melanoma recall to **0.8304**.
- **Inference Stability (04)**: The combination of Dropout, AdamW, and Test-Time Augmentation (TTA) yielded the best balance. TTA acts as a "majority vote" across different views of the same lesion, smoothing out noisy individual predictions and recovering precision lost in earlier iterations.
- **L1 + Dropout (05)**: L1 regularization encourages sparsity, pressuring the model to rely on a smaller, more robust set of features. However, combining L1 with Dropout proved too aggressive — both Test F2 (0.6507 vs 0.6704) and precision (0.4127 vs 0.4318) regressed compared to Iteration 04, suggesting the dual regularization overly constrained the model's capacity.
- **Metadata Fusion (06)**: Adding patient context (age, sex) improved validation performance but regressed on the test set. The metadata fusion increased model complexity without a corresponding adjustment to regularization, and the relatively small metadata feature space (17-dim) may not have provided enough discriminative signal to justify the additional parameters in the fusion head.

### Cross-Architecture Context
ResNet-18 sets the project baseline as the simplest architecture. Its best model (Iter 04, AUC=0.9133, F2=0.6704) outperforms every ResNet-50 variant despite having fewer parameters. This foreshadows a recurring theme: on a small, imbalanced dataset like HAM10000, **parameter efficiency matters more than raw capacity**. The lightweight 11.2M-parameter backbone was easier to regularize and less prone to overfitting, making it a stronger starting point than the deeper ResNet-50.

---

## 5. ResNet-50 Experiments (`notebooks/resnet50/`)

### Architecture & Strategy
ResNet-50 utilizes a deeper bottleneck architecture. The `ResNet50WithMetadata` model fuses 2048-dim image features with patient metadata.

### Results Summary
| # | Notebook | Unfrozen | Trainable Params | Regularisation | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|
| 01 | `01.resnet50_frozen_baseline` | None (FC only) | 2K | Dropout | 0.5807 | 0.5469 | 0.8187 | 0.2349 | 0.8290 |
| 02 | `02.resnet50_partial_unfreeze` | `layer4` (full) | 15M | Dropout + WD=1e-4 | 0.6575 | 0.5852 | 0.6667 | 0.3931 | 0.8804 |
| 03 | `03.resnet50_unfreeze_layer4_2` | `layer4[2]` only | 4.7M | Dropout + WD=1e-4 | 0.6067 | 0.5986 | 0.7953 | 0.3009 | 0.8678 |
| 04 | `04.resnet50_unfreeze_layer4_wd1e3` | `layer4` (full) | 15M | Dropout + WD=1e-3 | 0.6595 | 0.5859 | 0.7018 | 0.3529 | 0.8728 |
| 05 | `05.resnet50_layer4_l1_l2` | `layer4` (full) | 15M | Dropout + WD=1e-3 + L1=1e-3 + L2=1e-3 | 0.6459 | **0.6184** | 0.7485 | 0.3647 | 0.8841 |
| 06 | `06.resnet50_layer4_2_l1_l2` | `layer4[2]` only | 4.7M | Dropout + WD=1e-3 + L1=1e-3 + L2=1e-3 | 0.6003 | 0.5730 | 0.7485 | 0.2956 | 0.8460 |
| 07 | `07.resnet50_layer4_l1` | `layer4` (full) | 15M | WD=1e-3 + L1=1e-3 (no Dropout) | 0.6536 | 0.6074 | **0.8070** | 0.3053 | **0.8861** |
| 08 | `08.resnet50_layer4_l1_metadata` | `layer4` (full) | 15M | Dropout=0.5 + WD=1e-3 + L1=1e-3 + Metadata | 0.6575 | 0.6048 | 0.7018 | **0.3896** | 0.8848 |

### Iteration Intuition & Reasoning
- **Frozen baseline (01)**: With only 2K trainable parameters in the classification head, the model lacked sufficient capacity to learn a good decision boundary over the 2048-dim frozen feature space. While the features are transferable, the classifier needed more expressive power to separate hard negatives (nevus) from melanoma.
- **Overfitting & Weight Decay (02, 04)**: Weight Decay (WD) alone was unable to control the 15M parameters in `layer4`. Even at WD=1e-3, the model showed a massive generalization gap, indicating that standard L2-style decay isn't strong enough for this dataset size.
- **Narrow unfreeze (03, 06)**: Restricting unfreezing to a single sub-block reduced overfitting but at the cost of model capacity. The model "under-learned" the medical domain, performing worse than the full `layer4` equivalents.
- **L1+L2 Penalties (05)**: This was the breakthrough for ResNet-50. Explicit L1 and L2 penalties act as a much harsher constraint than standard WD, forcing the model to find a simpler, more general solution. This yielded the best Test F2 (**0.6184**).
- **L1 only, no Dropout (07)**: Without Dropout, the model retained more effective capacity during training, while L1 acted as a feature selector to enforce sparsity. This combination allowed the model to learn stronger discriminative features (highest AUC at **0.8861**) while being more aggressive in positive predictions (highest recall at 0.8070, but lower precision). The result confirms that for ResNet-50, a single well-chosen regularizer outperforms stacking multiple weaker ones.
- **Metadata (08)**: Metadata improved precision but decreased recall. The additional metadata parameters shifted the model toward more conservative predictions — fewer positive calls, but more of them correct. The regularization hyperparameters (tuned for image-only models) were not re-optimized for the larger fusion architecture, likely limiting the metadata branch's contribution.

### Cross-Architecture Context
ResNet-50 consistently underperformed ResNet-18 across comparable configurations (best F2: 0.6184 vs 0.6704, best AUC: 0.8861 vs 0.9133). The deeper bottleneck architecture introduces ~15M trainable parameters in `layer4` alone, but HAM10000's limited size (~10K images) cannot support this capacity without heavy regularization. Even with L1+L2+Dropout stacking, ResNet-50 could not close the gap. This demonstrates that for small medical datasets, **scaling depth is counterproductive** — the additional representational power is wasted on memorizing training noise rather than learning generalizable lesion features. The move to EfficientNet was motivated by this observation: the project needed architectures that scale *smarter*, not just deeper.

---

## 6. EfficientNet Experiments (`notebooks/efficientnet/`)

### Architecture & Strategy
EfficientNet variants (B0, B3, B4) were evaluated. The `EfficientNetB0WithMetadata` and `EfficientNetB4WithMetadata` models fuse image features with 32-dim metadata.

### Results Summary
| # | Notebook | Variant | Unfrozen | Regularisation | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|---|---|
| 01 | `01.efficientnet_b3` | B3 | Full | Basic Aug | 0.6685 | 0.6490 | 0.8304 | 0.3463 | 0.9029 |
| 02 | `02.efficientnet_b3_augment` | B3 | Full | Strong Aug + AdamW | 0.6719 | 0.6683 | 0.7778 | 0.4277 | 0.9070 |
| 03 | `03.efficientnet_b0` | B0 | Full | Basic Aug | 0.6716 | 0.6455 | 0.7836 | 0.3785 | 0.9077 |
| 04 | `04.efficientnet_b3_partial_unfreeze` | B3 | Last 3 Blocks | Basic Aug | 0.6913 | 0.6220 | 0.7661 | 0.3550 | 0.8743 |
| 05 | `05.efficientnet_b0_l1_metadata` | B0 | Last 6 Blocks | Dropout 0.5 + L1 + Metadata | 0.6872 | 0.6875 | 0.8363 | 0.4017 | **0.9199** |
| 06 | `06.efficientnet_b0_l1_metadata_tta` | B0 | Last 6 Blocks | Dropout 0.5 + L1 + Metadata + TTA | 0.6873 | **0.6952** | **0.8830** | 0.3756 | 0.9182 |
| 07 | `07.efficientnet_b0_label_smooth` | B0 | Last 6 Blocks | Label Smooth + WRS + Metadata | 0.6526 | 0.6138 | 0.7602 | 0.3467 | 0.8869 |
| 08 | `08.efficientnet_b0_focal_loss` | B0 | Last 6 Blocks | Focal Loss (γ=2.0) + Metadata | 0.6092 | 0.5734 | 0.7953 | 0.2709 | 0.8501 |
| 09 | `09.efficientnet_b4_metadata` | B4 | Last 6 Blocks | Dropout 0.5 + L1 + Metadata (380px) | 0.6751 | 0.6611 | 0.7895 | 0.4006 | 0.9014 |
| 10 | `10.efficientnet_b4_metadata_tta` | B4 | Last 6 Blocks | Dropout 0.5 + L1 + Metadata + TTA | 0.6731 | 0.6782 | 0.7544 | **0.4831** | 0.9058 |

### Iteration Intuition & Reasoning
- **Architecture Efficiency**: EfficientNet's advantage stems from its Compound Scaling strategy, which jointly optimizes network depth, width, and input resolution, along with Squeeze-and-Excitation (SE) blocks that enable channel-wise attention. This produces richer feature representations than standard ResNets, reflected in the >0.90 AUC baseline.
- **Metadata Fusion (05)**: Incorporating age and sex pushed the AUC to **0.9199**. The intuition is that metadata acts as a "prior" that helps the model decide on ambiguous image cases.
- **TTA (06)**: Applying 8-augment TTA at inference was the project's most effective step. It lowered the optimal threshold from 0.62 to 0.54, boosting recall to **0.8830** with minimal precision loss. TTA smooths the probability distribution, making the model more confident in its ranking.
- **Label Smoothing & Focal Loss (07, 08)**: These regressions taught a valuable lesson. Label smoothing "softened" the signal too much for a binary problem, diluting the already sparse melanoma supervision. Focal Loss (γ=2.0, α=0.75) was too aggressive for this dataset's class distribution — the high γ excessively downweighted easy examples, destabilizing gradient magnitudes and preventing stable convergence. The hyperparameters (γ and α) were not tuned for the specific imbalance ratio, leading to worse calibration than standard BCE with pos_weight.
- **EfficientNet-B4 (09, 10)**: Scaling up to B4 at 380px resolution underperformed B0. The intuition is **over-regularization**: the L1 penalty (1e-3) used for B0 (4M params) was too heavy for B4 (17.5M params), causing the model to underfit. Train AUC stayed below Val AUC throughout, a clear sign the model was too constrained to learn.

### Cross-Architecture Context
EfficientNet-B0 represents a significant leap over both ResNet variants. Its best model (Iter 06, AUC=0.9182, F2=0.6952) surpasses the best ResNet-18 (AUC=0.9133, F2=0.6704) by a clear margin, despite having fewer trainable parameters (~4M in the unfrozen blocks vs 11.2M). This confirms that **architecture design matters more than scale**: Compound Scaling and SE attention extract more from the same data budget than simply stacking residual blocks. Notably, the B0→B4 scaling within EfficientNet echoed the ResNet-18→ResNet-50 lesson — increasing model size without retuning regularization leads to regression, not improvement. The consistent pattern across both architecture families reinforces that HAM10000's size is the binding constraint, and gains must come from smarter training strategies (metadata, TTA) rather than larger models.

---

## 7. ViT-B/16 Experiments (`notebooks/vit/`)

### Architecture & Strategy
Pretrained ViT-B/16 adapted for binary melanoma classification. Two fine-tuning strategies were explored: a simple weighted BCE baseline and staged fine-tuning (head warm-up first, then full backbone), combined with focal loss vs weighted BCE.

### Results Summary
| # | Notebook | Key Change | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|
| 01 | `01.vit_b16_weighted` | Weighted BCE baseline | — | 0.6005 | **0.8421** | 0.2796 | 0.8837 |
| 02 | `02.vit_b16_focal_staged_finetune` | + Focal loss + staged fine-tuning | — | **0.6314** | 0.7953 | 0.3461 | 0.9042 |
| 03 | `03.vit_b16_weighted_staged_finetune` | Staged fine-tuning + weighted BCE | — | 0.6256 | 0.7251 | **0.4039** | **0.9066** |

### Iteration Intuition & Reasoning
- **Weighted BCE baseline (01)**: ViT's global self-attention provides a strong screening baseline out of the box (AUC 0.8837, recall 0.8421) — it sees the full lesion context in a single forward pass, unlike CNNs which rely on local receptive fields. However, the simple fine-tuning strategy leaves precision weak (0.2796).
- **Focal loss + staged fine-tuning (02)**: Warming up the classification head before unfreezing the backbone prevents the pretrained attention weights from being destroyed by a large gradient signal on the first epoch. Focal loss's down-weighting of easy negatives further sharpens discrimination, boosting AUC to 0.9042 and F2 to 0.6314.
- **Staged fine-tuning + weighted BCE (03)**: The staged schedule alone (without focal loss) achieves the best AUC (0.9066) and precision (0.4039), confirming that the fine-tuning schedule — not the loss function — is the primary driver of the AUC improvement. However, weighted BCE under this schedule becomes too conservative, dropping recall to 0.7251 and F2 to 0.6256.

### Cross-Architecture Context
ViT-B/16 (best AUC 0.9066, best F2 0.6314) sits between ResNet-18 (0.9133 / 0.6704) and the EfficientNet-B0 family. Despite ViT's theoretical advantage from global attention, it underperformed EfficientNet-B0 on this dataset — likely because the 10K-image HAM10000 is too small to fully leverage ViT's data-hungry attention mechanism. The staged fine-tuning insight (protect pretrained weights during early training) proved broadly applicable and influenced the partial-unfreeze strategy used in later EfficientNet and MobileNet experiments.

---

## 8. MobileNet Experiments (`notebooks/mobilenet/`)

### Architecture & Strategy
MobileNetV3-Large utilizes inverted residual blocks and Squeeze-and-Excitation.

### Results Summary
| # | Notebook | Config | Best Val F2 | Test F2 | Test Recall | Test Precision | AUC-ROC |
|---|---|---|---|---|---|---|---|
| 01 | `01.mobilenet_v3_metadata_tta` | Full backbone, L1+L2 (1e-3), Metadata, 8× TTA | 0.6802 | 0.6492 | 0.8421 | 0.3388 | 0.9114 |
| 02 | `02.mobilenet_v3_efficientnet_b0_ensemble` | Ensemble (MobileNetV3 + EfficientNet-B0), mean TTA probs | 0.7023 | 0.6781 | 0.8772 | 0.3555 | **0.9235** |

### Iteration Intuition & Reasoning
- **MobileNet Performance (01)**: MobileNetV3-Large achieved strong results but trailed EfficientNet-B0. While both architectures use depthwise separable convolutions, EfficientNet benefits from Compound Scaling (jointly optimizing depth, width, and resolution) and SE blocks, giving it richer feature representations. MobileNetV3's architecture prioritizes inference speed over representation capacity, which explains the performance gap on fine-grained medical classification.
- **Ensemble (02)**: The ensemble achieved the project's highest AUC (**0.9235**). The reasoning is "error decorrelation": since MobileNet and EfficientNet have different inductive biases, they tend to make different mistakes. Averaging their predictions cancels out these individual errors, raising the discriminative ceiling of the entire system.

### Cross-Architecture Context
MobileNetV3 as a standalone model (AUC=0.9114, F2=0.6492) slots between the ResNets and EfficientNet-B0, consistent with its design as a speed-optimized architecture. However, its real value emerged through ensembling: the MobileNet+EfficientNet ensemble achieved the project's highest AUC (0.9235), surpassing EfficientNet-B0 alone (0.9199). This demonstrates that **architectural diversity is itself a resource** — two mid-tier models with different inductive biases can outperform a single stronger model. The ensemble also validates the overall project trajectory: rather than continuing to scale individual models (which hit diminishing returns at B4/ResNet-50), combining complementary architectures proved to be the most effective path to improving discriminative performance.

---

## Overall Best Models by Category

| Folder | Best Notebook | AUC | Melanoma Recall | Test F2 |
|---|---|---|---|---|
| `cnn/` | `05.cnn_residual_batchnorm_weighted` | 0.8672 | 0.8187 | 0.5814 |
| `densenet/` | `03.densenet121_batchnorm_weighted` | 0.8580 | 0.8246 | 0.5964 |
| `resnet/` | `04.resnet_augmented_v2` | 0.9133 | 0.7778 | 0.6704 |
| `resnet50/` | `05.resnet50_layer4_l1_l2` | 0.8841 | 0.7485 | 0.6184 |
| `vit/` | `03.vit_b16_weighted_staged_finetune` | 0.9066 | 0.7251 | 0.6256 |
| **`efficientnet/`** | **`06.efficientnet_b0_l1_metadata_tta`** | 0.9182 | **0.8830** | **0.6952** |
| **`mobilenet/`** | **`02.mobilenet_v3_efficientnet_b0_ensemble`** | **0.9235** | 0.8772 | 0.6781 |

**Project Champion (AUC)**: `mobilenet/02.mobilenet_v3_efficientnet_b0_ensemble` (**0.9235**)  
**Project Champion (Recall/F2)**: `efficientnet/06.efficientnet_b0_l1_metadata_tta` (**0.8830** Recall, **0.6952** F2)

