# ============================================================
# model.py
# Swin Transformer Classifier
# Automated Lumbar Spine Disease Detection
# ============================================================

import timm
import torch
import torch.nn as nn


# ============================================================
# Swin Transformer Classifier
# ============================================================

class SwinClassifier(nn.Module):

    def __init__(
        self,
        model_name="swin_tiny_patch4_window7_224",
        num_classes=3,
        pretrained=True,
        dropout=0.30,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        self.num_features = self.backbone.num_features

        # ----------------------------------------------------
        # Classification Head
        # ----------------------------------------------------

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            self.num_features,
            num_classes
        )

        # ----------------------------------------------------
        # Feature storage for XAI
        # ----------------------------------------------------

        self.features = None
        self.gradients = None

    # ========================================================
    # Gradient Hook
    # ========================================================

    def activations_hook(self, grad):

        self.gradients = grad

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # Feature Extraction
        features = self.backbone.forward_features(x)

        # Save features for XAI
        self.features = features

        if features.requires_grad:
            features.register_hook(self.activations_hook)

        # ----------------------------------------------------
        # Global Pooling
        # ----------------------------------------------------

        if features.ndim == 4:
            features = features.mean(dim=(1, 2))

        elif features.ndim == 3:
            features = features.mean(dim=1)

        # ----------------------------------------------------
        # Classification Head
        # ----------------------------------------------------

        features = self.dropout(features)

        logits = self.classifier(features)

        return logits

    # ========================================================
    # XAI Utilities
    # ========================================================

    def get_activations_gradient(self):

        return self.gradients

    def get_activations(self):

        return self.features


# ============================================================
# Utility Function
# ============================================================

def create_model(
    model_name="swin_tiny_patch4_window7_224",
    num_classes=3,
    pretrained=True,
    dropout=0.30,
):

    model = SwinClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

    return model