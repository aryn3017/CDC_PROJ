import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, image, tabular):
        self.model.zero_grad()

        output = self.model(image, tabular)
        output.backward(torch.ones_like(output))

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)

        cam = F.relu(cam)
        cam = cam[0].cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = cv2.GaussianBlur(cam, (11, 11), 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam
def overlay_cam(image, cam, alpha=0.45):
     """
     image: numpy array (H, W, 3), normalized [0,1]
     cam: numpy array (H, W), normalized [0,1]
     """
     heatmap = cv2.applyColorMap(
         np.uint8(255 * cam),
         cv2.COLORMAP_TURBO
     )
     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
     heatmap = heatmap / 255.0

     overlay = alpha * heatmap + (1 - alpha) * image
     overlay = overlay / overlay.max()
     return overlay
