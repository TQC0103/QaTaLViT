# Prompt tạo hình pipeline đề xuất

Create a clean academic architecture diagram for a medical image segmentation paper, in the style of a computer vision conference figure. Use a white background, thin arrows, clean typography, blue blocks for vision modules, green blocks for language modules, orange blocks for cross-modal interaction modules, and gray blocks for losses. The figure should look like an improved variant of the LViT baseline architecture, but with a new proposed module. The layout should be horizontal, balanced, and easy to read on one paper page.

Title of figure: "Proposed BioBERT-Enhanced LViT with Bottleneck Cross-Attention"

Show the pipeline with these components from left to right:

1. Inputs
- chest X-ray image
- medical text prompt / clinical hint

2. Vision branch
- CNN encoder / downsampling path
- ViT branch / global context branch
- bottleneck feature map
- decoder / upsampling path
- segmentation output mask

3. Text branch
- BioBERT encoder
- token embeddings

4. Proposed interaction module
- place a highlighted orange module at the bottleneck level
- label it "Bottleneck Cross-Attention"
- image bottleneck features should act as query
- text tokens should act as key / value
- output should be "text-conditioned image features"
- then show a small "Residual Fusion" block combining original bottleneck image feature and cross-attended feature before the decoder

5. Comparison cue
- visually indicate that this replaces the original direct text injection style of LViT
- add a small side note near the new module: "explicit image-text interaction at high semantic level"

6. Losses
- supervised Dice + Cross Entropy loss

7. Output note
- add a small caption-like note inside the figure:
"BioBERT preserves domain-specific medical semantics; bottleneck cross-attention replaces rigid early text injection and improves image-text alignment."

Style requirements:
- professional, minimal, publication-quality
- no photorealism
- no decorative background
- no teacher model
- no pseudo-label branch
- no gated fusion block
- concise labels only
- arrows must be clear and directional
- spacing should be symmetric and neat
- overall design should feel like a stronger, cleaner successor to the original LViT diagram
