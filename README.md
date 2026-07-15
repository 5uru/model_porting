# Model Porting: PyTorch → JAX/Flax

Accompanying code for the blog post: [Portage de modèle : Migrer de PyTorch à JAX/Flax](https://jonathansuru.me/articles/model_porting.html)

Demonstrates how to port a trained PyTorch CNN (SmallCNN on Fashion-MNIST) to JAX/Flax by transferring weights across frameworks — including proper handling of NCHW → NHWC format differences, tensor permutations, and the critical flatten reordering.

## Repository Structure

- `pytorch_model.py` — PyTorch model definition, training, evaluation, and visualization
- `notebook.ipynb` — End-to-end notebook: trains the model, ports it via `Torch2Flax`, and verifies identical accuracy (~91.69%)

## The Challenge

PyTorch uses NCHW (batch, channels, height, width) while JAX/Flax uses NHWC (batch, height, width, channels). This affects:
- **Conv weights**: `(out_c, in_c, H, W)` → `(H, W, in_c, out_c)`
- **Linear weights**: transpose `(out, in)` → `(in, out)`
- **Flatten ordering**: channel-major vs spatial-major ordering requires permuting the first FC layer's rows

