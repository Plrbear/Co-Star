# Co-Star: Collaborative Curriculum Self-Training with Adaptive Regularization for Source-Free Video Domain Adaptation

<figure>
  <img src="src/co-star.png" alt="Co-star framework overview" width="600">
  <figcaption>
    <strong>Overview of the Co-STAR framework.</strong> The collaborative self-training component leverages a teacher-student architecture, where both the CLIP and teacher models contribute to pseudo-label generation. The curriculum learning component dynamically adjusts the balance between CE and KL divergence losses based on reliability scores r. The ACR module refines this balance by monitoring prediction stability through a history buffer. The student model is then trained using this adaptive balance, effectively combining knowledge distillation with pseudo-label learning for robust adaptation. Note that CLIP's zero-shot predictions are pre-computed before training to reduce computational overhead.
  </figcaption>
</figure>

## Environment Setup

The `co-star.yaml` file can be used to create the conda environment with all dependencies for training Co-STAR:

```bash
conda env create -f co-star.yaml
conda activate co-star

# Training

Please refer to the sample bash script (`run.sh`) located in the `src` directory to start training.

## Configuration Setup

Before training, you need to configure the data paths:

1. **Update data paths**: Edit `configs/experiment/arid_mit.yaml` and change the paths to your local data directory.

2. **For new experiments**: You can create additional `.yaml` files in the `configs/experiment/` directory for your custom experiments. However, make sure to update the experiment name in `configs/train.yaml` accordingly, as it is currently set to `arid_mit`.

## Quick Start

```bash
# Navigate to src directory
cd src

# Run training script
./run.sh
