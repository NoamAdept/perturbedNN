# perturbedNN

<img width="439" alt="image" src="https://github.com/user-attachments/assets/5c0065de-04bb-4e87-a7a7-ea71bac71e5f" />

# Adversarial Attacks on Image Classification Models

## Overview
This repository explores adversarial attacks on deep learning models, particularly convolutional neural networks (CNNs), with a focus on image classification. The study demonstrates the vulnerability of InceptionV3 to adversarial perturbations and evaluates the effectiveness of various attack methods, including Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and the Momentum Iterative Method (MIM).

## Motivation
Deep learning models have achieved remarkable success in image classification, but they remain highly susceptible to adversarial attacks. This project aims to analyze the mathematical foundations of adversarial perturbations, their impact on classification accuracy, and potential defense strategies.

## Attacks Implemented

### Fast Gradient Sign Method (FGSM)
FGSM is a single-step attack that perturbs the image along the gradient of the loss function:
```math
x^* = x + \epsilon \cdot \text{sign}(\nabla_x J(f(x), y))
```
FGSM is computationally efficient but often less effective against adversarially trained models.

### Projected Gradient Descent (PGD)
PGD iteratively refines the adversarial perturbation while constraining it within an \( \epsilon \)-ball around the original image:
```math
x_{t+1} = \text{Proj}_{x+\mathcal{S}}(x_t + \alpha \cdot \text{sign}(\nabla_x J(f(x_t), y)))
```
PGD is one of the most effective adversarial attacks, leveraging multiple refinement steps.

### Momentum Iterative Method (MIM)
MIM stabilizes gradient updates by incorporating momentum, improving transferability across different models:
```math
g_{t+1} = \mu g_t + \frac{\nabla_x J(f(x_t), y)}{||\nabla_x J(f(x_t), y)||_1}
x_{t+1} = x_t + \alpha \cdot \text{sign}(g_{t+1})
```
MIM prevents oscillatory updates and enhances attack effectiveness.

## Experimental Results
Experiments were conducted using a pre-trained InceptionV3 model. The original image, correctly classified as a tabby cat, was subjected to adversarial perturbations:

- **FGSM**: The attack was ineffective, and the model retained its original classification.
- **PGD and MIM**: Successfully misclassified the image as a toaster with 100% confidence.
- **Loss Progression**: PGD and MIM converge rapidly, whereas FGSM shows limited effectiveness.

<img width="889" alt="Screenshot 2025-02-25 at 14 04 27" src="https://github.com/user-attachments/assets/fed6f61e-7315-431a-93ad-6411602bca4b" />



## Implications
Adversarial attacks pose significant security threats in real-world applications, such as:
- Misleading autonomous vehicles
- Bypassing facial recognition systems
- Manipulating financial fraud detection models

Addressing these vulnerabilities requires robust defenses, including adversarial training and certified model robustness techniques.

## Future Work
- Implementing additional adversarial attacks such as Carlini & Wagner (C&W) and DeepFool.
- Exploring defense mechanisms, including adversarial training and detection-based countermeasures.
- Investigating attack transferability across different model architectures.

## References
- Goodfellow et al., "Explaining and Harnessing Adversarial Examples," 2014.
- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," 2017.
- Kurakin et al., "Adversarial Machine Learning at Scale," 2016.

## Citation
If you use this code, please cite the associated paper:
```
@article{adversarial_attacks,
  author    = {Noam Yakar},
  title     = {A Review of Adversarial Attacks on Image Classification Models},
  year      = {2025}
}


