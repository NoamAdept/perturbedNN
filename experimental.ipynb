import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications import inception_v3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from PIL import Image

# Load pre-trained InceptionV3 model
model = inception_v3.InceptionV3()

# Target ImageNet class index (e.g., 859 for "toaster")
target_class = 859

# Load and preprocess the image
img_path = "cat.png"  # Ensure the image is uploaded
img = image.load_img(img_path, target_size=(299, 299))
original_image = image.img_to_array(img)

# Normalize the image as required by InceptionV3
original_image = preprocess_input(original_image)

# Add batch dimension
original_image = np.expand_dims(original_image, axis=0)

# Convert to TensorFlow tensor
hacked_image = tf.Variable(np.copy(original_image), dtype=tf.float32)

# Attack parameters
epsilon = 0.01  # Perturbation magnitude for FGSM
alpha = 0.005   # Step size for PGD and MIM
steps = 100     # Increased iterations for PGD, MIM, and C&W

# Track loss and accuracy changes
loss_history = {"FGSM": [], "PGD": [], "MIM": []}
accuracy_history = {"FGSM": [], "PGD": [], "MIM": []}

def compute_loss(model, image, target_class):
    """Compute loss and predictions for the given image."""
    preds = model(image)
    loss = preds[0, target_class]  # Target class probability
    original_class_prob = np.max(preds.numpy()[0])  # Highest probability among predictions
    return loss, preds, original_class_prob

def fgsm_attack(image, epsilon):
    """Perform FGSM attack."""
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss, preds, original_class_prob = compute_loss(model, image, target_class)

    gradient = tape.gradient(loss, image)
    perturbed_image = image + epsilon * tf.sign(gradient)
    
    loss_history["FGSM"].append(loss.numpy())
    accuracy_history["FGSM"].append(original_class_prob)

    return tf.clip_by_value(perturbed_image, -1, 1), preds.numpy()

def pgd_attack(image, alpha, steps):
    """Perform PGD attack."""
    adv_image = tf.Variable(np.copy(image), dtype=tf.float32)

    for i in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            loss, preds, original_class_prob = compute_loss(model, adv_image, target_class)
        
        gradient = tape.gradient(loss, adv_image)
        adv_image.assign_add(alpha * tf.sign(gradient))
        adv_image.assign(tf.clip_by_value(adv_image, -1, 1))

        loss_history["PGD"].append(loss.numpy())
        accuracy_history["PGD"].append(original_class_prob)

    return adv_image, preds.numpy()

def mim_attack(image, alpha, steps, mu=0.9):
    """Perform MIM attack."""
    adv_image = tf.Variable(np.copy(image), dtype=tf.float32)
    velocity = tf.Variable(tf.zeros_like(adv_image), dtype=tf.float32)

    for i in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            loss, preds, original_class_prob = compute_loss(model, adv_image, target_class)
        
        gradient = tape.gradient(loss, adv_image)
        velocity.assign(mu * velocity + gradient / tf.reduce_sum(tf.abs(gradient)))
        adv_image.assign_add(alpha * tf.sign(velocity))
        adv_image.assign(tf.clip_by_value(adv_image, -1, 1))

        loss_history["MIM"].append(loss.numpy())
        accuracy_history["MIM"].append(original_class_prob)

    return adv_image, preds.numpy()

# Generate adversarial examples
fgsm_adv, fgsm_preds = fgsm_attack(hacked_image, epsilon)
pgd_adv, pgd_preds = pgd_attack(hacked_image, alpha, steps)
mim_adv, mim_preds = mim_attack(hacked_image, alpha, steps)

# Convert images back to displayable format
def process_image(img_tensor):
    img_numpy = img_tensor.numpy()[0]
    img_numpy = ((img_numpy + 1) / 2.0) * 255.0  # Convert back from [-1, 1] to [0, 255]
    return np.clip(img_numpy, 0, 255).astype(np.uint8)

fgsm_img = process_image(fgsm_adv)
pgd_img = process_image(pgd_adv)
mim_img = process_image(mim_adv)

# Decode predictions for probability display
original_preds = model(original_image)
original_top_pred = decode_predictions(original_preds.numpy())[0][0]

fgsm_top_pred = decode_predictions(fgsm_preds)[0][0]
pgd_top_pred = decode_predictions(pgd_preds)[0][0]
mim_top_pred = decode_predictions(mim_preds)[0][0]

# Display results with probability annotations
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
plt.imshow(image.load_img(img_path))
plt.title(f"Original\n{original_top_pred[1]}: {original_top_pred[2]*100:.2f}%")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(fgsm_img)
plt.title(f"FGSM Attack\n{fgsm_top_pred[1]}: {fgsm_top_pred[2]*100:.2f}%")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(pgd_img)
plt.title(f"PGD Attack\n{pgd_top_pred[1]}: {pgd_top_pred[2]*100:.2f}%")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(mim_img)
plt.title(f"MIM Attack\n{mim_top_pred[1]}: {mim_top_pred[2]*100:.2f}%")
plt.axis("off")

plt.show()

# Plot loss and accuracy history
plt.figure(figsize=(12, 5))
for method in loss_history:
    plt.plot(loss_history[method], label=f"{method} Loss")

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Progression During Attacks")
plt.show()

plt.figure(figsize=(12, 5))
for method in accuracy_history:
    plt.plot(accuracy_history[method], label=f"{method} Accuracy")

plt.xlabel("Iteration")
plt.ylabel("Max Predicted Probability")
plt.legend()
plt.title("Model's Confidence Progression")
plt.show()
