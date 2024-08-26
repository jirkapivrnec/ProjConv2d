import torch
import torch.nn.functional as F


def visualize_feature_maps(model, input_image):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output, feature_maps = model(input_image.unsqueeze(0))  # Add batch dimension
        print(output)
        # Get the predicted class
        _, predicted = torch.max(output.data, 1)

        # Apply softmax to get probabilities
        probabilities = F.softmax(output, dim=1)

        # Get the probability of the predicted class
        predicted_probability = probabilities[0][predicted.item()]

        print("Predicted class:", predicted.item())
        print("Probability of the predicted class:", predicted_probability.item())
    return feature_maps


def calculate_dot_product(feature_maps):
    dot_products = []
    for fmap in feature_maps:
        fmap_flat = fmap.view(fmap.size(0), fmap.size(1), -1)  # Flatten each feature map
        dot_product = torch.bmm(fmap_flat, fmap_flat.transpose(1, 2))  # Dot product between feature maps
        dot_products.append(dot_product)
    return dot_products