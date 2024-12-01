import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# PUT YOUR MODEL HERE
model = ...  # Replace with your trained model
model.save('saved_models/face_shape_identifier.keras')

# SHAPE IDENTIFICATION
mapping = {"0": "heart", "1": "oblong", "2": "oval", "3": "round", "4": "square"}

# SAMPLE POINTS FOR HEAD AND HAIR (2 x 46 matrices)
# Head Points: 46 x-coordinates (row 0) and 46 y-coordinates (row 1)
head_points = np.array([
    np.random.uniform(1.0, 20.0, 46),  # Random x-coordinates in range [1.0, 20.0]
    np.random.uniform(1.0, 20.0, 46)   # Random y-coordinates in range [1.0, 20.0]
])

# Hair Points: 46 x-coordinates (row 0) and 46 y-coordinates (row 1)
hair_points = np.array([
    np.random.uniform(10.0, 30.0, 46),  # Random x-coordinates in range [10.0, 30.0]
    np.random.uniform(10.0, 30.0, 46)   # Random y-coordinates in range [10.0, 30.0]
])

# CALCULATE CENTROIDS (Mean x and y for head and hair points)
centroid_head = np.mean(head_points, axis=1)  # [mean_x, mean_y] for head
centroid_hair = np.mean(hair_points, axis=1)  # [mean_x, mean_y] for hair

# CENTER POINTS BY SUBTRACTING CENTROIDS
head_centered = head_points - centroid_head[:, np.newaxis]
hair_centered = hair_points - centroid_hair[:, np.newaxis]

# CALCULATE ROTATION MATRIX USING SVD
H = np.dot(hair_centered, head_centered.T)  # Cross-covariance matrix
U, _, Vt = np.linalg.svd(H)  # Singular Value Decomposition
rotation_matrix = np.dot(U, Vt)  # Calculate the rotation matrix

# APPLY ROTATION AND TRANSLATION TO HAIR POINTS
transformed_hair = np.dot(rotation_matrix, hair_centered) + centroid_head[:, np.newaxis]

# PRINT RESULTS
print("Original Hair Points:\n", hair_points)
print("Original Head Points:\n", head_points)
print("Transformed Hair Points:\n", transformed_hair)

# PLOT POINTS IN 2D
fig, ax = plt.subplots(figsize=(10, 7))

# Plot Original Hair Points (blue)
ax.scatter(hair_points[0, :], hair_points[1, :], c='blue', label='Original Hair Points')

# Plot Head Points (red)
ax.scatter(head_points[0, :], head_points[1, :], c='red', label='Head Points')

# Plot Transformed Hair Points (green)
ax.scatter(transformed_hair[0, :], transformed_hair[1, :], c='green', label='Transformed Hair Points')

# Label Axes and Show Legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

# Display the Plot
plt.show()

# FUNCTION TO USE MODEL FOR FACE SHAPE PREDICTION
def preprocess_for_model(hair_image):
    # USE TRAINED MODEL TO PREDICT FACE SHAPE BASED ON IMAGE
    prediction = model.predict(hair_image)
    face_shape = mapping[str(np.argmax(prediction))]  # Map prediction to face shape
    return face_shape

# EXAMPLE USAGE:
# Assuming hair_image_data is your input
# hair_image = preprocess(hair_image_data)
# face_shape = preprocess_for_model(hair_image)
# print(f"Predicted Face Shape: {face_shape}")
