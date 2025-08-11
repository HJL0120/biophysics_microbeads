#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.ndimage import center_of_mass
from PIL import Image
import matplotlib.pyplot as plt

image_path = "C:\\Users\\PC\\Desktop\\final year project\\照片集\\微信图片_20241223203519.png"
# print(image_path)

# Load the image
# image_path = "~/Desktop/MovingBead-0_-260nm.png"  # Replace with your image path
image = Image.open(image_path).convert("L")  # Convert to grayscale
image_array = np.array(image)

# Calculate the centroid
centroid = center_of_mass(image_array)

# Display the image and mark the centroid
plt.imshow(image_array, cmap='gray')
plt.scatter(centroid[1], centroid[0], color='red', label=f"Centroid: {centroid}")
plt.legend()
plt.title("Centroid of the Image")
plt.show()

# Print the centroid coordinates
print(f"Centroid (row, column): {centroid}")


# In[3]:


import numpy as np
from scipy.ndimage import center_of_mass
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = "C:\\Users\\PC\\Desktop\\final year project\\照片集\\微信图片_20241223203519.png"
image = Image.open(image_path).convert("L")  # Convert to grayscale
image_array = np.array(image)

# Step 1: Threshold the image to remove the background
# Define a threshold value to distinguish the bead from the background
threshold_value = np.mean(image_array) + np.std(image_array)  # Adaptive threshold
binary_mask = image_array > threshold_value  # Create a binary mask (True for bead, False for background)

# Step 2: Apply the mask to isolate the bead
foreground_image = image_array * binary_mask  # Keeps only bead pixel values; background becomes 0

# Step 3: Calculate the centroid of the bead (foreground)
centroid = center_of_mass(foreground_image)  # Compute centroid only for non-zero pixel values

# Step 4: Visualize the bead and its centroid
plt.imshow(image_array, cmap='gray')
plt.imshow(binary_mask, cmap='gray', alpha=0.3)  # Overlay the mask
plt.scatter(centroid[1], centroid[0], color='red', label=f"Centroid: {centroid}")
plt.legend()
plt.title("Centroid of the Microbead")
plt.show()

# Print the centroid coordinates
print(f"Centroid (row, column): {centroid}")


# In[ ]:




