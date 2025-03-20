import cv2

# Load the image
image_path = "data/train/apple_9.jpg"  # Change this to your actual image path
image = cv2.imread(image_path)

# Bounding box coordinates from your data
xmin, ymin, xmax, ymax = 184, 110, 582, 525
class_name = "apple"

# Draw the bounding box
color = (0, 255, 0)  # Green color
thickness = 2
cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

# Add label
label = class_name
cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display image
cv2.imshow("Image with Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image
cv2.imwrite("output.jpg", image)
