import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image, ExifTags

# Define device
device = torch.device("cpu")

# Load the trained model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def fix_orientation_and_resize(image, max_size=800):
    """Fix EXIF orientation and resize large images"""
    # Fix EXIF orientation
    try:
        if hasattr(image, '_getexif') and image._getexif():
            exif = image._getexif()
            for tag, name in ExifTags.TAGS.items():
                if name == 'Orientation' and tag in exif:
                    orientation = exif[tag]
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
                    break
    except:
        pass
    
    # Resize if too large
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

def predict_image(image):
    """Predict if image is live or spoof"""
    # Convert to RGB and fix orientation/size
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = fix_orientation_and_resize(image)
    
    # Apply model transforms and predict
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    prediction = "live" if predicted.item() == 0 else "spoof"
    confidence_score = confidence.item()
    
    return prediction, confidence_score

# Streamlit app
st.title("CASIA-FASD Image Classifier")
st.write("Upload an image to check if it is 'live' or 'spoof'.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Show original size info
    st.write(f"Original size: {image.size[0]}x{image.size[1]} pixels")
    
    # Process and display
    processed_image = fix_orientation_and_resize(image.copy())
    st.image(processed_image, caption=f"Processed Image ({processed_image.size[0]}x{processed_image.size[1]})", use_container_width=True)
    
    # Make prediction
    prediction, confidence = predict_image(image)
    
    # Display result
    if prediction == "live":
        st.success(f"🟢 **LIVE** - Confidence: {confidence:.1%}")
    else:
        st.error(f"🔴 **SPOOF** - Confidence: {confidence:.1%}")
    
    if confidence < 0.7:
        st.warning("⚠️ Low confidence - result may be unreliable")




# import streamlit as st
# import torch
# from torchvision import transforms, models
# from PIL import Image, ExifTags

# # Define device
# device = torch.device("cpu")

# # Load the trained model
# @st.cache_resource
# def load_model():
#     model = models.resnet18(pretrained=False)
#     model.fc = torch.nn.Linear(model.fc.in_features, 2)
#     state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
#     new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
#     model.load_state_dict(new_state_dict)
#     model = model.to(device)
#     model.eval()
#     return model

# model = load_model()

# # Define transforms (model trained on 112x112)
# transform = transforms.Compose([
#     transforms.Resize((112, 112)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def fix_orientation_and_resize(image, max_size=800):
#     """Fix EXIF orientation and resize large images"""
#     # Fix EXIF orientation
#     try:
#         if hasattr(image, '_getexif') and image._getexif():
#             exif = image._getexif()
#             for tag, name in ExifTags.TAGS.items():
#                 if name == 'Orientation' and tag in exif:
#                     orientation = exif[tag]
#                     if orientation == 3:
#                         image = image.rotate(180, expand=True)
#                     elif orientation == 6:
#                         image = image.rotate(270, expand=True)
#                     elif orientation == 8:
#                         image = image.rotate(90, expand=True)
#                     break
#     except:
#         pass
    
#     # Resize if too large
#     if max(image.size) > max_size:
#         ratio = max_size / max(image.size)
#         new_size = tuple(int(dim * ratio) for dim in image.size)
#         image = image.resize(new_size, Image.Resampling.LANCZOS)
    
#     return image

# def predict_image(image):
#     """Predict if image is live or spoof"""
#     # Convert to RGB and fix orientation/size
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
    
#     image = fix_orientation_and_resize(image)
    
#     # Apply model transforms and predict
#     image_tensor = transform(image).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         output = model(image_tensor)
#         probabilities = torch.nn.functional.softmax(output, dim=1)
#         confidence, predicted = torch.max(probabilities, 1)
    
#     prediction = "live" if predicted.item() == 0 else "spoof"
#     confidence_score = confidence.item()
    
#     return prediction, confidence_score

# # Streamlit app
# st.title("CASIA-FASD Image Classifier")
# st.write("Upload an image to check if it is 'live' or 'spoof'.")

# uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
    
#     # Show original size info
#     st.write(f"Original size: {image.size[0]}x{image.size[1]} pixels")
    
#     # Process and display
#     processed_image = fix_orientation_and_resize(image.copy())
#     st.image(processed_image, caption=f"Processed Image ({processed_image.size[0]}x{processed_image.size[1]})", use_container_width=True)
    
#     # Make prediction
#     prediction, confidence = predict_image(image)
    
#     # Display result
#     if prediction == "live":
#         st.success(f"🟢 **LIVE** - Confidence: {confidence:.1%}")
#     else:
#         st.error(f"🔴 **SPOOF** - Confidence: {confidence:.1%}")
    
#     if confidence < 0.7:
#         st.warning("⚠️ Low confidence - result may be unreliable")
