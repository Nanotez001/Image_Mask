import streamlit as st
from PIL import Image
from PIL import ImageDraw
import pandas as pd
import requests
from io import BytesIO
import io
import zipfile
import re

class ImageAnalyzer:
    def __init__(self, image_input, tolerance=30):
        # Check if input is a path or a file-like object
        # File path
        if isinstance(image_input, str):  
            self.image = Image.open(image_input).convert("RGBA")
        # Pillow Image object
        elif isinstance(image_input, Image.Image):  
            self.image = image_input.convert("RGBA")
        # File-like object (e.g., BytesIO)
        else:  
            self.image = Image.open(image_input).convert("RGBA")
        
        self.pixels = self.image.load()
        self.width, self.height = self.image.size
        self.tolerance = tolerance

    def change_opacity(self, opacity_level):
        new_data = []
        for r, g, b, a in self.image.getdata():
            new_a = int(a * opacity_level)
            new_data.append((r, g, b, new_a))

        new_image = self.image.copy()
        new_image.putdata(new_data)
        return ImageAnalyzer(new_image)


    def convert(self, mode):
        # Convert the image to the specified mode
        converted_image = self.image.convert(mode)
        return ImageAnalyzer(converted_image)

    def is_almost_white(self, pixel):
        return all(255 - value <= self.tolerance for value in pixel[:3])

    def is_almost_black(self, pixel):
        return all(value <= self.tolerance for value in pixel[:3])

    def find_leftmost_nonwhite(self):
        for x in range(self.width):
            for y in range(self.height):
                if not self.is_almost_white(self.pixels[x, y]):  # Check using the tolerance
                    return x
        return -1

    def find_uppermost_nonwhite(self):
        for y in range(self.height):
            for x in range(self.width):
                if not self.is_almost_white(self.pixels[x, y]):  # Check using the tolerance
                    return y
        return -1

    def find_rightmost_nonwhite(self):
        for x in range(self.width - 1, -1, -1):
            for y in range(self.height):
                if not self.is_almost_white(self.pixels[x, y]):  # Check using the tolerance
                    return x
        return -1

    def find_downmost_nonwhite(self):
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if not self.is_almost_white(self.pixels[x, y]):  # Check using the tolerance
                    return y
        return -1
 
    def find_rim(self):
        leftmost_x = self.find_leftmost_nonwhite()
        uppermost_y = self.find_uppermost_nonwhite()
        rightmost_x = self.find_rightmost_nonwhite()
        downmost_y = self.find_downmost_nonwhite()
        return leftmost_x, uppermost_y, rightmost_x, downmost_y


    def paste_image(self, overlay_image, coordinates=(0, 0)):
        # Convert overlay_image to a proper RGBA Image if needed
        if isinstance(overlay_image, str):  # File path
            overlay = Image.open(overlay_image).convert("RGBA")
        elif isinstance(overlay_image, ImageAnalyzer):
            overlay = overlay_image.image.convert("RGBA")
        elif isinstance(overlay_image, Image.Image):
            overlay = overlay_image.convert("RGBA")
        else:
            raise TypeError("overlay_image must be a path, Image, or ImageAnalyzer")

        # Ensure coordinates are valid
        x, y = map(int, coordinates)
        if x + overlay.width > self.width or y + overlay.height > self.height:
            raise ValueError("Overlay image goes beyond the base image dimensions.")

        # Paste with alpha mask
        self.image.paste(overlay, (x, y), overlay)  # third arg is used as transparency mask
        return self  # Return self so you can chain
    
    def save(self, fp, format="JPEG"):
        self.image.save(fp, format=format)

    def crop(self, left, upper, right, lower):
        if left < 0 or upper < 0 or right > self.width or lower > self.height:
            raise ValueError("Crop coordinates are out of image bounds.")
        if left >= right or upper >= lower:
            raise ValueError("Invalid crop dimensions. Ensure left < right and upper < lower.")
        cropped_image = self.image.crop((left, upper, right, lower))
        return ImageAnalyzer(cropped_image)

    def resize_with_aspect_ratio(self, new_width=None, new_height=None):
        if new_width is None and new_height is None:
            raise ValueError("At least one of new_width or new_height must be specified.")

        # Ensure the image is in RGBA mode for PNG transparency
        if self.image.mode != 'RGBA':
            self.image = self.image.convert("RGBA")
        
        # Calculate the aspect ratio of the image
        aspect_ratio = self.width / self.height

        if new_width is not None:
            # Calculate new height while maintaining aspect ratio
            new_height = int(new_width / aspect_ratio)
        elif new_height is not None:
            # Calculate new width while maintaining aspect ratio
            new_width = int(new_height * aspect_ratio)

        # Resize the image
        resized_image = self.image.resize((new_width, new_height))
        
        # Ensure resized image is still RGBA
        resized_image = resized_image.convert("RGBA")
        return ImageAnalyzer(resized_image)
    
    def draw_layout_area(self, top_left, size, color=(255, 0, 0, 100)):
        """
        Draw a semi-transparent color rectangle (mask) to visualize layout area.
        :param top_left: (x, y) tuple where the box starts
        :param size: (width, height) of the layout area
        :param color: RGBA color for the mask (default: red with 100 alpha)
        """
        x, y = top_left
        w, h = size

        # Create a transparent overlay
        overlay = Image.new("RGBA", self.image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([x, y, x + w, y + h], fill=color)

        # Paste the overlay onto the image using alpha blending
        self.image = Image.alpha_composite(self.image, overlay)
        return self  # for chaining
    
    def resize_fill(self, target_size:tuple):
        original_ratio = self.image.width/self.image.height
        target_ratio = target_size[0]/target_size[1]
        
        # Resize the image
        # Depend on Height
        if target_ratio - original_ratio >= 1:
            resized_image = self.image.resize((int(target_size[0]*original_ratio),int(target_size[1])))
        # Depend on Width
        else:
            resized_image = self.image.resize((int(target_size[0]),int(target_size[1]*original_ratio)))
        
        # Ensure resized image is still RGBA
        resized_image = resized_image.convert("RGBA")
        return ImageAnalyzer(resized_image)
    
        st.write("AAA",img_copy.size)
        img_copy.thumbnail(target_size)  # in-place resize on copy
        new_analyzer = ImageAnalyzer(img_copy)
        return new_analyzer
    
# ==================================================================
# Code outer class
n_check=0
def Check():
     global n_check
     n_check = n_check+1
     st.write("Check_Point",n_check)


def importfromGit(image_url):

    # Send a GET request to fetch the raw image
    response = requests.get(image_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the image from the response content
        img = Image.open(BytesIO(response.content))
        return img
    else:
        return print("Failed to retrieve the image. Status code:", response.status_code)

def Alignment(xy_alignment,image_size):
    x_align = xy_alignment[0]
    y_align = xy_alignment[1]
    image_width = image_size[0]
    image_height = image_size[1]
    size_align = xy_alignment[2]

    if size_align == "small":
        width_coordinate = image_width*0.1
        height_coordinate = image_height*0.1
    elif size_align == "medium":
        width_coordinate = image_width*0.2
        height_coordinate = image_height*0.2
    elif size_align == "large":
        width_coordinate = image_width*0.3
        height_coordinate = image_height*0.3
    else:
        return "INPUT WRONG"

    overlay_width = width_coordinate
    overlay_height = height_coordinate
    
    
    diff_width = image_width - overlay_width
    diff_height = image_height - overlay_height

    if x_align == "left":
        x_coordinate = 0
    elif x_align == "center":
        x_coordinate = diff_width//2
    elif x_align == "right":
        x_coordinate = diff_width-1
    else:
        return "INPUT WRONG"
    
    if y_align == "up":
        y_coordinate = 0
    elif y_align == "center":
        y_coordinate = diff_height//2
    elif y_align == "down":
        y_coordinate = diff_height-85
    else:
        return "INPUT WRONG"
    return [x_coordinate,y_coordinate,width_coordinate,height_coordinate]

# ====================================
def main():
    st.title("Batch IMAGE Mask v0.10")

    # Sidebar components
    st.sidebar.title("upload")
    watermask_file = st.sidebar.file_uploader("Upload Watermask PNG File (PNG Only)", type=["png"], accept_multiple_files=False)

    opacity = st.sidebar.slider("LOGO Opacity",value=100,min_value=1,max_value=100)
    opacity_scale = opacity/100

    size_buffer_slide = st.sidebar.slider("Size",value=50,min_value=1,max_value=100)
    size_buffer = size_buffer_slide/100

    width_buffer=st.sidebar.slider("Width_buffer",value=50,min_value=1,max_value=100)
    width_buffer_scale = width_buffer/100

    height_buffer=st.sidebar.slider("Height_buffer",value=50,min_value=1,max_value=100)
    height_buffer_scale = height_buffer/100
# =======================================================================
    # Main Columns
    uploaded_files = st.file_uploader("Upload JPG Files (คำแนะนำ แก้ไขชื่อไฟล์ให้เรียบร้อยก่อน)", type=["jpg", "jpeg","png"], accept_multiple_files=True)

    if uploaded_files and watermask_file:
        
        result_images=[]
        original_name=[]
        # Process each uploaded file
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                
                # Fix and Save Name
                name_without_ext = uploaded_file.name
                name_cleaned = re.sub(r'\(\d+\)', '', name_without_ext)
                name_cleaned = name_cleaned.strip()
                original_name.append(name_cleaned)
                

                # =======================================
                original_image = ImageAnalyzer(uploaded_file)
                watermask_png = ImageAnalyzer(watermask_file)

                scaled = tuple(x * size_buffer for x in original_image.image.size)
                watermask_png = watermask_png.resize_fill(scaled)
                watermask_png = watermask_png.change_opacity(opacity_scale)
                # ====================================================================
                # Mix
                # width_buffer_scale = 1
                # height_buffer_scale = 1

                # original_image.width-watermask_png.width

                original_center = ((original_image.width-watermask_png.width)*width_buffer_scale,\
                                   (original_image.height-watermask_png.height)*height_buffer_scale)
                result = original_image.paste_image(watermask_png,original_center)
                # st.write(original_center)
                # =======================================
                # Append the result to the results list
                result_images.append(result)

            except Exception as e:
                st.error(f"Error processing the image '{uploaded_file.name}': {e}")


            # ========================================================
            # Display the before and after images
            col1, col2,col3 = st.columns(3)
            with col1:
                st.image(original_image.image, caption="Before", use_container_width=True)

            with col2:
                # result_image_path = "C:/Users/LEGION by Lenovo/Desktop/Image_Editor/Result_Test.jpg"
                st.image(result.image, caption="After", use_container_width=True)
            with col3:
                # Save result as JPEG to in-memory buffer
                img_bytes = io.BytesIO()

                result.image.convert("RGB").save(img_bytes, format="JPEG")

                platform_fullname = "Teammonij"
                img_bytes.seek(0)
                # Download button for the individual image
                st.download_button(
                    label=f"Download",
                    data=img_bytes,
                    file_name=f"{name_cleaned}_{platform_fullname}_{i+1}.jpg",
                    mime="image/jpeg"
    )

        # Download button to download the image
        if result_images:
            # Create a BytesIO object for the ZIP file
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for i, result in enumerate(result_images):
                    try:
                        img_bytes = io.BytesIO()
                        result.image.convert("RGB").save(img_bytes, format="JPEG")
                        img_bytes.seek(0)
                    except Exception as e:
                        st.error(f"Error saving image {original_name[i]}: {e}")
                        continue  # Skip corrupted image
                    zip_file.writestr(f"{name_cleaned}_{platform_fullname}_{i+1}.jpg", img_bytes.read())

            zip_buffer.seek(0)
                
            # Add a single download button for all images as a ZIP file
            st.download_button(
                label="Download ALL Images as ZIP",
                data=zip_buffer,
                file_name="Result_Images.zip",
                mime="application/zip",
                )
    else:
        st.info("Please upload at least one image to proceed.")

# Run the app
if __name__ == "__main__":
    main()
