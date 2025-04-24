from PIL import Image
import base64
import io

# === Configuration ===
SIZE = 128  # Target square size in pixels
PATH = "55.jpg"  # Path to the image file

# === Resize image to square ===
def resize_to_square(image_path, size):
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure no alpha channel
        img = img.resize((size, size), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

# === Convert to base64 ===
def encode_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

# === Main ===
if __name__ == "__main__":
    image_bytes = resize_to_square(PATH, SIZE)
    base64_str = encode_to_base64(image_bytes)

    print("Base64 encoded image:")
    print(base64_str)