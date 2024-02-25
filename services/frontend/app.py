import gradio as gr
import requests
from PIL import Image
import io

def gradio_interface(input_pil):
    image_io = io.BytesIO()
    input_pil.save(image_io, format='PNG')
    image_io.seek(0)

    files = {"input_image": image_io}
    response = requests.post("http://localhost:5000/generate", files=files)

    if response.status_code == 200:
        gen_img = Image.open(io.BytesIO(response.content))
        return gen_img
    else:
        return None

input_image = gr.Image(label="Input Image", type='pil')
generated_image = gr.Image(label="Generated Image")

gradio_app = gr.Interface(fn=gradio_interface, inputs=input_image, outputs=generated_image)

if __name__ == "__main__":
    gradio_app.launch()