import gradio as gr
from groq import Groq
from gtts import gTTS
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def on_image_change(image):
    if image:
        # set mode to Calorie Breakdown and disable the radio control
        return gr.update(value="Calorie Breakdown", interactive=False, choices=None), gr.update(visible=True)
    else:
        # re-enable radio; 
        return gr.update(value=None, interactive=True, choices=None), gr.update(visible=False)

def run_ocr_and_calories(image, recipe_input=""):
    ocr_prompt = f"""
    Extract ALL visible ingredients and estimated quantities from this food image.
    {recipe_input}
    Then estimate calorie breakdown based on what you see:
        - Ingredient name
        - Estimated quantity
        - Calories
        - Macros (Protein/Fat/Carbs)

    End with: TOTAL CALORIES: X kcal

    Do not invent ingredients not present in the image.
    """

    import base64, io
    image = image.convert("RGB")
    image.thumbnail((512, 512))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG",quality=80)
    image_bytes = buffer.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode()

    vision_response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role":"system","content":"You are an expert food image OCR and calorie estimator."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ocr_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
    )
    extracted_and_cal = vision_response.choices[0].message.content
    return extracted_and_cal

def build_prompt_for_mode(mode, source_text):
    """Return the appropriate prompt for the chosen mode using source_text."""
    if mode == "Calorie Breakdown":
        return f"""
        Provide a calorie breakdown of this recipe/text below. Format strictly as:
        - Ingredient name
        - Calories (kcal)
        - Quantity
        - Macros (Protein/Fat/Carbs)

        END with: TOTAL CALORIES: X kcal

        Make it easy to paste into MyFitnessPal / Chronometer / LoseIt/ MyNetDiary.

        Text:
        {source_text}
        """
    else:
        return f"Task: {mode}\n\nRecipe:\n{source_text}\n\nProvide helpful, concise output."

def run_text_mode(recipe_input, mode, last_generated):
    if mode == "Calorie Breakdown":
        source = last_generated if last_generated else recipe_input
    else:
        source = recipe_input

    prompt = build_prompt_for_mode(mode, source)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}]
    )
    recipe_text = response.choices[0].message.content

    new_last_generated = last_generated
    if mode in ["Improve", "Healthy Substitute", "Simplify", "Generate Shopping List"]:
        new_last_generated = recipe_text

    # audio generation only if Convert to Audio is chosen
    audio_path = None
    if mode == "Convert to Audio":
        tts = gTTS(recipe_text)
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(audio_file.name)
        audio_path = audio_file.name

    return recipe_text, audio_path, new_last_generated

# Gradio UI
with gr.Blocks(title="Smart Recipe Assistant") as demo:
    gr.Markdown("""
                # 🧑‍🍳 Your Own Chef 
                Enhances recipes, simplifies steps, substitutes ingredients, extracts recipe text from images,  
                and even **generates calories or reads recipes aloud — powered by **Groq LLMs**.
            """)

    with gr.Row():
        recipe_input = gr.Textbox(label="Ask Recipe", lines=10, placeholder="Ask recipe here...")
        image_input = gr.Image(label="Upload Food Image", type="pil",sources=["upload", "webcam"],format="jpeg",
        height=512,
        width=512
        )
       

    mode = gr.Radio(
        ["Improve", "Healthy Substitute", "Simplify", "Generate Shopping List", "Calorie Breakdown", "Convert to Audio"],
        value=None,
        label="Choose Action",
    )

    last_generated = gr.State("")  

    submit_btn = gr.Button("Generate Result")

    output_text = gr.Textbox(label="Output", lines=12)
    output_audio = gr.Audio(label="Listen to Recipe", type="filepath")

    image_input.change(
        fn=on_image_change,
        inputs=[image_input],
        outputs=[mode, output_text],
        queue=False
    )

    def main_click(recipe_input, mode_value, image, last_generated_value):
        # IMAGE flow
        if image is not None:
            extracted_text_and_cal = run_ocr_and_calories(image, recipe_input)
            return extracted_text_and_cal, None, last_generated_value

        # TEXT flow
        result_text, audio_path, new_last = run_text_mode(recipe_input, mode_value, last_generated_value)
        return result_text, audio_path, new_last

    submit_btn.click(
        fn=main_click,
        inputs=[recipe_input, mode, image_input, last_generated],
        outputs=[output_text, output_audio, last_generated]
    )

if __name__ == "__main__":
    demo.launch(share=True)

