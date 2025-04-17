import os
import sys
import subprocess
import tempfile
import time
from typing import Optional

try:
    import google.generativeai as genai
except ImportError:
    print("Google Generative AI package not found. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "google-generativeai"])
    import google.generativeai as genai

# Check if manim is installed
try:
    import manim
except ImportError:
    print("Manim not found. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "manim"])

# Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")  # Make sure to set your API key
if not GEMINI_API_KEY:
    GEMINI_API_KEY = input("Please enter your Google Gemini API key: ").strip()
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)


def generate_manim_code(topic: str) -> str:
    """Generate Manim code for a given topic using Gemini."""
    prompt = f"""
    I need complete, runnable Python code using the Manim library to create an educational animation about {topic}, in the style of 3Blue1Brown.

    The animation should:
    - Start with an intuitive introduction to {topic}
    - Visualize the key concepts clearly
    - Show important equations or formulas as needed
    - Include well-timed animations that help understanding
    - Have a logical flow from basic concepts to more advanced ideas
    - Focus on clarity and educational value
    - Use appropriate visual elements (graphs, equations, vectors, etc.)
    - Include smooth animations and transitions
    - Add explanatory text labels where appropriate
    - Make sure the words and object is not overlapping
    - Please remove the remaining words and graph from the last scene if necessary

    Provide ONLY the complete Python code without any explanations or markdown formatting.
    The code should be complete and runnable without any modifications needed.
    """

    print(f"Generating Manim animation code for '{topic}'...")

    try:
        # List available models first to confirm what we can use
        models = genai.list_models()
        available_models = [model.name for model in models]
        print(f"Available models: {available_models}")

        # Choose the best available model
        model_name = None
        preferred_models = ["gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
        for preferred in preferred_models:
            if any(preferred in model for model in available_models):
                for model in available_models:
                    if preferred in model:
                        model_name = model
                        break
                if model_name:
                    break

        if not model_name:
            print("No suitable Gemini model found. Using the first available model.")
            model_name = available_models[0] if available_models else "gemini-pro"

        print(f"Using model: {model_name}")

        # Configure the model
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        # Use the selected model for code generation
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Extract code from response
        code = response.text.strip()

        # Remove any markdown code blocks if present
        if code.startswith("```python"):
            code = code.split("```python", 1)[1]
        elif code.startswith("```"):
            code = code.split("```", 1)[1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]

        code = code.strip()
        return code

    except Exception as e:
        print(f"Error generating code: {e}")

        # Fallback to local template if API fails
        print("Using fallback template for basic vector animation...")
        if topic.lower() == "vector":
            return create_fallback_vector_code()
        else:
            print("No fallback template available for this topic.")
            return None


def create_fallback_vector_code():
    """Create a basic vector animation as fallback."""
    return """
from manim import *

class VectorIntroduction(Scene):
    def construct(self):
        # Title
        title = Text("Introduction to Vectors")
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.scale(0.6).to_edge(UP))
        self.wait(0.5)

        # Create a coordinate system
        plane = NumberPlane(
            x_range=[-5, 5, 1],
            y_range=[-3, 3, 1],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.6
            }
        )
        self.play(Create(plane), run_time=1.5)

        # Define vectors
        vector1 = Arrow(plane.coords_to_point(0, 0), plane.coords_to_point(2, 1), 
                        buff=0, color=RED)
        vector2 = Arrow(plane.coords_to_point(0, 0), plane.coords_to_point(-1, 2), 
                        buff=0, color=GREEN)

        # Label for vectors
        vector1_label = MathTex(r"\\vec{a} = (2, 1)").next_to(vector1.get_end(), RIGHT)
        vector2_label = MathTex(r"\\vec{b} = (-1, 2)").next_to(vector2.get_end(), LEFT)

        # Show vectors
        self.play(GrowArrow(vector1))
        self.play(Write(vector1_label))
        self.wait(1)

        self.play(GrowArrow(vector2))
        self.play(Write(vector2_label))
        self.wait(1)

        # Vector Addition
        vector_sum = Arrow(plane.coords_to_point(0, 0), 
                          plane.coords_to_point(1, 3), 
                          buff=0, color=YELLOW)
        vector_sum_label = MathTex(r"\\vec{a} + \\vec{b} = (1, 3)").next_to(vector_sum.get_end(), RIGHT)

        # Demonstrate addition
        v1_copy = vector1.copy().set_opacity(0.5)
        v2_copy = vector2.copy().set_opacity(0.5)

        self.play(
            v2_copy.animate.shift(vector1.get_end() - plane.coords_to_point(0, 0))
        )
        self.wait(1)

        self.play(GrowArrow(vector_sum))
        self.play(Write(vector_sum_label))
        self.wait(1)

        # Vector scaling
        scaled_vector = Arrow(plane.coords_to_point(0, 0), 
                             plane.coords_to_point(4, 2), 
                             buff=0, color=PURPLE)
        scaled_label = MathTex(r"2\\vec{a} = (4, 2)").next_to(scaled_vector.get_end(), RIGHT)

        self.play(
            FadeOut(v1_copy),
            FadeOut(v2_copy),
            FadeOut(vector_sum),
            FadeOut(vector_sum_label),
            FadeOut(vector2),
            FadeOut(vector2_label)
        )

        self.play(ReplacementTransform(vector1, scaled_vector))
        self.play(
            ReplacementTransform(vector1_label, scaled_label)
        )
        self.wait(1)

        # Final notes
        final_text = Text("Vectors have both magnitude and direction", 
                         font_size=24).to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(2)
"""


def save_and_run_code(code: str, topic: str) -> Optional[str]:
    """Save the generated code to a file and run it with Manim."""
    if not code:
        return None

    # Create a safe filename from the topic
    safe_topic = ''.join(c if c.isalnum() else '_' for c in topic)
    filename = f"manim_{safe_topic}.py"

    # Save the code to a file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"Code saved to {filename}")

    # Find the main Scene class name
    scene_class = None
    for line in code.split('\n'):
        if line.strip().startswith("class ") and "(Scene)" in line:
            scene_class = line.split("class ", 1)[1].split("(", 1)[0].strip()
            break

    if not scene_class:
        print("Error: Could not find a Scene class in the generated code.")
        return None

    # Run the Manim code
    print(f"Running Manim animation for scene: {scene_class}")
    try:
        command = [
            sys.executable, "-m", "manim",
            filename, scene_class,
            "-p",  # Play the animation once done
            "--quality", "m"  # Medium quality for faster rendering
        ]
        subprocess.run(command, check=True)

        # Return the output video path (approximate based on manim conventions)
        video_dir = os.path.join("media", "videos", os.path.splitext(filename)[0])
        if os.path.exists(video_dir):
            for file in os.listdir(video_dir):
                if file.endswith(".mp4") and scene_class in file:
                    return os.path.join(video_dir, file)
    except subprocess.CalledProcessError as e:
        print(f"Error running Manim: {e}")
        print("The generated code may have errors.")

        # Offer to edit the code
        edit_choice = input("Would you like to edit the code before trying again? (y/n): ")
        if edit_choice.lower() == 'y':
            # Open in default editor
            if os.name == 'nt':  # Windows
                os.system(f"notepad {filename}")
            else:  # macOS and Linux
                editor = os.environ.get('EDITOR', 'nano')
                os.system(f"{editor} {filename}")

            # Try running again
            retry = input("Run the edited code? (y/n): ")
            if retry.lower() == 'y':
                return save_and_run_code(open(filename).read(), topic)

    return None


def main():
    print("=" * 80)
    print("💫 Topic to 3Blue1Brown-style Animation Generator (Gemini Version) 💫")
    print("=" * 80)
    print("This program generates educational animations in the style of 3Blue1Brown")
    print("using the Manim library based on your chosen math or physics topic.")
    print("Powered by Google Gemini AI")
    print()

    while True:
        topic = input("\nEnter a mathematics or physics topic (or 'quit' to exit): ")
        if topic.lower() in ('quit', 'exit'):
            break

        if not topic.strip():
            print("Please enter a valid topic.")
            continue

        # Generate code
        code = generate_manim_code(topic)
        if not code:
            print("Failed to generate code. Please try another topic.")
            continue

        # Run the animation
        video_path = save_and_run_code(code, topic)

        if video_path and os.path.exists(video_path):
            print(f"\n✨ Animation successfully created: {video_path}")
        else:
            print("\n❌ There were issues creating the animation.")

        choice = input("\nGenerate another animation? (y/n): ")
        if choice.lower() != 'y':
            break

    print("\nThank you for using the 3B1B-style Animation Generator!")


if __name__ == "__main__":
    main()