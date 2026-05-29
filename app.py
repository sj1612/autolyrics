import sys
import os

# Add src/ to system path to import modules if necessary
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the demo application block
from demo import demo

if __name__ == "__main__":
    # Launch Gradio interface (Hugging Face Spaces automatically sets the port and handles routing)
    demo.launch()
