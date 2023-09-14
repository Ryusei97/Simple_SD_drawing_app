
import tkinter as tk
import io
import torch
from PIL import Image, ImageTk
from diffusers import StableDiffusionImg2ImgPipeline
from time import time

# Load model. Change device type if gpu is available
device = "mps"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("nitrosocke/Ghibli-Diffusion").to(device)
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)


class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.canvas_width = 512
        self.canvas_height = 512
        self.brush_size = 16
        self.color = "black"
        self.model = None
        self.previous_x = None
        self.previous_y = None
        self.text_prompt = None
        
        # Create canvas
        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(expand = True, fill="both", side="left")
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.unclick)
        
        # Create prediction display
        self.img_display = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.img_display.pack(expand=True, fill="both", side="right")

        # Create clear button
        clear_button = tk.Button(self.master, text="Clear Canvas", command=self.clear_canvas)
        clear_button.pack(side=tk.BOTTOM)
        
        # Create line width slider
        self.line_width_slider = tk.Scale(self.master, from_=1, to=50, orient="horizontal", label="Line Width",
                                          command=self.update_line_width)
        self.line_width_slider.pack()

        # Create text input section
        self.text_input = tk.Entry(self.master)
        self.text_input.pack(side=tk.BOTTOM)
        self.text_input.bind("<Return>", self.process_text)

        # Create label to display entered text
        self.display_label = tk.Label(self.master)
        self.display_label.configure(text=f"Prompt: ")
        self.display_label.pack(side=tk.BOTTOM)

        # Create generate image button
        generate_button = tk.Button(self.master, text="Generate", command=self.generate_image)
        generate_button.pack(side=tk.BOTTOM)

        # Create color picker dropdown
        self.color_picker = tk.StringVar()
        self.color_picker.set("black")  # Initial color is black
        self.color_menu = tk.OptionMenu(self.master, self.color_picker, 
                                        "black", "white", "Gray", "Brown", "pink", "red", "orange",  
                                        "yellow", "green", "blue", "purple")
        self.color_menu.pack()

        
    def draw(self, event):
        x = event.x
        y = event.y
        if self.previous_x is not None and self.previous_y is not None:
            self.canvas.create_line(self.previous_x, self.previous_y, x, y, width=self.brush_size,
                                    fill=self.color_picker.get(), capstyle=tk.ROUND, smooth=True)
        self.previous_x = x
        self.previous_y = y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.previous_x = None
        self.previous_y = None
    
    def unclick(self, event):
        self.previous_x = None
        self.previous_y = None
    
    def update_line_width(self, value):
        self.brush_size = int(value)
    
    def run(self):
        self.master.mainloop()

    def process_text(self, event):
        text = self.text_input.get()
        # Process the entered text
        self.text_prompt = text
        self.display_label.configure(text=f"Prompt: {text}")
        self.text_input.delete(0, tk.END)
    
    def generate_image(self):
        ps_data = self.canvas.postscript(colormode="color")
        init_image = Image.open(io.BytesIO(ps_data.encode("utf-8"))).convert("RGB")

        generator = torch.Generator(device=device).manual_seed(1024)
        image = pipe(prompt=self.text_prompt, image=init_image,
                        strength=0.85, guidance_scale=7, generator=generator).images[0]

        tk_image = ImageTk.PhotoImage(image)
        self.img_display.delete("all")

        self.img_display.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.img_display.image = tk_image

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    app.run()
