import tkinter as tk
import numpy as np
from neural_network import Network


class ResultGUI:
    def __init__(self, test_data: list[tuple[np.ndarray, np.ndarray]], network: Network, size: int = 560):
        self.set = test_data
        self.i = 0
        self.network = network
        self.size = size
        self.window = tk.Tk()
        self.window.title("Recognition-Results")
        self.window.resizable(0, 0)
        self.canvas = tk.Canvas(width=size, height=size)
        self.label = tk.Label()
        self.output = tk.Label()
        self.button = tk.Button(text="Next", command=self.next)
        self.print()
        self.window.mainloop()

    def next(self):
        self.i += 1
        self.print()

    def print(self):
        pixel_size = int(self.size / 28)
        self.canvas.delete("all")
        for i in range(28):
            for j in range(28):
                pixel = self.set[self.i][0][i * 28 + j]
                x, y = j * pixel_size, i * pixel_size
                self.canvas.create_rectangle(x, y, x + pixel_size, y + pixel_size, fill="#" + hex(int(255 * (1 - pixel)))[2:] * 3, width=0)
        self.label["text"] = f"Label: {self.set[self.i][1].argmax()}"
        self.output["text"] = f"Output: {self.network.feed_forward(self.set[self.i][0].argmax())}"
        self.canvas.pack()
        self.label.pack()
        self.output.pack()
        self.button.pack()


class DrawGUI:
    def __init__(self, network: Network, size: int = 560, brush_size: int = 45):
        self.network = network
        self.size = size
        self.brush_size = brush_size
        self.image = [[0 for __ in range(size)] for _ in range(size)]
        self.window = tk.Tk()
        self.window.title("Live-Recognition")
        self.window.resizable(0, 0)
        self.canvas = tk.Canvas(width=size, height=size)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-3>", lambda event: self.clear())
        self.label = tk.Label()
        self.button = tk.Button(text="Predict", command=self.predict)
        self.transform = tk.Button(text="Transform", command=self.transform)
        self.canvas.pack()
        self.label.pack()
        self.button.pack()
        self.transform.pack()
        self.window.mainloop()

    def paint(self, event):
        half_brush = int(self.brush_size / 2)
        x, y = event.x - half_brush, event.y - half_brush
        if 0 <= x <= self.size - self.brush_size and 0 <= y <= self.size - self.brush_size:
            for cx in range(x, x + self.brush_size):
                for cy in range(y, y + self.brush_size):
                    distance = ((event.x - cx) ** 2 + (event.y - cy) ** 2) ** 0.5
                    if not distance > half_brush:
                        self.image[cy][cx] = 1
            self.canvas.create_oval(x, y, x + self.brush_size, y + self.brush_size, fill="#000", width=0)

    def clear(self):
        del self.image
        self.image = [[0 for __ in range(self.size)] for _ in range(self.size)]
        self.canvas.delete("all")

    def predict(self):
        pixels = np.zeros(784)
        pixel_size = int(self.size / 28)
        for i in range(28):
            for j in range(28):
                pixel = 0
                x, y = j * pixel_size, i * pixel_size
                for cy in range(y, y + pixel_size):
                    for cx in range(x, x + pixel_size):
                        pixel += self.image[cy][cx]
                pixels[i * 28 + j] = pixel / pixel_size ** 2
        output = self.network.feed_forward(pixels)
        index = output.argmax()
        self.label["text"] = f"Prediction: {index} - certainty: {output[index] * 100}%"

    def transform(self):
        pixel_size = int(self.size / 28)
        window = tk.Toplevel(self.window)
        window.title("Transformed")
        window.resizable(0, 0)
        canvas = tk.Canvas(master=window, width=self.size, height=self.size)
        for i in range(28):
            for j in range(28):
                pixel = 0
                x, y = j * pixel_size, i * pixel_size
                for cy in range(y, y + pixel_size):
                    for cx in range(x, x + pixel_size):
                        pixel += self.image[cy][cx]
                canvas.create_rectangle(x, y, x + pixel_size, y + pixel_size, fill="#" + hex(int(255 * (1 - pixel / pixel_size ** 2)))[2:] * 3, width=0)
        canvas.pack()
        window.mainloop()
