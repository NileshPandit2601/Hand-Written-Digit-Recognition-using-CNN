from tkinter import *
import numpy as np
from PIL import ImageGrab
from Prediction import predict

class HandwrittenDigitRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Recognition")
        self.lastx, self.lasty = None, None

        # Load model parameters
        try:
            self.Theta1 = np.loadtxt('Theta1.txt')
            self.Theta2 = np.loadtxt('Theta2.txt')
        except Exception as e:
            print("Error loading model parameters:", e)
            self.show_error("Error loading model parameters.")
            return

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Title Label
        self.title_label = Label(self.master, text="Handwritten Digit Recognition", font=('Algerian', 25), fg="blue")
        self.title_label.place(x=35, y=10)

        # Canvas for drawing
        self.cv = Canvas(self.master, width=350, height=290, bg='black')
        self.cv.place(x=120, y=70)
        self.cv.bind('<Button-1>', self.event_activation)

        # Clear Canvas Button
        self.clear_button = Button(self.master, text="1. Clear Canvas", font=('Algerian', 15), bg="orange", fg="black", command=self.clear_widget)
        self.clear_button.place(x=120, y=370)

        # Prediction Button
        self.predict_button = Button(self.master, text="2. Prediction", font=('Algerian', 15), bg="white", fg="red", command=self.MyProject)
        self.predict_button.place(x=320, y=370)

        # Exit Button
        self.exit_button = Button(self.master, text="Exit", font=('Algerian', 15), bg="red", fg="white", command=self.master.quit)
        self.exit_button.place(x=250, y=420)

        # Label for displaying prediction result
        self.result_label = Label(self.master, text="", font=('Algerian', 20))
        self.result_label.place(x=230, y=450)

    def clear_widget(self):
        self.cv.delete("all")
        self.result_label.config(text="")

    def event_activation(self, event):
        self.cv.bind('<B1-Motion>', self.draw_lines)
        self.lastx, self.lasty = event.x, event.y

    def draw_lines(self, event):
        x, y = event.x, event.y
        self.cv.create_line((self.lastx, self.lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
        self.lastx, self.lasty = x, y

    def MyProject(self):
        widget = self.cv
        x = self.master.winfo_rootx() + widget.winfo_x()
        y = self.master.winfo_rooty() + widget.winfo_y()
        x1 = x + widget.winfo_width()
        y1 = y + widget.winfo_height()

        # Capture and resize image
        img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))
        img = img.convert('L')  # Convert to grayscale

        # Extract pixel matrix and convert to vector
        x = np.asarray(img)
        vec = x.flatten().reshape(1, 784)  # Flatten and reshape to (1, 784)

        # Prediction
        try:
            pred = predict(self.Theta1, self.Theta2, vec / 255.0)  # Normalize
            self.result_label.config(text="Digit = " + str(pred[0]))
        except Exception as e:
            print("Error during prediction:", e)
            self.show_error("Prediction error.")

    def show_error(self, message):
        self.result_label.config(text=message, fg='red')

if __name__ == "__main__":
    root = Tk()
    app = HandwrittenDigitRecognitionApp(root)
    root.geometry("600x500")
    root.mainloop()