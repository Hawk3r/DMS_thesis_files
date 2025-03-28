import kivy
import kivymd
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
from kivymd.uix.textfield import textfield
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.slider import Slider
from plyer import tts


# Define KV content as a string
kv_code = """
<MyWidget>:

    Label:
        text: "Hello, Kivy!"
        font_size: 40
        color: 1, 0, 0, 1  # Red color (RGBA)
    Button:
        text: "speak"
        font_size: 32
        size: 100,50
        on_press:root.hello()
"""

# Load the KV string
Builder.load_string(kv_code)

class MyWidget(BoxLayout):
    def hello(self):
        tts.speak("hello world")
    pass

class MyApp(App):
    def build(self):
        return MyWidget()

if __name__ == "__main__":
    MyApp().run()
