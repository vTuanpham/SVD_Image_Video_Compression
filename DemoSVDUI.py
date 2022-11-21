import concurrent.futures
from kivy.config import Config
MIN_SIZE = (1000,650)
Config.set('graphics','width',MIN_SIZE[0])
Config.set('graphics','height',MIN_SIZE[1])
import kivy
from kivy.app import App
from kivy.graphics import Rectangle, Color
from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.uix.screenmanager import  ScreenManager,Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.properties import StringProperty
from kivy.uix.image import Image
from kivy.uix.video import Video
from kivy.uix.gridlayout import GridLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import time
import svdbaend
import re
import cv2 as cv
import numpy as np

class MainWindow(Screen):
    filePath = StringProperty('')
    def __init__(self,**kwargs):
        super(MainWindow,self).__init__(**kwargs)
        Window.bind(on_drop_file=self._on_file_drop)
        # Window.bind(on_drop_begin=self._on_drop_begin)
    def _on_file_drop(self,window,file_path,x,y):
        App.get_running_app().restart()
        self.ids.vid.state = 'pause'
        self.ids.vid.unload()
        self.filePath = file_path.decode("utf-8")
        if re.search('.mp4',self.filePath) != None:
            self.srcmode = 'video'
            self.ids.img.opacity = 0
            self.ids.vid.source = self.filePath
            self.ids.vid.state = 'play'
            self.ids.vid.options = {'eos': 'stop'}
        else:
            self.srcmode = 'image'
            self.ids.vid.opacity = 0
            self.ids.img.source = self.filePath
            self.ids.img.reload()
    def reset(self,btn):
        App.get_running_app().restart()

    # def _on_drop_begin(self,window,x,y):
    #     self.ids.bgL.background_color = (0,1,1,0)
    #     self.ids.bgL.color = (1,0,0,1)
class RankDropDown(DropDown):
    pass
class ModeDropDown(DropDown):
    pass
class ImageShow(Image,Widget):
    pass
class VideoShow(Video,Widget):
    pass
class ImageVideoWindow(Screen):
    def __init__(self,**kwargs):
        super(ImageVideoWindow,self).__init__(**kwargs)
    btn = Button()
    rankk = None
    rankOpt = 'full'
    mode = 'rgb'
    percentage = 1
    def on_transit(self,Srcmode='empty',filePath=''):
        if Srcmode == 'image':
            org = ImageShow()
            self.ids['org'] = org
            org.size_bg = org.size
            org.pos_bg = org.pos
            org.background_color = .5,.5,.55,1
            org.source = filePath
            self.ids.showGrid.add_widget(org)
            svd = ImageShow()
            self.ids['svd'] = svd
            svd.size_bg = svd.size
            svd.pos_bg = svd.pos
            svd.background_color = .5,.5,.55,1
            self.ids.showGrid.add_widget(svd)
        if Srcmode == 'video':
            org = VideoShow()
            self.ids['org'] = org
            org.size_bg = org.size
            org.pos_bg = org.pos
            org.background_color = .5,.5,.55,1
            org.source = filePath
            self.ids.showGrid.add_widget(org)
            org.state = 'play'
            org.options = {'eos': 'stop'}
            svd = VideoShow()
            self.ids['svd'] = svd
            svd.size_bg = svd.size
            svd.pos_bg = svd.pos
            svd.background_color = .5,.5,.55,1
            self.ids.showGrid.add_widget(svd)
            svd.state = 'play'
            svd.options = {'eos': 'stop'}
    def on_rank_dropdown(self, btn):
        rDropdown = RankDropDown()
        self.btn = btn
        btn.bind(on_release=rDropdown.open)
        rDropdown.bind(on_select=lambda instance, x: setattr(btn, 'text', x))
        rDropdown.bind(on_dismiss=self.get_User_choice_rank)
    def on_mode_dropdown(self,btn):
        mDropdown = ModeDropDown()
        self.btn = btn
        btn.bind(on_release=mDropdown.open)
        mDropdown.bind(on_select=lambda instance, x: setattr(btn, 'text', x))
        mDropdown.bind(on_dismiss=self.get_User_choice_mode)
    def get_User_choice_rank(self,x):
        if self.btn.text != '' and self.btn.text!= 'Select rank options':
            self.rankOpt = self.btn.text
    def get_User_choice_mode(self,x):
        if self.btn.text != '' and self.btn.text != 'Select mode':
            self.mode = self.btn.text
    def on_process(self,btn,img):
        if re.search('.mp4',img)!= None:
            if self.ids.rankinput.text != '':
                self.rankk = int(self.ids.rankinput.text)
            svdbaend.videosvd(img,rankk=self.rankk,rankOpt=self.rankOpt,mode = self.mode,multiThreaded = True)
            self.ids.svd.source = 'resultVid.mp4'
            self.ids.svd.reload()
            self.ids.svd.state = 'play'
        else:
            if self.ids.rankinput.text != '':
                self.rankk = int(self.ids.rankinput.text)
            svdbaend.imagesvd(img,rankk=self.rankk,rankOpt=self.rankOpt,mode = self.mode,forVid = False,multiThreaded = True)
            self.ids.svd.source = 'result.png'
            self.ids.svd.reload()
    def on_enter_rank(self,text):
        num = int(text)
        if num <= 0:
            self.ids.warning.opacity = 1
        else:
            self.ids.warning.opacity = 0
class WebcamWindow(Screen):
    def __init__(self, **kwargs):
        super(WebcamWindow, self).__init__(**kwargs)

        # org = Image()
        # self.ids['org'] = org
        # self.ids.showGrid.add_widget(org)
        #
        # svd = Image()
        # self.ids['svd'] = svd
        # self.ids.showGrid.add_widget(svd)

    btn = Button()
    rankk = None
    rankOpt = 'full'
    mode = 'rgb'

    def on_transit(self):
        self.capture = cv.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 64.0)
        org = Image()
        self.ids['org'] = org
        self.ids.showGrid.add_widget(org)

        svd = Image()
        self.ids['svd'] = svd
        self.ids.showGrid.add_widget(svd)
    def update(self, dt):
        ret, frame = self.capture.read()
        frame = cv.resize(frame, dsize=None, fx=0.6, fy=0.6)
        svdframe = np.array(svdbaend.imagesvd(frame,rankk = self.rankk,rankOpt = self.rankOpt,mode = self.mode))
        # convert it to texture
        buf1 = cv.flip(frame, 0)
        buf11 = buf1.tostring()

        buf2 = cv.flip(svdframe, 0)
        buf22 = buf2.tostring()
        colorfmt = 'bgr'
        if self.mode == 'gs':
            colorfmt = 'luminance'
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture2 = Texture.create(size=(svdframe.shape[1], svdframe.shape[0]), colorfmt=colorfmt)
        texture1.blit_buffer(buf11,colorfmt='bgr', bufferfmt='ubyte')
        texture2.blit_buffer(buf22,colorfmt=colorfmt, bufferfmt='ubyte')

        # display image from the texture
        self.ids.org.texture = texture1
        self.ids.svd.texture = texture2
    def on_reload(self):
        Clock.unschedule(self.update)
        self.capture.release()
        self.ids.showGrid.clear_widgets()
        self.on_transit()
        if self.ids.rankinput.text != '':
            self.rankk = int(self.ids.rankinput.text)
    def on_back(self):
        Clock.unschedule(self.update)
        self.capture.release()
        self.btn = Button()
        self.rankk = None
        self.rankOpt = 'full'
        self.mode = 'rgb'
    def on_rank_dropdown(self, btn):
        rDropdown = RankDropDown()
        self.btn = btn
        btn.bind(on_release=rDropdown.open)
        rDropdown.bind(on_select=lambda instance, x: setattr(btn, 'text', x))
        rDropdown.bind(on_dismiss=self.get_User_choice_rank)
    def on_mode_dropdown(self,btn):
        mDropdown = ModeDropDown()
        self.btn = btn
        btn.bind(on_release=mDropdown.open)
        mDropdown.bind(on_select=lambda instance, x: setattr(btn, 'text', x))
        mDropdown.bind(on_dismiss=self.get_User_choice_mode)
    def get_User_choice_rank(self,x):
        if self.btn.text != '' and self.btn.text!= 'Select rank options':
            self.rankOpt = self.btn.text
    def get_User_choice_mode(self,x):
        if self.btn.text != '' and self.btn.text != 'Select mode':
            self.mode = self.btn.text
    def on_enter_rank(self,text):
        num = int(text)
        if num <= 0:
            self.ids.warning.opacity = 1
        else:
            self.ids.warning.opacity = 0

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("svdtemplate.kv")

class MyMainApp(App):
    def check_resize(self,instance,x,y):
        if x < MIN_SIZE[0]:
            Window.size = (1000 , Window.size[1])
        if y < MIN_SIZE[1]:
            Window.size = (Window.size[0],650)
    def restart(self):
        Imageshow = self.root.screens[0].children[2]
        Imageshow.source = str(None)
        Videoshow = self.root.screens[0].children[1]
        Videoshow.state = 'pause'
        Videoshow.unload()
        Videoshow.source = str(None)
        self.root.screens[0].srcmode = 'empty'
        self.root.screens[1].children[0].children[1].srcmode = 'empty'
        self.root.screens[1].children[0].children[1].clear_widgets()
        self.root.screens[0].children[0].children[1].opacity = 1
    def build(self):
        Window.bind(on_resize=self.check_resize)
        return kv
if __name__=="__main__":
    MyMainApp().run()


