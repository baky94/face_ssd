
import simpleguitk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import time
from PIL import ImageTk, Image
import xml.etree.ElementTree as xml_tree
import os.path, sys, glob, time
import cv2
import numpy
from datetime import datetime


set_debug_level = 0

def init_k():
  global help_text, configm_msg, OName, OShow, OSearch, OunSelect, OSave, OErase, FName, FRead, FWrite, ASave
  help_text = ("<<객체선택 조작>>\n"
       "전체선택  : Ctl-A \n"
       "전체선택취소 : ESC\n"
       "선책추가  : 왼쪽마우스 더블클릭 \n"
       "선책취소  : 오른마우스 클릭 \n"
       "선택삭제   : Del   \n"
       "자동분석 : Ctrl-P  \n"
       "객체명저장 : Enter  \n"
       "\n"
       "<<파일>>"
       "파일 읽기 : Ctl-R    \n"
       "파일 저장 : Ctl-S    \n"
       "이전 파일 : PageUp   \n"
       "다음 파일 : PageDown   \n"
       "\n"
       "<<객체생성>>\n"
       "-Drag하여 객체영역선택\n"
       "-객체명을 수정한후 저장[Enter]\n"
       "\n"
       "<<객체명 변경>>\n"
       "-여러개 객체를 더블클릭\n"
       "-객체명을 수정한 후 저장[Enter]\n"
       "\n"
       "<<객체명 이동>>\n"
       "-객체선택후 화살표\n"
       "<<객체명 크기변경>>\n"
       "-객체선택후 Cntl+화살표\n")

  configm_msg = "총 {} 개의 선택 캑체의 명칭 및 (diffcult, truncated) 을 {}({},{}) 로 통일할까요 ?"
  ASave = "자동저장"
  OName = "O명"
  OShow = "O보임"
  OSearch = "검색"
  OunSelect = "O취소"
  OSave = "O저장"
  OErase = "O지움"
  FName = "파일명"
  FRead = "읽기"
  FWrite = "저장"

def init_e():
  global help_text, configm_msg, OName, OShow, OSearch, OunSelect, OSave, OErase, FName, FRead, FWrite, ASave
  help_text = ("<< Object selection operation >> \n"
    "Select all: Ctl-A \n"
    "Unselect all: ESC \n"
    "Add measure: left mouse double click \n"
    "Cancel the measure: right click \n"
    "Delete selected: Del \n"
    "automatic analysis: Ctrl-P \n"
    "Save Object Name: Enter \n"
    "\n"
    "<< File >>"
    "Read File: Ctl-R \n"
    "Save file: Ctl-S \n"
    "Previous file: PageUp \n"
    "The following file: PageDown \n"
    "\n"
    "<< Create Object >> \n"
    "-Drag to select object region \n"
    "- modify the objectname and save it[Enter]\n"
    "\n"
    "<< rename object >> \n"
    "- double click on multiple objects \n"
    "- modify the objectname and save it[Enter]\n"
    "\n"
    "<< Move Object Name >> \n"
    "- arrow to select object \n"
     "<< resize object name >> \n"
     "- Cntl + arrow after selecting object \n"
     )

  configm_msg = "Do you want to unify the names (diffcult, truncated) of the {}({},{}) selection bodies with {}?"
  ASave = "AutoSave"
  OName   = "ObjName"
  OShow   = "ObjShow"
  OSearch   = "Search"
  OunSelect = "ObjUnselect"
  OSave   = "ObjSave"
  OErase  = "ObjErase"
  FName   = "FileName"
  FRead   = "FileRead"
  FWrite  = "FileWrite"


def log(debug, s):
  global set_debug_level
  if debug > set_debug_level:
    return
  print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], s)


class Obj:
  def __init__(self, name, xmin, ymin, xmax, ymax, truncated=0, difficult=0, objectBox=None, objectNm=None,
         objectNmBg=None, parent=None):
    self.name = name
    self.xmin = xmin
    self.ymin = ymin
    self.xmax = xmax
    self.ymax = ymax
    self.truncated = truncated
    self.difficult = difficult
    self.objectBox = objectBox
    self.objectNm = objectNm
    self.objectNmBg = objectNmBg
    self.parent = parent
  def __str__(self):
    return "<{}<{},{} ~ {},{}>_".format(self.name, self.xmin, self.ymin, self.xmax, self.ymax)

class Rect:
  def __init__(self, x, y, w, h):
    self.x = x
    self.y = y
    self.w = w
    self.h = h

  def list(self):
    return [[self.x, self.y], [self.x, self.y + self.h], [self.x + self.w, self.y + self.h],
        [self.x + self.w, self.y]]

  def __str__(self):
    return "<{},{} ~ {},{}>".format(self.x, self.y, self.w, self.h)


class VocEditor:
  def __init__(self, image_list):
    self.init_image(image_list)
    self.init_screen()
    self.loadXml()

  def init_image(self, image_list):
    # image
    self.image_list = image_list
    self.curr_image_index = 0

    # working image object
    self.org_img = None
    self.canvas_img = None
    self.org_img_width = 0
    self.org_img_height = 0
    self.object_list = []
    self.select_list = []

    # Canvas,  GUI Screen
    self.canvas_img_width = 1
    self.canvas_img_height = 1

  def init_screen(self):
    # CUI
    self.root = Tk()
    self.Title = "Pascal VOC Data Editor"
    self.root.title(self.Title)

    # toolbar var
    self.curr_obj_name = StringVar()
    self.curr_file_name = StringVar()
    self.is_autosave = IntVar()
    self.is_autosave.set(1)
    self.is_show_label = IntVar()
    self.is_show_label.set(1)
    self.is_difficult = IntVar()
    self.is_difficult.set(0)
    self.is_truncated = IntVar()
    self.is_truncated.set(0)

    # Tool bar , event define
    toolbar = Frame(self.root, height=3)
    toolbar.pack(side=TOP, anchor=NW)
    
    self.btn_flist = Button(toolbar, text="File", command=self.onFlist)
    self.btn_flist.pack(side=LEFT)
    self.root.bind('<Control-O>', self.onFlist)
    
    self.label3 = Label(toolbar, text=FName)
    self.label3.pack(side=LEFT)
    self.curr_fname = Entry(toolbar, textvariable=self.curr_file_name, bd=1, width=20)
    self.curr_fname.pack(side=LEFT)

    self.btn_read = Button(toolbar, text=FRead, command=self.onRead)
    self.btn_read.pack(side=LEFT)
    self.root.bind('<Control-r>', self.onRead)
    self.btn_save = Button(toolbar, text=FWrite, command=self.onSave)
    self.btn_save.pack(side=LEFT)
    self.root.bind('<Control-s>', self.onSave)
    self.btn_prev = Button(toolbar, text="<<", command=self.onPrev)
    self.btn_prev.pack(side=LEFT)
    self.root.bind('<Prior>', self.onPrev)
    self.btn_next = Button(toolbar, text=">>", command=self.onNext)
    self.btn_next.pack(side=LEFT)
    self.root.bind('<Next>', self.onNext)
    self.auto_save = Checkbutton(toolbar, text=ASave, variable=self.is_autosave, command=self.onAutosave)
    self.auto_save.pack(side=LEFT)

    Label(toolbar, width=1).pack(side=LEFT)
    self.label1 = Label(toolbar, text=OName)
    self.label1.pack(side=LEFT)
    self.entry1 = Entry(toolbar, textvariable=self.curr_obj_name, bd=3, width=20)
    self.entry1.pack(side=LEFT)
    self.chk_difficult = Checkbutton(toolbar, text="di.", variable=self.is_difficult, command=self.ondifficult)
    self.chk_difficult.pack(side=LEFT)
    self.chk_truncated = Checkbutton(toolbar, text="tr.", variable=self.is_truncated, command=self.onTruncated)
    self.chk_truncated.pack(side=LEFT)
    Label(toolbar, width=1).pack(side=LEFT)
    self.btn_Update = Button(toolbar, text=OSave, command=self.onUpdate)
    self.btn_Update.pack(side=LEFT)
    self.root.bind('<Return>', self.onUpdate)

    self.chk_show_label = Checkbutton(toolbar, text=OShow, variable=self.is_show_label, command=self.onShowLabel)
    self.chk_show_label.pack(side=LEFT)

    self.btn_search = Button(toolbar, text=OSearch, command=self.onSearch)
    self.btn_search.pack(side=LEFT)
    Label(toolbar, width=1).pack(side=LEFT)

    self.btn_unselect = Button(toolbar, text=OunSelect, command=self.onUnSelectAll)
    self.btn_unselect.pack(side=LEFT)
    self.root.bind('<Escape>', self.onUnSelectAll)
    self.btn_erase = Button(toolbar, text=OErase, command=self.onErase)
    self.btn_erase.pack(side=LEFT)
    self.root.bind('<Delete>', self.onErase)

    Label(toolbar, width=1).pack(side=LEFT)

    
    self.btn_next = Button(toolbar, text="Help", command=self.onHelp)
    self.btn_next.pack(side=LEFT)
    self.root.bind('<Control-h>', self.onHelp)


    # image mouse event
    self.canvas = Canvas(self.root, width=800, height=600)
    self.canvas.pack(fill=BOTH, expand=YES)
    self.canvas.bind('<Button-1>', self.onStart)
    self.canvas.bind('<B1-Motion>', self.onGrow)
    self.canvas.bind('<Double-1>', self.onSelect)
    self.canvas.bind('<Button-3>', self.onUnSelect1)
    self.canvas.bind("<Configure>", self.onResize)

    # self.root.bind("<Key>",self.key)
    self.root.bind('<Control-a>', self.onSelectAll)
    self.root.bind('<Control-p>', self.onAuto)
    # object move
    self.root.bind('<Up>', self.onUp)
    self.root.bind('<Down>', self.onDown)
    self.root.bind('<Left>', self.onLeft)
    self.root.bind('<Right>', self.onRight)
    self.root.bind('<Control-Up>', self.onCtrlUp)
    self.root.bind('<Control-Left>', self.onCtrlLeft)
    self.root.bind('<Control-Down>', self.onCtrlDown)
    self.root.bind('<Control-Right>', self.onCtrlRight)


  ##############################################################################
  # Button event
  ##############################################################################
  def onFlist(self, event=None):
    log(2, 'onNext')
    filez = filedialog.askopenfilenames(parent=self.root,title='Choose a file')
    filez = [file for file in filez if file.endswith('.jpg')]
    if len(filez) > 0:
      self.image_list = filez
      self.curr_image_index = 0
      self.loadFile()

  def onHelp(self, event=None):
    messagebox.showinfo("Help", help_text)

  def ondifficult(self):
    pass

  def onAutosave(self):
    pass

  def onTruncated(self):
    pass

  def onShowLabel(self):
    log(2, 'onShowLabel')
    self.displayBox()

  def onResize(self, event):
    log(1, 'onResize ({}, {})'.format(self.canvas_img_width, self.canvas_img_height))
    self.canvas_img_width = event.width - 2
    self.canvas_img_height = event.height - 2
    self.loadImage()
    self.displayBox()

  def onSearch(self, event=None):
    log(2, 'onSearch')

    del_list = []
    for obj in self.select_list:
      if obj in self.select_list:
        del_list.append(obj)
    for o in del_list:
      self.select_list.remove(o)


    # search same name
    for obj in self.object_list:
      if obj.name == self.curr_obj_name.get():
        self.select_list.append(obj)

    self.displayBox()

  def onUpdate(self, is_same_name=False):
    log(1, 'onUpdate {} {}'.format(is_same_name, len(self.select_list)))
    nsel = len(self.select_list)
    if nsel == 0:
      return

    same_name = 'no'
    if nsel > 1:
      same_name = messagebox.askquestion("confirm",
        configm_msg.format(nsel, self.toobarObjName(), self.is_difficult.get(), self.is_truncated.get()), icon='warning')
      if same_name != 'yes':
        return
    for obj in self.select_list:
      obj.name = self.toobarObjName()
      obj.difficult = self.is_difficult.get()
      obj.truncated = self.is_truncated.get()
      if obj not in self.object_list:
        # New Object <  Select Object
        parent = self.findParentObjectInSelectList(obj)
        obj.parent = parent
        self.object_list.append(obj)
        log(1, 'object_list.append({}'.format(obj))
    self.select_list.clear()
    self.displayBox()
    log(1, 'onUpdate add ok {}'.format(len(self.object_list)))
    self.beep1()

  def onErase(self, event=None):
    log(2, 'onErase {}'.format(len(self.select_list)))

    del_list = []
    if len(self.select_list) == 1 and self.select_list[0] not in self.object_list:
      r = self.select_list[0]
      del_list = self.findObjByRect(r.xmin, r.ymin, r.xmax, r.ymax)
    else:
      for obj in self.select_list:
        del_list.append(obj)
        for part in self.getPartList(obj):
          del_list.append(part)

    for o in del_list:
      if o.objectBox != None:
        self.canvas.delete(o.objectBox)
      if o.objectNm != None:
        self.canvas.delete(o.objectNm)
      if o.objectNmBg != None:
        self.canvas.delete(o.objectNmBg)
      if o in self.object_list:
        self.object_list.remove(o)

    log(2, 'onErase add ok {}'.format(len(self.object_list)))
    self.select_list.clear()

  def onAuto(self, event=None):
    log(2, 'onAuto')
    self.org_img_width, self.org_img_height, self.object_list = self.parseImageInfo(self.org_img)
    self.displayBox()

  def onRead(self, event=None):
    log(2, 'onRead')
    self.btn_read.config(state=DISABLED)
    self.loadFile()
    self.btn_read.config(state=NORMAL)
    self.beep1()

  def onSave(self, event=None):
    log(1, 'onSave nobj={}'.format(len(self.object_list)))
    self.saveVocXml(self.image_list[self.curr_image_index], self.org_img_width, self.org_img_height, self.object_list)
    self.beep1()

  def onPrev(self, event=None):
    log(2, 'onPrev')
    if self.is_autosave.get() == 1:
       self.saveVocXml(self.image_list[self.curr_image_index], self.org_img_width, self.org_img_height, self.object_list)
    if self.curr_image_index > 0:
      # self.btn_prev.config(state=DISABLED)
      self.curr_image_index -= 1
      self.loadFile()
    # self.btn_prev.config(state=NORMAL)
    self.beep1()
    
  def onNext(self, event=None):
    log(2, 'onNext')
    if self.is_autosave.get() == 1:
       self.saveVocXml(self.image_list[self.curr_image_index], self.org_img_width, self.org_img_height, self.object_list)
    if self.curr_image_index + 1 < len(self.image_list):
      # self.btn_next.config(state=DISABLED)
      self.curr_image_index += 1
      self.loadFile()
    # self.btn_next.config(state=NORMAL)
    self.beep1()


  def onSelectAll(self, event):
    log(2, 'onSelectAll')
    for obj in self.object_list:
      if obj not in self.select_list:
        self.select_list.append(obj)
    self.displayBox()

  def onUp(self, event=None):
    for obj in self.select_list:
      obj.ymin -= 1
      obj.ymax -= 1
    self.displayBox()

  def onDown(self, event=None):
    for obj in self.select_list:
      obj.ymin += 1
      obj.ymax += 1
    self.displayBox()

  def onLeft(self, event=None):
    for obj in self.select_list:
      obj.xmin -= 1
      obj.xmax -= 1
    self.displayBox()

  def onRight(self, event=None):
    for obj in self.select_list:
      obj.xmin += 1
      obj.xmax += 1
    self.displayBox()


  def onCtrlUp(self, event=None):
    for obj in self.select_list:
      obj.ymin -= 1
      obj.ymax += 1
    self.displayBox()

  def onCtrlDown(self, event=None):
    for obj in self.select_list:
      if obj.ymin + 4 < obj.ymax:
        obj.ymin += 1
        obj.ymax -= 1
    self.displayBox()

  def onCtrlLeft(self, event=None):
    for obj in self.select_list:
      if obj.xmin + 4 < obj.xmax:
        obj.xmin += 1
        obj.xmax -= 1
    self.displayBox()

  def onCtrlRight(self, event=None):
    for obj in self.select_list:
      obj.xmin -= 1
      obj.xmax += 1
    self.displayBox()

  ##############################################################################
  # canvas  event
  ##############################################################################
  def onStart(self, event):
    log(3, 'onStart')
    self.start = event

  def onGrow(self, event):
    log(3, 'onGrow')
    canvas = event.widget

    #  before draw object
    del_list = []
    for obj in self.select_list:
      if obj not in self.object_list:
        if obj.objectBox != None:
          canvas.delete(obj.objectBox)
        if obj.objectNm != None:
          canvas.delete(obj.objectNm)
        if obj.objectNmBg != None:
          canvas.delete(obj.objectNmBg)
        del_list.append(obj)
        log(3, 'onGrow del={}/{}/{}'.format(obj.objectBox, obj.objectNm, obj.objectNmBg))
    for o in del_list:
      self.select_list.remove(o)
    # new draw object
    objectBox, objectNm, objectNmBg = self.create_rectangle_tagged(self.toobarObjName(),
                    min(self.start.x, event.x), min(self.start.y, event.y),
                    max(self.start.x, event.x), max(self.start.y, event.y),
                    thickness=1, color='blue')
    x = event.x / self.canvas_img_width * self.org_img_width
    y = event.y / self.canvas_img_height * self.org_img_height
    obj = Obj(None,
          min(self.start.x, event.x) / self.canvas_img_width * self.org_img_width,
          min(self.start.y, event.y) / self.canvas_img_height * self.org_img_height,
          max(self.start.x, event.x) / self.canvas_img_width * self.org_img_width,
          max(self.start.y, event.y) / self.canvas_img_height * self.org_img_height,
          objectBox=objectBox, objectNm=objectNm, objectNmBg=objectNmBg)
    self.select_list.append(obj)
    log(3,
      'onGrow ADD={}/{}/{}, TOT={}/{}'.format(obj.objectBox, obj.objectNm, obj.objectNmBg, len(self.select_list),
                          len(self.object_list)))

  def onSelect(self, event):
    log(2, 'onSelect')

    obj = self.findObj(event)
    if obj != None:
      if obj not in self.select_list:
        self.select_list.append(obj)
      # 현제 Object Name Update
      self.curr_obj_name.set(obj.name)
      self.is_difficult.set(obj.difficult)
      self.is_truncated.set(obj.truncated)
      log(4, "onSelected ..")
      self.displayBox()

  def onUnSelect1(self, event):
    log(2, 'onUnSelect1')

    obj = self.findObj(event, include_temp=True)
    if obj != None:
      del_list = []
      if obj in self.select_list:
        del_list.append(obj)
      for o in del_list:
        self.select_list.remove(o)
      log(4, "onSelected ..")
      self.displayBox()

  def onUnSelectAll(self, event=None):
    log(2, 'onUnSelectAll')
    del_list = []
    for obj in self.select_list:
      del_list.append(obj)

    for o in del_list:
      self.select_list.remove(o)
    log(4, "onUnSelect ..")
    self.displayBox()

  #############################################################################################
  # common function
  #############################################################################################
  def findParentObjectInSelectList(self, obj):
    for parent in self.select_list:
      if parent in self.object_list:
        if parent.xmin <= obj.xmin and parent.ymin <= obj.ymin and obj.xmax <= parent.xmax and obj.ymax <= parent.ymax:
          return parent
    return None

  def toobarObjName(self):
    name = self.curr_obj_name.get()
    if name == None or len(name) == 0:
      name = '?'
    return name.strip()

  def getPartList(self, obj):
    part_list = []
    for o in self.object_list:
      if o.parent == obj:
        part_list.append(o)
    return part_list

  def findObjByRect(self, xmin, ymin, xmax, ymax):
    finds = []
    for obj in self.object_list:
      if xmin <= obj.xmin and obj.xmax <= xmax and ymin <= obj.ymin and obj.ymax <= ymax:
        finds.append(obj)
    return finds

  def findObj(self, event, include_temp=False):
    log(4, "len(self.object_list)=>{}".format(len(self.object_list)))
    log(4, "{},{}".format(event.x, event.y))
    x = event.x / self.canvas_img_width * self.org_img_width
    y = event.y / self.canvas_img_height * self.org_img_height
    finds = []
    for obj in reversed(self.object_list):
      if obj.xmin < x and x < obj.xmax and obj.ymin < y and y < obj.ymax:
        finds.append(obj)
    if include_temp:
      for obj in self.select_list:
        if obj.xmin < x and x < obj.xmax and obj.ymin < y and y < obj.ymax:
          finds.append(obj)
    if len(finds) == 0:
      return None
    finds.sort(key=self.calcSize)
    return finds[0]

  def calcSize(self, obj):
    return abs(obj.xmax - obj.xmin) * abs(obj.ymax - obj.ymin)

  def loadFile(self):
    log(2, 'loadFile start')
    self.loadXml()
    self.loadImage()
    self.displayBox()
    log(2, 'loadFile ok')

  def loadXml(self):
    log(2, 'loadXml start')
    #if self.curr_image_index <= len(self.image_list):
    curr_img_fname = self.image_list[self.curr_image_index]
    self.curr_file_name.set(self.image_list[self.curr_image_index])
    self.root.title(self.Title+":"  + curr_img_fname)
    self.loadVocXml(curr_img_fname.replace(".jpg", ""))
    self.org_img_width, self.org_img_height, self.object_list = self.loadVocXml(curr_img_fname.replace(".jpg", ""))
    log(2, 'loadXml ok self.object_list= {}, org_img_width={}'.format(len(self.object_list), self.org_img_width))

  def loadImage(self):
    log(2, 'loadImage start')
    self.select_list.clear()
    
    #if self.curr_image_index < len(self.image_list):
    curr_img_fname = self.image_list[self.curr_image_index]
    if os.path.isfile(curr_img_fname):
      self.org_img = Image.open(curr_img_fname)
    elif curr_img_fname == ".vocedit.jpg":
      self.org_img = Image.new('RGB', (300, 300), (0, 0, 0))
    self.canvas_img = self.org_img.resize((self.canvas_img_width, self.canvas_img_height))
    self.photo = ImageTk.PhotoImage(self.canvas_img)
    self.canvas.create_image(0, 0, anchor=NW, image=self.photo)
    self.canvas.image = self.photo
    log(2, 'loadImage ok')

  def displayBox(self):
    self.canvas.delete(ALL)
    self.canvas.create_image(0, 0, anchor=NW, image=self.photo)
    for obj in self.object_list:
      thickness = 1
      color = 'white'
      if obj in self.select_list:
        thickness = 4
      if obj.parent != None:
        color = 'light gray'
      #log(1, 'displayBox() {}/{} '.format(self.org_img_width,self.canvas_img_width ))

      objectBox, objectNm, objectNmBg = self.create_rectangle_tagged(
        obj.name,
        obj.xmin / self.org_img_width * self.canvas_img_width,
        obj.ymin / self.org_img_height * self.canvas_img_height,
        obj.xmax / self.org_img_width * self.canvas_img_width,
        obj.ymax / self.org_img_height * self.canvas_img_height,
        thickness=thickness, color=color)
      obj.objectBox = objectBox
      obj.objectNm = objectNm
      obj.objectNmBg = objectNmBg
    log(2, 'displayBox() ok')

  def create_rectangle_tagged(self, name, x1, y1, x2, y2, thickness=2, color='white'):
    objectBox = objectNm = objectNmBg = None
    if self.is_show_label.get() != 0:
      objectNm = self.canvas.create_text(x1+5, y1 + 8, text=name, fill='black')
      objectNmBg = self.canvas.create_rectangle(self.canvas.bbox(objectNm), fill=color)
      self.canvas.tag_lower(objectNmBg, objectNm)
    lcolor ='yellow'
    if name == 'a': lcolor='white'
    if name == 'b': lcolor='light gray'
    if name == 'c': lcolor='yellow'
    if name == 'd': lcolor='orange'
    if name == 'e': lcolor='purple'
    if name == 'f': lcolor='red'
    if name == 'x': lcolor='black'
    objectBox = self.canvas.create_rectangle(x1, y1, x2, y2, width=thickness, outline=lcolor)

    # self.canvas.itemconfig(objectBox, tag=name)
    # self.canvas.itemconfig(objectNm,  tag=name)
    # self.canvas.itemconfig(objectNmBg, tag=name)
    return objectBox, objectNm, objectNmBg

  def getMinMax(self, ctrs):
    r = Rect(9999, 9999, 0, 0)
    for ctr in ctrs:
      x, y, w, h = cv2.boundingRect(ctr)
      r.x = min(r.x, x)
      r.y = min(r.y, y)
      r.w = max(r.w, w + x)
      r.h = max(r.h, h + y)
    return r.x, r.y, r.w, r.h

  def beep1(self, s1=1):
    None
    #print(ws.Beep(2000, 1000))
    #print('\a', end='')
    #sys.stdout.flush()
    #from pygame import mixer  # Load the required library
    #mixer.init()
    #mixer.music.load('mp3/sound{}.mp3'.format(s1))
    #mixer.music.play()

  #############################################################################################
  def saveVocXml(self, path_file_name, width, height, object_list):
    log(2, 'saveVocXml()')
    recalc_width = 1
    recalc_height = 1
    if width == 1 and height == 1:
      recalc_width, recalc_height = self.org_img.size
    max_w = int(width * recalc_width)
    max_h = int(height * recalc_height)

    fname = os.path.basename(path_file_name)
    xml = []
    xml.append("<annotation>")
    xml.append("  <folder>carno</folder>")
    xml.append("  <filename>{}</filename>".format(fname))
    xml.append("  <source>")
    xml.append("    <database>carno</database>")
    xml.append("    <annotation>carno</annotation>")
    xml.append("    <image>flickr</image>")
    xml.append("  </source>")
    xml.append("  <size>")
    xml.append("        <width>{}</width>".format(max_w))
    xml.append("        <height>{}</height>".format(max_h))
    xml.append("    <depth>3</depth>")
    xml.append("  </size>")
    xml.append("  <segmented>0</segmented>")

    log(1, "saveVocXml={}".format(len(object_list)))
    for obj in object_list:
      if obj.parent != None:
        continue
      log(1, "obj={}".format(obj))
      xml.append("  <object>")
      xml.append("    <name>{}</name>".format(obj.name))
      xml.append("    <pose>Unspecified</pose>")
      xml.append("    <truncated>{}</truncated>".format(obj.truncated))
      xml.append("    <difficult>{}</difficult>".format(obj.difficult))
      xml.append("    <bndbox>")
      xml.append("            <xmin>{}</xmin>".format(max(0,       int(obj.xmin * recalc_width))))
      xml.append("            <ymin>{}</ymin>".format(max(0,       int(obj.ymin * recalc_height))))
      xml.append("            <xmax>{}</xmax>".format(min(max_w-1, int(obj.xmax * recalc_width))))
      xml.append("            <ymax>{}</ymax>".format(min(max_h-1, int(obj.ymax * recalc_height))))
      xml.append("    </bndbox>")
      part_list = self.getPartList(obj)
      for sobj in part_list:
        xml.append("    <part>")
        xml.append("      <name>{}</name>".format(sobj.name))
        xml.append("      <bndbox>")
        xml.append("                <xmin>{}</xmin>".format(max(0,       int(sobj.xmin * recalc_width))))
        xml.append("                <ymin>{}</ymin>".format(max(0,       int(sobj.ymin * recalc_height))))
        xml.append("                <xmax>{}</xmax>".format(min(max_w-1, int(sobj.xmax * recalc_width))))
        xml.append("                <ymax>{}</ymax>".format(min(max_h-1, int(sobj.ymax * recalc_height))))
        xml.append("      </bndbox>")
        xml.append("    </part>")
      xml.append("  </object>")
    xml.append("</annotation>")

    f = open(path_file_name.replace(".jpg", ".xml"), "w")
    f.write('\n'.join(xml))
    f.close()
    log(2, 'saveVocXml({} -->())'.format(path_file_name.replace(".jpg", ".xml"), len(object_list)))

  def loadVocXml(self, file_name):
    if not os.path.isfile(file_name + ".xml"):
      return 1, 1, []

    log(2, 'loadVocXml ({})'.format(file_name + ".xml"))
    object_list = []
    tree = xml_tree.parse(file_name + ".xml")
    root = tree.getroot()
    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)]
    # Find annotations.
    for xml_obj in root.findall('object'):
      label = xml_obj.find('name').text
      difficult_val = int(xml_obj.find('difficult').text)
      truncated_val = int(xml_obj.find('truncated').text)
      bbox = xml_obj.find('bndbox')
      obj = Obj(label, float(bbox.find('xmin').text), float(bbox.find('ymin').text),
            float(bbox.find('xmax').text), float(bbox.find('ymax').text),
            truncated=truncated_val, difficult=difficult_val)
      object_list.append(obj)
      for part in xml_obj.findall('part'):
        p_label = part.find('name').text
        p_bbox = part.find('bndbox')
        p_obj = Obj(p_label, float(p_bbox.find('xmin').text), float(p_bbox.find('ymin').text),
              float(p_bbox.find('xmax').text), float(p_bbox.find('ymax').text),
              parent=obj)
        object_list.append(p_obj)

    log(2, object_list)
    return int(size.find('width').text), int(size.find('height').text), object_list

  def parseImageInfo(self, p_img):
    log(2, 'parseImageInfo')
    img = numpy.array(p_img)

    object_list = []
    bin = cv2.split(img)[0]

    #_retval, bin = cv2.threshold(bin, 110, 110, cv2.THRESH_BINARY)
    _retval, bin = cv2.threshold(bin, 110, 110, cv2.THRESH_BINARY)

    bin, contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cont_list = []
    tot_size = 0
    tot_cnt = 0
    for contour in contours:
      cont_len = cv2.arcLength(contour, True)
      cont = cv2.approxPolyDP(contour, 0.001 * cont_len, True)
      if cv2.contourArea(cont) < 1000:
        continue
      if cv2.contourArea(cont) > 800000:
        continue
      cont_list.append(cont)
      if not cv2.isContourConvex(cont):
        continue
      elif 50000 < cv2.contourArea(cont):
        continue
      tot_size += cv2.contourArea(cont)
      tot_cnt += 1

    avg_size = 50000
    if tot_cnt != 0:
      avg_size = tot_size / tot_cnt

    for cont in cont_list:
      c = (0, 0, 0)
      # noise
      if cv2.contourArea(cont) < 1000:
        continue
      elif not cv2.isContourConvex(cont):
        c = (255, 0, 0)
      elif avg_size * 3 > cv2.contourArea(cont):
        c = (255, 0, 255)  # yellow
      else:
        c = (255, 0, 0)
      xmin, ymin, xmax, ymax = self.getMinMax(cont)
      log(3, "parseImageInfo>{},{}-{},{}".format(xmin, ymin, xmax, ymax))
      object_list.append(Obj('b', xmin, ymin, xmax, ymax))
    log(2, 'parseImageInfo ok {}'.format(len(object_list)))
    return len(img[0]), len(img), object_list


if __name__ == '__main__':
    init_k()
    flist = []
    if len(sys.argv) <= 1 :
        flist.append(".vocedit.jpg")
    else:
        for f in sys.argv[1:]:
            if os.path.isfile(f):
                flist.append(f)
            else:
                flist += glob.glob(f)
                
    log(1, flist)
    VocEditor(flist)
    mainloop()
