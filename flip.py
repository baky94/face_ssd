# -*- coding: utf-8 -*-
from __future__ import print_function
import click, os, glob, re, multiprocessing,sys, itertools,cv2,random
import xml.etree.ElementTree as xml_tree

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
    
def loadVocXml(file_name):
    if not os.path.isfile(file_name + ".xml"):
        return 1, 1, []
  
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
  
    return int(size.find('width').text), int(size.find('height').text), object_list
    
  
def saveVocXml(path_file_name, width, height, object_list):  
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
    xml.append("        <width>{}</width>".format(width))
    xml.append("        <height>{}</height>".format(height))
    xml.append("    <depth>3</depth>")
    xml.append("  </size>")
    xml.append("  <segmented>0</segmented>")
  
    for obj in object_list:
        if obj.parent != None:
          continue
        xml.append("  <object>")
        xml.append("    <name>{}</name>".format(obj.name))
        xml.append("    <pose>Unspecified</pose>")
        xml.append("    <truncated>{}</truncated>".format(obj.truncated))
        xml.append("    <difficult>{}</difficult>".format(obj.difficult))
        xml.append("    <bndbox>")
        xml.append("            <xmin>{}</xmin>".format(obj.xmin))
        xml.append("            <ymin>{}</ymin>".format(obj.ymin))
        xml.append("            <xmax>{}</xmax>".format(obj.xmax))
        xml.append("            <ymax>{}</ymax>".format(obj.ymax))
        xml.append("    </bndbox>")
        part_list = getPartList(obj, object_list)
        for sobj in part_list:
            xml.append("    <part>")
            xml.append("      <name>{}</name>".format(sobj.name))
            xml.append("      <bndbox>")
            xml.append("                <xmin>{}</xmin>".format(sobj.xmin))
            xml.append("                <ymin>{}</ymin>".format(sobj.ymin))
            xml.append("                <xmax>{}</xmax>".format(sobj.xmax))
            xml.append("                <ymax>{}</ymax>".format(sobj.ymax))
            xml.append("      </bndbox>")
            xml.append("    </part>")
        xml.append("  </object>")
    xml.append("</annotation>")
  
    f = open(path_file_name.replace(".jpg", ".xml"), "w")
    f.write('\n'.join(xml))
    f.close()

def getPartList(obj, object_list):
    part_list = []
    for o in object_list:
        if o.parent == obj:
            part_list.append(o)
    return part_list
if __name__ == "__main__" and len(sys.argv) > 1:
    flist = []
    for f in sys.argv[1:]:
        print(f)
        if os.path.isfile(f):
            flist.append(f)
        else:
            flist += glob.glob(f)
    print(flist)
    for fname in flist:
        print(fname)
        if fname.find('.jpg')  >= 0:
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
            img =  cv2.flip(img, 1)
            cv2.imwrite(fname.replace('.jpg', '_R.jpg'), img)

        elif fname.find('.xml') >= 0 :
            fname = fname.replace(".xml", "")
            width, height, object_list = loadVocXml(fname)
            for obj in object_list:
                o1 = Obj(obj.name, obj.xmin, obj.ymin, obj.xmax, obj.ymax) 
                obj.xmax = width - o1.xmin
                obj.xmin = width - o1.xmax
                part_list = getPartList(obj, object_list)
                for sobj in part_list:
                  o1 = Obj(sobj.name, sobj.xmin, sobj.ymin, sobj.xmax, sobj.ymax) 
                  sobj.xmax = width - o1.xmin
                  sobj.xmin = width - o1.xmax
            saveVocXml(fname + '_R.jpg', width, height, object_list)
