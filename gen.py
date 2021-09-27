import time
import xml.etree.ElementTree as xml_tree
import os.path
import cv2
import numpy
import time
from datetime import datetime
import sys

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

class Genxml:
    def saveVocXml(self, path_file_name, width, height, object_list):
        fname = os.path.basename(path_file_name)
        xml = []
        xml.append("<annotation>")
        xml.append("    <folder>face</folder>")
        xml.append("    <filename>{}</filename>".format(fname))
        xml.append("    <source>")
        xml.append("        <database>carno</database>")
        xml.append("        <annotation>carno</annotation>")
        xml.append("    </source>")
        xml.append("    <size>")
        xml.append("        <width>{}</width>".format(int(width)))
        xml.append("        <height>{}</height>".format(int(height)))
        xml.append("        <depth>3</depth>")
        xml.append("    </size>")
        xml.append("    <segmented>0</segmented>")

        for obj in object_list:
            if obj.parent != None:
                continue
            xml.append("    <object>")
            xml.append("        <name>{}</name>".format(obj.name))
            xml.append("        <pose>Unspecified</pose>")
            xml.append("        <truncated>{}</truncated>".format(obj.truncated))
            xml.append("        <difficult>{}</difficult>".format(obj.difficult))
            xml.append("        <bndbox>")
            xml.append("            <xmin>{}</xmin>".format((obj.xmin)))
            xml.append("            <ymin>{}</ymin>".format((obj.ymin)))
            xml.append("            <xmax>{}</xmax>".format((obj.xmax)))
            xml.append("            <ymax>{}</ymax>".format((obj.ymax)))
            xml.append("        </bndbox>")
            '''
            part_list = self.getPartList(obj)
            for sobj in part_list:
                xml.append("        <part>")
                xml.append("            <name>{}</name>".format(sobj.name))
                xml.append("            <bndbox>")
                xml.append("                <xmin>{}</xmin>".format((sobj.xmin)))
                xml.append("                <ymin>{}</ymin>".format((sobj.ymin)))
                xml.append("                <xmax>{}</xmax>".format((sobj.xmax)))
                xml.append("                <ymax>{}</ymax>".format((sobj.ymax)))
                xml.append("            </bndbox>")
                xml.append("        </part>")
            '''
            xml.append("    </object>")
        xml.append("</annotation>")

        f = open(path_file_name.replace(".jpg", ".xml"), "w")
        f.write('\n'.join(xml))
        f.close()
        
    def loadLabel(self, path_file_name):
        lines = []
        
        with open(path_file_name) as f:
            lines = f.read().splitlines()

        label = 0
        if path_file_name.find("c1") >= 0:
            label = "1"
        if path_file_name.find("c2") >= 0:
            label = "2"
        xmin = ymin = 99999
        xmax = ymax = -1
        
        line_no = 0
        for line in lines:
            if  line.find(label) >= 0:
                l = int(line.index(label) / 2)
                r = int(line.rindex(label) / 2)
                if l >= 0 and l < xmin:
                    xmin = l
                if r >= 0 and r > xmax:
                    xmax = r
                if line_no < ymin:
                    ymin = line_no
                if line_no> ymax:
                    ymax = line_no
            line_no += 1
        if  xmin <  xmax : 
            print(xmin, ymin, xmax, ymax)

            obj = Obj(label, xmin, ymin, xmax, ymax)
            self.saveVocXml(path_file_name.replace('.txt', '.jpg'), int(len(lines[0])/2), len(lines), [obj])
        else :
            print("f{path_file_name} 에 Label이 없습니다.")

gen = Genxml()  
for fname in sys.argv[1:]:
    gen.loadLabel(fname)
