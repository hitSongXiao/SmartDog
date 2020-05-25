# -*-coding:utf-8 -*-
# Author   : zzp
# Date     : 2020/4/27 8:32
# Email AD ：2410520561@qq.com
# SoftWare : PyCharm
# Project Name   : SmartGog
# Python Version : 3.7
# This python script goal: overwrite the class QLabel in PyQt5

from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt,QRect
from PyQt5.QtGui import QPainter,QPen

class Label(QLabel):
    x0=0
    y0=0
    x1=0
    y1=0
    open_mouse_flag=False
    select_roi_flag=False
    draw_roi_flag=False
    clear_flag=False
    rect = QRect()

    #按下鼠标
    def mousePressEvent(self, event):
        if self.open_mouse_flag is True:
            self.select_roi_flag=True
            self.x0=event.x()
            self.y0=event.y()

    #释放鼠标
    def mouseReleaseEvent(self, event):
        self.select_roi_flag=False

    #移动鼠标
    def mouseMoveEvent(self, event):
        if self.select_roi_flag is True:
            self.x1=event.x()
            self.y1=event.y()
            if self.draw_roi_flag is True:
                self.update()

    #绘制事件
    def paintEvent(self,event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 5, Qt.SolidLine))
        if self.clear_flag is True:
            self.x0=0
            self.y0=0
            self.x1=0
            self.y1=0
        self.rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        painter.drawRect(self.rect)
        self.update()





