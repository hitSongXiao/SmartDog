# -*-coding:utf-8 -*-
# Author   : zzp
# Date     : 2020/4/28 11:40
# Email AD ：2410520561@qq.com
# SoftWare : PyCharm
# Project Name   : SmartDog_v2
# Python Version : 3.7

from Main import Window
import sys
from PyQt5.QtWidgets import QApplication

def main():
    app=QApplication(sys.argv)
    window=Window()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()