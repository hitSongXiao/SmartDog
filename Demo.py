# -*-coding:utf-8 -*-
# Author   : zzp
# Date     : 2020/4/28 10:35
# Email AD ：2410520561@qq.com
# SoftWare : PyCharm
# Project Name   : SmartDog_v2
# Python Version : 3.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from PyQt5.QtWidgets import QApplication, QMainWindow
from UI_SmartDog import Ui_Smartdog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2

torch.set_num_threads(1)


class Window(QMainWindow, Ui_Smartdog):
    # init Window class
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

        # pushbutton init
        self.btn_open_camera.setEnabled(True)
        self.btn_open_video.setEnabled(True)
        self.btn_select_target.setEnabled(False)
        self.btn_track_over.setEnabled(False)
        self.btn_track_start.setEnabled(False)

        # slot init
        self.btn_select_target.clicked.connect(self.slot_press_select_roi)
        self.btn_open_camera.clicked.connect(self.slot_press_camera)
        self.btn_open_video.clicked.connect(self.slot_press_video)
        self.btn_track_over.clicked.connect(self.slot_press_over)
        self.btn_track_start.clicked.connect(self.slot_press_track)

        # parameters init
        self.open_keyboard_flag = False
        self.open_camera_flag = False
        self.open_video_flag = False
        self.open_track_flag=False
        self.width=1000
        self.height=600
        self.target_rect = []
        self.video_path = './source/bag.avi'

    # 按下摄像头跟踪按钮
    def slot_press_camera(self):
        # init pushbotton
        self.btn_open_camera.setEnabled(False)
        self.btn_open_video.setEnabled(False)
        self.btn_select_target.setEnabled(True)
        self.btn_track_over.setEnabled(True)
        self.btn_track_start.setEnabled(False)

        # 开始线程
        self.label_show.clear_flag = False
        self.camera = cv2.VideoCapture(0)
        self.open_camera_flag = True

        # camera object init
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.slot_camera_show_image)
        self.camera_timer.start(70)

    # label对象展示摄像头内容
    def slot_camera_show_image(self):
        if self.open_camera_flag is True:
            if self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret is True:
                    # label 控件的size=(900,800)
                    self.frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
            else:
                self.camera_timer.stop()
        else:
            self.camera_timer.stop()

    # 按下视频文件跟踪按钮
    def slot_press_video(self):
        # init pushbotton
        self.btn_open_camera.setEnabled(False)
        self.btn_open_video.setEnabled(False)
        self.btn_select_target.setEnabled(True)
        self.btn_track_over.setEnabled(True)
        self.btn_track_start.setEnabled(False)

        # init 参数
        self.label_show.clear_flag = False
        self.open_video_flag = True

        # 创建视频读取对象
        self.capture_video = cv2.VideoCapture(self.video_path)
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.slot_video_show_image)
        self.video_timer.start(70)

    # label 对象展示视频内容
    def slot_video_show_image(self):
        if self.open_video_flag is True:
            if self.capture_video.isOpened() is True:
                ret, frame = self.capture_video.read()
                if ret is True:
                    # label 控件的size=(900,800)
                    self.frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
            else:
                self.video_timer.stop()
                self.camera.release()
        else:
            self.video_timer.stop()
            self.camera.release()

    # 按下选择跟踪目标按钮
    def slot_press_select_roi(self):
        # init pushbotton
        self.btn_open_camera.setEnabled(False)
        self.btn_open_video.setEnabled(False)
        self.btn_select_target.setEnabled(False)
        self.btn_track_over.setEnabled(True)
        self.btn_track_start.setEnabled(True)

        # 打开事件监控
        self.open_keyboard_flag = True

        # 暂停摄像头、视频线程

        # 如果打开的是视频
        if self.open_video_flag is True:
            self.first_frame = self.frame
            self.video_timer.stop()

        # 如果打开的是摄像头
        if self.open_camera_flag is True:
            self.first_frame = self.frame
            self.camera_timer.stop()


    # 按下开始跟踪按钮
    def slot_press_track(self):
        if self.open_keyboard_flag is False:
            self.btn_open_camera.setEnabled(False)
            self.btn_open_video.setEnabled(False)
            self.btn_select_target.setEnabled(False)
            self.btn_track_over.setEnabled(True)
            self.btn_track_start.setEnabled(False)

            # 跟踪器选择
            self.init_track()
            print('跟踪器初始完毕！')
            # 获取目标位置
            self.target_rect = self.transformrect(self.label_show.rect)
            print(self.target_rect)

            init_rect = tuple(self.target_rect)
            self.tracker.init(self.first_frame, init_rect)

            self.clear_label()  # 清理label

            # 创建跟踪线程
            self.tracker_timer = QTimer(self)
            self.tracker_timer.timeout.connect(self.slot_track_process)
            self.tracker_timer.start(70)
            self.open_select_roi=False

    # 跟踪过程
    def slot_track_process(self):
        self.open_track_flag=True
        # 视频跟踪
        if self.open_video_flag is True:
            if self.capture_video.isOpened() is True:
                ret, frame = self.capture_video.read()
                if ret is True:
                    # 修正图片大小
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

                    outputs = self.tracker.track(frame)
                    if 'polygon' in outputs:
                        polygon = np.array(outputs['polygon']).astype(np.int32)
                        cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                      True, (0, 255, 0), 3)
                        mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                        mask = mask.astype(np.uint8)
                        mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                        frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                    else:
                        bbox = list(map(int, outputs['bbox']))
                        cv2.rectangle(frame, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 0), 3)

                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
            else:
                self.tracker_timer.stop()
        elif self.open_camera_flag is True:
            if self.camera.isOpened() is True:
                ret, frame = self.camera.read()
                if ret is True:
                    # 修正图片大小
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                    outputs = self.tracker.track(frame)
                    if 'polygon' in outputs:
                        polygon = np.array(outputs['polygon']).astype(np.int32)
                        cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                      True, (0, 255, 0), 3)
                        mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                        mask = mask.astype(np.uint8)
                        mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                        frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                    else:
                        bbox = list(map(int, outputs['bbox']))
                        cv2.rectangle(frame, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 0), 3)

                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
            else:
                self.tracker_timer.stop()

    #
    def slot_press_over(self):
        if self.open_camera_flag is True:
            self.camera_timer.stop()
            self.camera.release()
            self.clear_label()  # 清除Label
            self.open_camera_flag = False
            self.open_keyboard_flag = False
            self.btn_open_camera.setEnabled(True)
            self.btn_open_video.setEnabled(True)
            self.btn_select_target.setEnabled(False)
            self.btn_track_start.setEnabled(False)
            if self.open_track_flag is True:
                self.tracker_timer.stop()
                self.camera.release()
                self.open_track_flag=False

        if self.open_video_flag is True:
            self.video_timer.stop()
            self.capture_video.release()
            self.clear_label()  # 清除label
            self.open_video_flag = False
            self.open_keyboard_flag = False
            self.btn_open_camera.setEnabled(True)
            self.btn_open_video.setEnabled(True)
            if self.open_track_flag is True:
                self.tracker_timer.stop()
                self.capture_video.release()
                self.open_track_flag=False
            self.btn_select_target.setEnabled(False)
            self.btn_track_start.setEnabled(False)

    # 捕获键盘事件
    def keyPressEvent(self, QKeyEvent):
        if self.open_keyboard_flag is True:
            if QKeyEvent.key() == Qt.Key_S:
                self.label_show.setCursor(Qt.CrossCursor)
                self.label_show.open_mouse_flag = True
                self.label_show.draw_roi_flag = True
            if QKeyEvent.key() == Qt.Key_Q:
                self.label_show.unsetCursor()
                self.label_show.draw_roi_flag = False
                self.label_show.open_mouse_flag = False
                self.open_keyboard_flag = False
                # print(self.transformrect(self.label_show.rect))

    # QRect 转化为list
    def transformrect(self, roi_rect):
        t_rect = [roi_rect.x(), roi_rect.y(),
                  roi_rect.width(), roi_rect.height()]
        return t_rect

    # 初始化跟踪模型
    def init_track(self):

        # 配置config文件
        config_path = './models/siamrpn_alex_dwxcorr/config.yaml'
        # 配置snapshot 文件
        snapshot_path = './models/siamrpn_alex_dwxcorr/model.pth'

        # 参数整合
        cfg.merge_from_file(config_path)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create model
        model = ModelBuilder()

        # load model
        model.load_state_dict(torch.load(snapshot_path, map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

        # 创建跟踪器
        self.tracker = build_tracker(model)

    def clear_label(self):
        self.label_show.clear_flag = True
        self.label_show.clear()


def main():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
