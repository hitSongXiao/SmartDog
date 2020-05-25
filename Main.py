# -*-coding:utf-8 -*-
# Author   : zzp
# Date     : 2020/4/28 0:53
# Email AD ：2410520561@qq.com
# SoftWare : PyCharm
# Project Name   : SmartDog_v2
# Python Version : 3.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

from PyQt5.QtWidgets import QMainWindow,QFileDialog
from UI_SmartDog import Ui_Smartdog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import gc

# 创建窗口主类
class Window(QMainWindow,Ui_Smartdog):

    # 初始化窗口类
    def __init__(self):
        super(Window,self).__init__()
        self.setupUi(self)

        #***************************标志参数*****************************#
        self.open_keyboard_flag = False      #判断是否打开键盘事件监控
        self.open_camera_flag = False        #判断是使用摄像头进行跟踪
        self.open_video_flag = False         #判断是使用视频文件进行跟踪
        self.open_track_flag=False           #判断跟踪是否确认
        self.width=1000                      #显示label的宽度
        self.height=600                      #显示label的高度
        self.target_rect = []                #跟踪目标框
        self.video_path = ''                 #选择的视频文件路径
        self.FPS=40
        self.open_algo_flag=False           #判断是否选了算法
        self.open_select_algo_flag=False    #判断算法选择是否确认
        # ***************************标志参数*****************************#

        # ***************************按键初始化*****************************#
        self.btn_open_camera.setEnabled(True)         #摄像头打开按钮->打开
        self.btn_open_video.setEnabled(True)          #视屏文件打开按钮->打开
        self.btn_select_target.setEnabled(False)      #选择跟踪目标区域->关闭
        self.btn_track_over.setEnabled(False)         #结束跟踪->关闭
        self.btn_track_start.setEnabled(False)        #开始跟踪->关闭
        self.btn_algo_ok.setEnabled(True)             #算法确认按钮->打开
        self.btn_algo_ok.setEnabled(True)
        # self.checkBox_Siammask.setChecked(True)
        # ***************************按键初始化*****************************#

        # ***************************槽函数初始化*****************************#
        self.btn_select_target.clicked.connect(self.slot_press_select_roi)
        self.btn_open_camera.clicked.connect(self.slot_press_camera)
        self.btn_open_video.clicked.connect(self.slot_press_video)
        self.btn_track_over.clicked.connect(self.slot_press_over)
        self.btn_track_start.clicked.connect(self.slot_press_track)
        self.btn_algo_ok.clicked.connect(self.slot_config_model)
        self.checkBox_Siammask.clicked.connect(self.slot_press_algomask)
        self.checkBox_SiamrpnA1.clicked.connect(self.slot_press_algoA1)
        self.checkBox_SiamrpnA2.clicked.connect(self.slot_press_algoA2)
        self.checkBox_SiamrpnM.clicked.connect(self.slot_press_algoM)
        self.checkBox_SiamrpnR1.clicked.connect(self.slot_press_algoR1)
        self.checkBox_SiamrpnR2.clicked.connect(self.slot_press_algoR2)
        self.checkBox_SiamrpnR3.clicked.connect(self.slot_press_algoR3)
        # ***************************槽函数初始化*****************************#

        self.textBws_show_process.append('程序初始化完毕!')
        self.textBws_show_process.append('请您选择跟踪模型...')


#############################################按键槽函数##############################################
    # 按下选择算法1
    def slot_press_algomask(self):
        self.open_algo_flag=True
        self.checkBox_SiamrpnA1.setChecked(False)
        self.checkBox_SiamrpnA2.setChecked(False)
        self.checkBox_SiamrpnM.setChecked(False)
        self.checkBox_SiamrpnR1.setChecked(False)
        self.checkBox_SiamrpnR2.setChecked(False)
        self.checkBox_SiamrpnR3.setChecked(False)
        self.config_path = './models/siammask_r50_l3/config.yaml'           # 配置config文件
        self.snapshot_path = './models/siammask_r50_l3/model.pth'           # 配置snapshot 文件
        if self.checkBox_Siammask.isChecked() is True:
            self.textBws_show_process.append('已选择SiamMask_r50_l3模型!')
        else:
            self.textBws_show_process.append('已取消!')

    # 按下选择算法2
    def slot_press_algoA1(self):
        self.open_algo_flag = True
        self.checkBox_Siammask.setChecked(False)
        self.checkBox_SiamrpnA2.setChecked(False)
        self.checkBox_SiamrpnM.setChecked(False)
        self.checkBox_SiamrpnR1.setChecked(False)
        self.checkBox_SiamrpnR2.setChecked(False)
        self.checkBox_SiamrpnR3.setChecked(False)
        self.config_path = './models/siamrpn_alex_dwxcorr/config.yaml'  # 配置config文件
        self.snapshot_path = 'models/siamrpn_alex_dwxcorr/model.pth'  # 配置snapshot 文件
        if self.checkBox_SiamrpnA1.isChecked() is True:
            self.textBws_show_process.append('已选择siamrpn_alex_dwxcorr模型!')
        else:
            self.textBws_show_process.append('已取消!')


    # 按下选择算法3
    def slot_press_algoA2(self):
        self.open_algo_flag = True
        self.checkBox_Siammask.setChecked(False)
        self.checkBox_SiamrpnA1.setChecked(False)
        self.checkBox_SiamrpnM.setChecked(False)
        self.checkBox_SiamrpnR1.setChecked(False)
        self.checkBox_SiamrpnR2.setChecked(False)
        self.checkBox_SiamrpnR3.setChecked(False)
        self.config_path = './models/siamrpn_alex_dwxcorr_otb/config.yaml'  # 配置config文件
        self.snapshot_path = 'models/siamrpn_alex_dwxcorr_otb/model.pth'  # 配置snapshot 文件
        if self.checkBox_SiamrpnA2.isChecked() is True:
            self.textBws_show_process.append('已选择siamrpn_alex_dwxcorr_otb模型!')
        else:
            self.textBws_show_process.append('已取消!')


    # 按下选择算法4
    def slot_press_algoM(self):
        self.open_algo_flag = True
        self.checkBox_Siammask.setChecked(False)
        self.checkBox_SiamrpnA1.setChecked(False)
        self.checkBox_SiamrpnA2.setChecked(False)
        self.checkBox_SiamrpnR1.setChecked(False)
        self.checkBox_SiamrpnR2.setChecked(False)
        self.checkBox_SiamrpnR3.setChecked(False)
        self.config_path = './models/siamrpn_mobilev2_l234_dwxcorr/config.yaml'  # 配置config文件
        self.snapshot_path = 'models/siamrpn_mobilev2_l234_dwxcorr/model.pth'  # 配置snapshot 文件
        if self.checkBox_SiamrpnM.isChecked() is True:
            self.textBws_show_process.append('已选择siamrpn_mobilev2_l234_dwxcorr模型!')
        else:
            self.textBws_show_process.append('已取消!')


    # 按下选择算法5
    def slot_press_algoR1(self):
        self.open_algo_flag = True
        self.checkBox_Siammask.setChecked(False)
        self.checkBox_SiamrpnA1.setChecked(False)
        self.checkBox_SiamrpnA2.setChecked(False)
        self.checkBox_SiamrpnM.setChecked(False)
        self.checkBox_SiamrpnR2.setChecked(False)
        self.checkBox_SiamrpnR3.setChecked(False)
        self.config_path = './models/siamrpn_r50_l234_dwxcorr/config.yaml'  # 配置config文件
        self.snapshot_path = 'models/siamrpn_r50_l234_dwxcorr/model.pth'  # 配置snapshot 文件
        if self.checkBox_SiamrpnR1.isChecked() is True:
            self.textBws_show_process.append('已选择siamrpn_r50_l234_dwxcorr模型!')
        else:
            self.textBws_show_process.append('已取消!')


    # 按下选择算法6
    def slot_press_algoR2(self):
        self.open_algo_flag = True
        self.checkBox_Siammask.setChecked(False)
        self.checkBox_SiamrpnA1.setChecked(False)
        self.checkBox_SiamrpnA2.setChecked(False)
        self.checkBox_SiamrpnM.setChecked(False)
        self.checkBox_SiamrpnR1.setChecked(False)
        self.checkBox_SiamrpnR3.setChecked(False)
        self.config_path = './models/siamrpn_r50_l234_dwxcorr_lt/config.yaml'  # 配置config文件
        self.snapshot_path = 'models/siamrpn_r50_l234_dwxcorr_lt/model.pth'  # 配置snapshot 文件
        if self.checkBox_SiamrpnR2.isChecked() is True:
            self.textBws_show_process.append('已选择siamrpn_r50_l234_dwxcorr模型!')
        else:
            self.textBws_show_process.append('已取消!')

    # 按下选择算法7
    def slot_press_algoR3(self):
        self.open_algo_flag = True
        self.checkBox_Siammask.setChecked(False)
        self.checkBox_SiamrpnA1.setChecked(False)
        self.checkBox_SiamrpnA2.setChecked(False)
        self.checkBox_SiamrpnM.setChecked(False)
        self.checkBox_SiamrpnR1.setChecked(False)
        self.checkBox_SiamrpnR2.setChecked(False)
        self.config_path = './models/siamrpn_r50_l234_dwxcorr_otb/config.yaml'  # 配置config文件
        self.snapshot_path = 'models/siamrpn_r50_l234_dwxcorr_otb/model.pth'  # 配置snapshot 文件
        if self.checkBox_SiamrpnR3.isChecked() is True:
            self.textBws_show_process.append('已选择siamrpn_r50_l234_dwxcorr模型!')
        else:
            self.textBws_show_process.append('已取消!')

    # 按下摄像头跟踪按钮
    def slot_press_camera(self):
        self.textBws_show_process.append('摄像头打开中...')
        # init pushbotton
        self.btn_open_camera.setEnabled(False)
        self.btn_open_video.setEnabled(False)
        self.btn_select_target.setEnabled(True)
        self.btn_track_over.setEnabled(True)
        self.btn_track_start.setEnabled(False)

        # 开始线程
        self.label_show.clear_flag = False
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened:
            self.textBws_show_process.append('摄像头已打开!')
        self.open_camera_flag = True
        # camera object init
        self.camera_timer = QTimer(self)
        self.textBws_show_process.append('线程已创建...')
        self.camera_timer.timeout.connect(self.slot_camera_show_image)
        self.textBws_show_process.append('摄像头拍摄中...')
        self.camera_timer.start(self.FPS)

    # 按下视频文件跟踪按钮
    def slot_press_video(self):
        #弹出视频文件提示框
        video_path=QFileDialog.getOpenFileName(self,
                                                    '请选择您要打开的视频文件',
                                                    'C:\\Users\\Administrator\\video',
                                                    '*.mp4 *.avi')
        self.video_path=video_path[0]
        self.textBws_show_process.append('要加载的视频文件路径:'+self.video_path)
        if self.video_path is None:
            self.textBws_show_process.append('视频文件路径错误!')
            self.video_path='./Source/bag.avi'
        else:
            # init 参数
            self.label_show.clear_flag = False
            self.open_video_flag = True
            # init pushbutton
            self.btn_open_camera.setEnabled(False)
            self.btn_open_video.setEnabled(False)
            self.btn_select_target.setEnabled(True)
            self.btn_track_over.setEnabled(True)
            self.btn_track_start.setEnabled(False)
            # 创建视频读取对象
            self.capture_video = cv2.VideoCapture(self.video_path)
            self.video_timer = QTimer(self)
            self.textBws_show_process.append('线程对象已经创建...')
            self.video_timer.timeout.connect(self.slot_video_show_image)
            self.textBws_show_process.append('视频播放中...')
            self.video_timer.start(self.FPS)

    # 按下选择跟踪目标按钮
    def slot_press_select_roi(self):
        # init pushbotton
        self.btn_open_camera.setEnabled(False)
        self.btn_open_video.setEnabled(False)
        self.btn_select_target.setEnabled(False)
        self.btn_track_over.setEnabled(True)
        self.btn_track_start.setEnabled(True)
        self.open_keyboard_flag = True              # 打开事件监控
        # 如果打开的是视频
        if self.open_video_flag is True:
            self.first_frame = self.frame
            self.video_timer.stop()
        # 如果打开的是摄像头
        if self.open_camera_flag is True:
            self.first_frame = self.frame
            self.camera_timer.stop()
        self.textBws_show_process.append('请选择您的跟踪目标: 按下s键选择，q键确认')

    # 按下开始跟踪按钮
    def slot_press_track(self):
        if self.open_keyboard_flag is False:
            self.btn_open_camera.setEnabled(False)
            self.btn_open_video.setEnabled(False)
            self.btn_select_target.setEnabled(False)
            self.btn_track_over.setEnabled(True)
            self.btn_track_start.setEnabled(False)

            self.textBws_show_process.append('模型加载完毕!')
            # 获取目标位置
            self.target_rect = self.transformrect(self.label_show.rect)
            self.textBws_show_process.append('目标框格式转码中...')
            init_rect = tuple(self.target_rect)
            self.tracker.init(self.first_frame, init_rect)
            self.textBws_show_process.append('目标框初始化完毕!')
            self.clear_label()  # 清理label
            self.textBws_show_process.append('跟踪界面清除...')

            # 创建跟踪线程
            self.tracker_timer = QTimer(self)
            self.textBws_show_process.append('跟踪线程已创建!')
            self.tracker_timer.timeout.connect(self.slot_track_process)
            self.tracker_timer.start(self.FPS)
            self.open_select_roi=False
            self.textBws_show_process.append('目标已经选定！')

    # 按下结束跟踪按钮
    def slot_press_over(self):
        if self.open_camera_flag is True:
            self.camera_timer.stop()
            self.camera.release()
            self.clear_label()  # 清除Label
            self.open_keyboard_flag = False
            self.btn_open_camera.setEnabled(True)
            self.btn_open_video.setEnabled(True)
            self.btn_select_target.setEnabled(False)
            self.btn_track_start.setEnabled(False)
            self.textBws_show_process.append('模型释放完毕！')
            if self.open_track_flag is True:
                self.tracker_timer.stop()
                self.camera.release()
                self.open_track_flag=False
            self.open_camera_flag = False
        if self.open_video_flag is True:
            self.video_timer.stop()
            self.capture_video.release()
            self.clear_label()  # 清除label
            self.open_keyboard_flag = False
            self.btn_open_camera.setEnabled(True)
            self.btn_open_video.setEnabled(True)
            self.btn_select_target.setEnabled(False)
            self.btn_track_start.setEnabled(False)
            self.textBws_show_process.append('模型释放完毕！')
            if self.open_track_flag is True:
                self.tracker_timer.stop()
                self.capture_video.release()
                self.open_track_flag=False
            self.open_video_flag = False
        self.textBws_show_process.append('跟踪线程已结束!')
        self.textBws_show_process.append('跟踪界面已清理完毕!')
        self.textBws_show_process.append('跟踪结束!')

#############################################按键槽函数##############################################


#############################################执行槽函数##############################################
    def slot_config_model(self):
        if self.open_algo_flag is True:
            self.init_track()
            self.open_select_algo_flag =True
            self.open_algo_flag=False
            self.btn_algo_ok.setEnabled(False)


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
                self.camera.release()
                self.textBws_show_process.append('跟踪异常!')
        else:
            self.camera_timer.stop()
            self.camera.release()
            self.textBws_show_process.append('跟踪异常!')

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
                self.capture_video.release()
                self.textBws_show_process.append('跟踪异常!')

        else:
            self.video_timer.stop()
            self.capture_video.release()
            self.textBws_show_process.append('跟踪异常!')

    # 跟踪过程
    def slot_track_process(self):
        self.open_track_flag=True
        if self.open_select_algo_flag is True:
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
                    self.camera.release()
                    self.textBws_show_process.append('跟踪异常!')
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
                    self.capture_video.release()
                    self.textBws_show_process.append('跟踪异常!')
#############################################执行槽函数##############################################


#############################################其他函数##############################################
    # 初始化跟踪模型
    def init_track(self):
        # 参数整合
        cfg.merge_from_file(self.config_path)
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
        device = torch.device('cuda' if cfg.CUDA else 'cpu')

        # create model
        self.textBws_show_process.append('模型对象创建...')
        self.checkpoint=torch.load(self.snapshot_path, map_location=lambda storage, loc: storage.cpu())

        self.model = ModelBuilder()
        print('断点')
        # load model
        self.model.load_state_dict(self.checkpoint)

        self.model.eval().to(device)
        self.textBws_show_process.append('加载跟踪模型完毕!')
        # 创建跟踪器
        self.tracker = build_tracker(self.model)


    # 清除label对象的绘制内容
    def clear_label(self):
        self.label_show.clear_flag = True
        self.label_show.clear()

    # QRect转化为list
    def transformrect(self, roi_rect):
        t_rect = [roi_rect.x(), roi_rect.y(),
                  roi_rect.width(), roi_rect.height()]
        return t_rect

    # 重写键盘事件
    def keyPressEvent(self, QKeyEvent):
        if self.open_keyboard_flag is True:                  # 当键盘事件为真的是才有键盘事件监控
            if QKeyEvent.key() == Qt.Key_S:
                self.label_show.setCursor(Qt.CrossCursor)    # 切换游标为十字型
                self.label_show.open_mouse_flag = True
                self.label_show.draw_roi_flag = True
            if QKeyEvent.key() == Qt.Key_Q:                  # 按下'q'键键盘监控关闭
                self.label_show.unsetCursor()
                self.label_show.draw_roi_flag = False
                self.label_show.open_mouse_flag = False
                self.open_keyboard_flag = False
#############################################其他函数##############################################

























