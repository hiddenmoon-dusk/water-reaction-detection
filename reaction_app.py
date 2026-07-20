"""
反应管检测与分类系统 — PyQt5 桌面应用
启动: python reaction_app.py
"""
import sys, os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.FATAL)
except: pass

class _NoisyFilter:
    _k = ["oneDNN","GPU support","absl","compiled metrics","This TensorFlow","To enable"]
    def __init__(self,o): self._o=o
    def write(self,s):
        if not any(k in s for k in self._k): self._o.write(s)
    def flush(self): self._o.flush()
if not getattr(sys,'frozen',False) and hasattr(sys.stderr,'write'):
    sys.stderr = _NoisyFilter(sys.stderr)

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
import cv2, numpy as np, tensorflow as tf
from ultralytics import YOLO
from pathlib import Path
from client_config import app_data_root, load_release, model_paths, result_root
from result_storage import save_result
from upload_queue import UploadQueue, UploadTask
from upload_worker import ClientApi, UploadWorker

CLASS_NAMES = {0:("未反应",QColor(220,60,60)), 1:("已反应",QColor(60,180,60))}

WATER_TYPES = ["污水", "生活用水", "养殖水体"]

ORBITAL_STYLE = """
QMainWindow,QWidget{background:#ffffff;color:#111111;font-family:"Segoe UI","Microsoft YaHei UI";font-size:14px}
QGroupBox{border:1px solid #111111;border-radius:0;margin-top:16px;padding:18px 14px 14px 14px;font-weight:600;color:#111111}
QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 6px;background:#ffffff;color:#00a7b5}
QPushButton{background:#ffffff;border:1px solid #111111;border-radius:0;padding:9px 16px;color:#111111;font-weight:600;min-height:20px}
QPushButton:hover{background:#d9ff3f;border-color:#111111;color:#111111}
QPushButton:pressed{background:#00a7b5;border-color:#111111;color:#ffffff}
QPushButton:disabled{color:#8b918c;border-color:#d7dad5;background:#f2f2ee}
QPushButton#waterBtn{background:#f2f2ee;border:1px solid #111111;border-radius:0;padding:24px 32px;font-size:20px;min-height:80px}
QPushButton#waterBtn:hover{background:#d9ff3f;border-color:#111111;color:#111111}
QRadioButton{spacing:8px;color:#111111}
QRadioButton::indicator{width:16px;height:16px;border-radius:0;border:1px solid #111111;background:#ffffff}
QRadioButton::indicator:checked{background:#d9ff3f;border:3px solid #111111}
QComboBox{background:#ffffff;border:1px solid #111111;border-radius:0;padding:7px 12px;color:#111111;min-width:140px}
QComboBox:hover{border:2px solid #00a7b5}
QComboBox QAbstractItemView{background:#ffffff;color:#111111;selection-background-color:#d9ff3f;selection-color:#111111}
QScrollArea{border:none;background:#f2f2ee}
QStatusBar{background:#111111;color:#ffffff;border-top:3px solid #d9ff3f;padding:4px}
QSplitter::handle{background:#111111;width:1px}
QLabel#imageLabel{background:#f2f2ee;border:1px solid #111111;border-radius:0}
QLabel#title{font-size:32px;font-weight:700;color:#111111}
QLabel#subtitle{font-size:14px;color:#59605b}
QLabel#eyebrow{font-size:10px;letter-spacing:2px;color:#00a7b5;font-weight:700}
QFrame[frameShape="5"]{border:1px solid #d7dad5;background:#ffffff}
"""


class UploadStatusBridge(QObject):
    statusChanged = pyqtSignal(str, int)

    @pyqtSlot(str, int)
    def publish(self, status, pending):
        self.statusChanged.emit(status, pending)


class AnalysisWorker(QThread):
    resultReady = pyqtSignal(list)
    statusMsg = pyqtSignal(str)

    def __init__(self, detector, classifier, img_size, img_rgb, mode, conf, cls_thresh):
        super().__init__()
        self.detector, self.classifier = detector, classifier
        self.img_size, self.img_rgb = img_size, img_rgb
        self.mode, self.conf, self.cls_thresh = mode, conf, cls_thresh
        self.manual_regions = []

    def run(self):
        h, w = self.img_rgb.shape[:2]; results = []
        if self.mode == "manual_regions": regions = self.manual_regions
        elif self.mode == "scan":
            self.statusMsg.emit("精细扫描中..."); regions = self._tiled_detect()
        else:
            self.statusMsg.emit("检测中..."); regions = self._full_detect()
        for (x1,y1,x2,y2) in regions:
            x1,y1,x2,y2 = int(max(0,x1)),int(max(0,y1)),int(min(w,x2)),int(min(h,y2))
            if x2-x1<5 or y2-y1<5: continue
            crop = self.img_rgb[y1:y2,x1:x2]
            if crop.size==0: continue
            resized = cv2.resize(crop, self.img_size)
            pred = self.classifier.predict(np.expand_dims(resized,0),verbose=0)[0][0]
            pc = 1 if pred > self.cls_thresh else 0
            label, _ = CLASS_NAMES[pc]
            conf = float(pred if pc==1 else 1-pred)
            results.append((x1,y1,x2,y2,label,conf))
        self.resultReady.emit(results)

    def _full_detect(self):
        r = self.detector(self.img_rgb, conf=self.conf, verbose=False); boxes = []
        for res in r:
            if res.boxes is None: continue
            for b in res.boxes: boxes.append(tuple(b.xyxy[0].cpu().numpy()))
        return boxes

    def _tiled_detect(self):
        h,w = self.img_rgb.shape[:2]; ts,ov = 640,0.2
        if h<=ts and w<=ts: return self._full_detect()
        sh,sw = int(ts*(1-ov)),int(ts*(1-ov))
        rows = max(1,(h-ts)//sh+2); cols = max(1,(w-ts)//sw+2)
        all_boxes = []
        for row in range(rows):
            for col in range(cols):
                x = min(col*sw,w-ts); y = min(row*sh,h-ts)
                x2 = min(x+ts,w); y2 = min(y+ts,h)
                x = max(0,x2-ts); y = max(0,y2-ts)
                tile = self.img_rgb[y:y2,x:x2]
                res = self.detector(tile, conf=self.conf, verbose=False)
                for r in res:
                    if r.boxes is None: continue
                    for b in r.boxes:
                        bx1,by1,bx2,by2 = b.xyxy[0].cpu().numpy()
                        all_boxes.append((bx1+x,by1+y,bx2+x,by2+y))
        if len(all_boxes)<2: return all_boxes
        return all_boxes


class ImageView(QLabel):
    regionSelected = pyqtSignal(int,int,int,int)
    regionDeleted = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter); self.setMouseTracking(True)
        self.setObjectName("imageLabel")
        self.setMinimumSize(400,300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_pixmap = None; self.scaled_pixmap = None; self.scale = 1.0
        self.annotations = []
        self.manual_mode = False; self.drawing = False
        self.draw_start = None; self.draw_current = None
        self.motion_kind = None
        self.motion_progress = 0.0
        self.motion_timer = QTimer(self)
        self.motion_timer.setInterval(33)
        self.motion_timer.timeout.connect(self._advance_motion)

    def setImage(self, img_rgb):
        h,w,c = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, w*3, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qimg)
        self.annotations = []; self.fitToWindow()

    def fitToWindow(self):
        if self.original_pixmap is None: return
        scaled = self.original_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scaled_pixmap = scaled
        self.scale = self.original_pixmap.width()/scaled.width()
        self.setPixmap(scaled)

    def resizeEvent(self, e):
        self.fitToWindow(); super().resizeEvent(e)

    def setAnnotations(self, anns):
        self.annotations = anns; self.update()

    def startScanMotion(self):
        self.motion_kind = "scan"
        self.motion_progress = 0.0
        self.motion_timer.start()
        self.update()

    def startResultMotion(self):
        self.motion_kind = "result"
        self.motion_progress = 0.0
        self.motion_timer.start()
        self.update()

    def stopMotion(self):
        self.motion_timer.stop()
        self.motion_kind = None
        self.motion_progress = 1.0
        self.update()

    def _advance_motion(self):
        step = 0.05 if self.motion_kind == "scan" else 0.10
        self.motion_progress = min(1.0, self.motion_progress + step)
        if self.motion_progress >= 1.0:
            self.stopMotion()
        else:
            self.update()

    def widgetToImage(self, wx, wy):
        if self.scaled_pixmap is None: return 0,0
        pw,ph = self.scaled_pixmap.width(),self.scaled_pixmap.height()
        lw,lh = self.width(),self.height()
        ox,oy = (lw-pw)//2,(lh-ph)//2
        return int((wx-ox)*self.scale), int((wy-oy)*self.scale)

    def imageToWidget(self, ix, iy):
        if self.scaled_pixmap is None: return 0,0
        pw,ph = self.scaled_pixmap.width(),self.scaled_pixmap.height()
        lw,lh = self.width(),self.height()
        ox,oy = (lw-pw)//2,(lh-ph)//2
        return int(ix/self.scale+ox), int(iy/self.scale+oy)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.scaled_pixmap is None: return
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        pw,ph = self.scaled_pixmap.width(),self.scaled_pixmap.height()
        lw,lh = self.width(),self.height()
        ox,oy = (lw-pw)//2,(lh-ph)//2
        painter.setPen(QPen(QColor(17,17,17,28), 1))
        for x in range(ox, ox + pw + 1, 32):
            painter.drawLine(x, oy, x, oy + ph)
        for y in range(oy, oy + ph + 1, 32):
            painter.drawLine(ox, y, ox + pw, y)
        if self.motion_kind == "scan":
            scan_y = int(oy + ph * self.motion_progress)
            painter.setPen(QPen(QColor("#00a7b5"), 2))
            painter.drawLine(ox, scan_y, ox + pw, scan_y)
        painter.setOpacity(self.motion_progress if self.motion_kind == "result" else 1.0)
        for ann in self.annotations:
            x1,y1,x2,y2,label,conf,color = ann
            wx1 = int(x1/self.scale+ox); wy1 = int(y1/self.scale+oy)
            wx2 = int(x2/self.scale+ox); wy2 = int(y2/self.scale+oy)
            painter.setPen(QPen(color,3)); painter.drawRect(wx1,wy1,wx2-wx1,wy2-wy1)
            text = f"{label} {conf:.1%}"
            font = QFont("Microsoft YaHei",11,QFont.Bold); painter.setFont(font)
            fm = painter.fontMetrics(); tw = fm.horizontalAdvance(text); th = fm.height()
            painter.fillRect(wx1,wy1-th-6,tw+8,th+6,color)
            painter.setPen(QColor(255,255,255)); painter.drawText(wx1+4,wy1-6,text)
        painter.setOpacity(1.0)
        if self.drawing and self.draw_start and self.draw_current:
            wx1,wy1 = self.imageToWidget(*self.draw_start)
            wx2,wy2 = self.imageToWidget(*self.draw_current)
            painter.setPen(QPen(QColor("#d9ff3f"),2,Qt.DashLine))
            painter.drawRect(min(wx1,wx2),min(wy1,wy2),abs(wx2-wx1),abs(wy2-wy1))

    def mousePressEvent(self, e):
        if e.button()==Qt.LeftButton and self.manual_mode:
            ix,iy = self.widgetToImage(e.pos().x(),e.pos().y())
            if ix<0 or iy<0: return
            for i,ann in enumerate(self.annotations):
                x1,y1,x2,y2,_,_,_ = ann
                ax1,ay1 = self.imageToWidget(x1,y1); ax2,ay2 = self.imageToWidget(x2,y2)
                wx,wy = self.imageToWidget(ix,iy)
                if ax1-8<=wx<=ax2+8 and ay1-8<=wy<=ay2+8:
                    self.annotations.pop(i); self.update(); self.regionDeleted.emit(i); return
            self.drawing=True; self.draw_start=(ix,iy); self.draw_current=(ix,iy)
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.drawing and self.manual_mode:
            ix,iy = self.widgetToImage(e.pos().x(),e.pos().y())
            self.draw_current=(ix,iy); self.update()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button()==Qt.LeftButton and self.drawing and self.manual_mode:
            self.drawing=False; ix,iy=self.widgetToImage(e.pos().x(),e.pos().y())
            x1=min(self.draw_start[0],ix); y1=min(self.draw_start[1],iy)
            x2=max(self.draw_start[0],ix); y2=max(self.draw_start[1],iy)
            if x2-x1>10 and y2-y1>10: self.regionSelected.emit(x1,y1,x2,y2)
            self.draw_start=None; self.draw_current=None; self.update()
        super().mouseReleaseEvent(e)


class SelectScreen(QWidget):
    """水样类型选择界面"""
    waterSelected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        eyebrow = QLabel("FIELD SAMPLE  /  ARCHIVE 01")
        eyebrow.setObjectName("eyebrow"); eyebrow.setAlignment(Qt.AlignCenter)
        layout.addWidget(eyebrow)

        title = QLabel("水体反应管检测系统")
        title.setObjectName("title"); title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("请选择待检测的水样类型")
        sub.setObjectName("subtitle"); sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        layout.addSpacing(20)

        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        btn_layout.setSpacing(24)

        indices = ["01", "02", "03"]
        for wt, index in zip(WATER_TYPES, indices):
            btn = QPushButton(f"{index} / WATER BODY\n{wt}")
            btn.setObjectName("waterBtn")
            btn.setFixedSize(200, 140)
            btn.clicked.connect(lambda checked, w=wt: self.waterSelected.emit(w))
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)
        layout.addSpacing(20)

        hint = QLabel("选择后将进入检测界面，结果自动保存到对应水体文件夹")
        hint.setStyleSheet("color:#59605b;font-size:12px"); hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)


class DetectScreen(QWidget):
    """检测界面"""
    def __init__(
        self,
        water_type,
        release=None,
        upload_queue=None,
        result_root=None,
        archive_root=None,
        wake_upload=None,
    ):
        super().__init__()
        self.water_type = water_type
        self.release = release
        self.upload_queue = upload_queue
        self.result_root = Path(result_root) if result_root is not None else None
        self.archive_root = Path(archive_root) if archive_root is not None else None
        self.wake_upload = wake_upload
        self.detector = None; self.classifier = None; self.img_size = None
        self.img_rgb = None; self.worker = None
        self._initUI()

    def _initUI(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10,10,10,10); layout.setSpacing(8)

        # 左侧
        left = QWidget()
        ll = QVBoxLayout(left); ll.setContentsMargins(0,0,0,0)

        # 工具栏
        eyebrow = QLabel("DETECTION LAYER  /  FIELD WORKSPACE")
        eyebrow.setObjectName("eyebrow")
        ll.addWidget(eyebrow)

        tb = QHBoxLayout()
        self.btn_open = QPushButton("打开图片")
        self.btn_open.clicked.connect(self.openImage); self.btn_open.setFixedHeight(40)
        tb.addWidget(self.btn_open)
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.saveResult); self.btn_save.setFixedHeight(40)
        tb.addWidget(self.btn_save)
        tb.addStretch()

        self.cb_water = QComboBox()
        self.cb_water.addItems(WATER_TYPES)
        self.cb_water.setCurrentText(self.water_type)
        self.cb_water.currentTextChanged.connect(self._onWaterChanged)
        tb.addWidget(self.cb_water)

        mode_group = QButtonGroup(self)
        ml = QHBoxLayout()
        self.rb_normal = QRadioButton("默认检测"); self.rb_normal.setChecked(True)
        self.rb_normal.toggled.connect(self.onModeChanged); mode_group.addButton(self.rb_normal)
        self.rb_scan = QRadioButton("精细扫描"); self.rb_scan.toggled.connect(self.onModeChanged)
        mode_group.addButton(self.rb_scan)
        self.rb_manual = QRadioButton("手动框选"); self.rb_manual.toggled.connect(self.onModeChanged)
        mode_group.addButton(self.rb_manual)
        ml.addWidget(self.rb_normal); ml.addWidget(self.rb_scan); ml.addWidget(self.rb_manual)
        tb.addLayout(ml); ll.addLayout(tb)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        self.image_view = ImageView()
        self.image_view.regionSelected.connect(self.onManualRegion)
        self.image_view.regionDeleted.connect(self.onManualRegionDeleted)
        scroll.setWidget(self.image_view); ll.addWidget(scroll, 1)

        # 右侧面板
        right = QWidget(); right.setFixedWidth(300)
        rl = QVBoxLayout(right); rl.setContentsMargins(4,0,4,0); rl.setSpacing(8)

        self.status_group = QGroupBox("STATUS / 状态")
        sl = QVBoxLayout()
        self.lbl_status = QLabel("等待打开图片...")
        self.lbl_status.setWordWrap(True); self.lbl_status.setStyleSheet("color:#59605b")
        sl.addWidget(self.lbl_status); self.status_group.setLayout(sl); rl.addWidget(self.status_group)

        self.result_group = QGroupBox("检测结果")
        self.result_layout = QVBoxLayout()
        self.lbl_no_result = QLabel("暂无结果"); self.lbl_no_result.setStyleSheet("color:#111111")
        self.result_layout.addWidget(self.lbl_no_result)
        self.result_group.setLayout(self.result_layout); rl.addWidget(self.result_group, 1)

        self.settings_group = QGroupBox("THRESHOLD / 阈值")
        stl = QVBoxLayout()
        self.lbl_thresholds = QLabel("检测阈值: 0.3\n分类阈值: 0.5")
        self.lbl_thresholds.setStyleSheet("color:#59605b;font-size:12px"); stl.addWidget(self.lbl_thresholds)
        self.settings_group.setLayout(stl); rl.addWidget(self.settings_group)
        rl.addStretch()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left); splitter.addWidget(right)
        splitter.setStretchFactor(0,3); splitter.setStretchFactor(1,1)
        layout.addWidget(splitter)

    def setModels(self, detector, classifier, img_size):
        self.detector = detector; self.classifier = classifier; self.img_size = img_size

    def openImage(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择反应管图片", "", "图片 (*.jpg *.jpeg *.png);;所有文件 (*)")
        if not path: return
        img_bgr = cv2.imread(path)
        if img_bgr is None: QMessageBox.warning(self,"错误","无法读取图片"); return
        self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.image_view.setImage(self.img_rgb)
        self.image_view.manual_mode = self.rb_manual.isChecked()
        self.clearAnnotations()
        self.lbl_no_result.setText("暂无结果")
        self.lbl_status.setText(f"图片: {Path(path).name}\n{self.img_rgb.shape[1]}×{self.img_rgb.shape[0]} | {self.water_type}")
        if not self.rb_manual.isChecked(): self.runAnalysis()

    def _onWaterChanged(self, text):
        """从下拉框切换水体类型"""
        self.water_type = text
        self.lbl_status.setText(self.lbl_status.text().rsplit("|",1)[0].strip() + f" | {self.water_type}")

    def onModeChanged(self):
        if self.img_rgb is None: return
        manual = self.rb_manual.isChecked(); self.image_view.manual_mode = manual
        if manual:
            self._stopWorker(); self.clearAnnotations()
            self.lbl_no_result.setText("🖱 在图片上拖拽鼠标框选反应管")
        else:
            self.runAnalysis()

    def runAnalysis(self):
        if self.img_rgb is None or self.detector is None: return
        self._startWorker("scan" if self.rb_scan.isChecked() else "normal")

    def _stopWorker(self):
        self.image_view.stopMotion()
        if self.worker is None: return
        try:
            if self.worker.isRunning():
                self.worker.quit()
                if not self.worker.wait(3000): self.worker.terminate(); self.worker.wait(2000)
        except: pass
        try: self.worker.deleteLater()
        except: pass
        self.worker = None

    def _startWorker(self, mode, manual_regions=None):
        self._stopWorker()
        if mode in ("normal", "scan"):
            self.image_view.startScanMotion()
        self.worker = AnalysisWorker(self.detector,self.classifier,self.img_size,self.img_rgb,mode,0.3,0.5)
        if manual_regions: self.worker.manual_regions = manual_regions
        self.worker.resultReady.connect(self._onWorkerFinished)
        self.worker.start()

    def _onWorkerFinished(self, results):
        if self.rb_manual.isChecked():
            for (x1,y1,x2,y2,label,conf) in results:
                color = CLASS_NAMES[1][1] if label=="已反应" else CLASS_NAMES[0][1]
                self.image_view.annotations.append((x1,y1,x2,y2,label,conf,color))
        else:
            anns = []
            for (x1,y1,x2,y2,label,conf) in results:
                color = CLASS_NAMES[1][1] if label=="已反应" else CLASS_NAMES[0][1]
                anns.append((x1,y1,x2,y2,label,conf,color))
            self.image_view.setAnnotations(anns)
        self.image_view.startResultMotion()
        self.image_view.update(); self.updateResultPanel()
        self.btn_save.setStyleSheet("border:2px solid #00a7b5;color:#111111")

    def onManualRegion(self, x1,y1,x2,y2):
        self._startWorker("manual_regions",manual_regions=[(x1,y1,x2,y2)])

    def onManualRegionDeleted(self, idx):
        self.updateResultPanel()

    def clearAnnotations(self):
        self.image_view.annotations = []; self.image_view.update()
        self.btn_save.setStyleSheet("")

    def updateResultPanel(self):
        for i in reversed(range(self.result_layout.count())):
            w = self.result_layout.itemAt(i).widget()
            if w and w != self.lbl_no_result: w.deleteLater()
        results = self.image_view.annotations
        if not results:
            self.lbl_no_result.setText("暂无结果"); self.lbl_no_result.setVisible(True); return
        self.lbl_no_result.setVisible(False)
        pos = sum(1 for r in results if r[4]=="已反应")
        neg = len(results)-pos
        s = QLabel(f"共 {len(results)} 个 | 🟢 {pos} 已反应 | 🔴 {neg} 未反应")
        s.setStyleSheet("font-weight:bold;padding:4px 0"); self.result_layout.addWidget(s)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        sw = QWidget(); sl = QVBoxLayout(sw); sl.setSpacing(4)
        prefix = "M" if self.rb_manual.isChecked() else "A"
        for i,r in enumerate(results):
            x1,y1,x2,y2,label,conf = r[:6]
            box = QFrame(); box.setFrameShape(QFrame.StyledPanel)
            box.setStyleSheet("background:#ffffff;border:1px solid #d7dad5;border-radius:0;padding:6px;margin:2px")
            bl = QVBoxLayout(box); bl.setSpacing(2)
            bl.addWidget(QLabel(f"{prefix}#{i+1} / {label}")); bl.itemAt(0).widget().setStyleSheet("font-weight:600;font-size:13px")
            bl.addWidget(QLabel(f"置信度: {conf:.1%}")); bl.itemAt(1).widget().setStyleSheet("color:#59605b;font-size:11px")
            bl.addWidget(QLabel(f"({x1},{y1})-({x2},{y2})")); bl.itemAt(2).widget().setStyleSheet("color:#7d8782;font-size:10px")
            sl.addWidget(box)
        sl.addStretch(); scroll.setWidget(sw); self.result_layout.addWidget(scroll)

    def saveResult(self):
        if self.img_rgb is None:
            QMessageBox.warning(self,"提示","请先打开图片并检测")
            return
        anns = self.image_view.annotations
        if not anns:
            QMessageBox.warning(self,"提示","没有检测结果可保存")
            return

        if self.release is None or self.upload_queue is None:
            QMessageBox.warning(self, "错误", "上传与发布配置尚未初始化")
            return
        mode = "manual" if self.rb_manual.isChecked() else ("scan" if self.rb_scan.isChecked() else "normal")
        saved = save_result(
            result_root=self.result_root,
            archive_root=self.archive_root,
            water_type=self.water_type,
            mode=mode,
            image_rgb=self.img_rgb,
            annotations=anns,
            release=self.release,
        )
        self.upload_queue.enqueue(
            UploadTask(
                upload_id=saved.upload_id,
                archive_path=saved.archive_path,
                dataset_generation=self.release.dataset_generation,
                app_release_id=self.release.app_release_id,
                model_generation=self.release.model_generation,
            )
        )
        if self.wake_upload:
            self.wake_upload()

        QMessageBox.information(self,"保存成功",
            f"本地结果已保存到:\n{saved.directory}\n\n"
            f"服务器同步已进入后台队列，待上传: {self.upload_queue.pending_count()}")
        self.btn_save.setStyleSheet("")


class ReactionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水体反应管检测系统 / ORBITAL WATER LAB")
        self.resize(1400,850); self.setMinimumSize(1000,600)
        self.release = load_release()
        paths = model_paths()
        self.detector_path = paths.detector
        self.classifier_path = paths.classifier
        data_root = app_data_root()
        self.result_root = result_root()
        self.archive_root = data_root / "pending_archives"
        self.upload_queue = UploadQueue(data_root / "upload_queue.db")
        self.upload_status_bridge = UploadStatusBridge()
        self.upload_status_bridge.statusChanged.connect(self._onUploadStatus)
        self.upload_worker = UploadWorker(
            self.upload_queue,
            ClientApi(self.release),
            status_callback=self.upload_status_bridge.publish,
        )
        self.upload_worker.start()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.select_screen = SelectScreen()
        self.select_screen.waterSelected.connect(self.onWaterSelected)
        self.stack.addWidget(self.select_screen)

        self.detect_screen = None
        self.statusBar().showMessage("就绪 / READY — 请选择水样类型")
        self.loadModels()

    def _onUploadStatus(self, status, pending):
        self.statusBar().showMessage(f"{status} / 待上传 {pending}")

    def loadModels(self):
        self.statusBar().showMessage("加载模型中...")
        try:
            missing = [
                path.name
                for path in (self.detector_path, self.classifier_path)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError("缺少外置模型: " + ", ".join(missing))
            self._detector = YOLO(str(self.detector_path))
            self._classifier = tf.keras.models.load_model(str(self.classifier_path))
            self._img_size = self._classifier.input_shape[1:3]
            self.statusBar().showMessage("模型加载完成 ✓ — 请选择水样类型")
        except Exception as e:
            QMessageBox.critical(self,"错误",f"模型加载失败:\n{e}")

    def onWaterSelected(self, water_type):
        if self.detect_screen is None:
            self.detect_screen = DetectScreen(
                water_type,
                release=self.release,
                upload_queue=self.upload_queue,
                result_root=self.result_root,
                archive_root=self.archive_root,
                wake_upload=self.upload_worker.wake,
            )
            self.detect_screen.setModels(self._detector, self._classifier, self._img_size)
            self.stack.addWidget(self.detect_screen)
        else:
            # 更新水体类型
            self.detect_screen.water_type = water_type
            self.detect_screen.cb_water.setCurrentText(water_type)
        self.stack.setCurrentWidget(self.detect_screen)
        self.statusBar().showMessage(f"当前水样类型: {water_type}")

    def closeEvent(self, event):
        if self.detect_screen:
            self.detect_screen._stopWorker()
        self.upload_worker.stop()
        self.upload_worker.join(timeout=3)
        self.upload_queue.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(ORBITAL_STYLE)
    window = ReactionApp()
    window.show()
    sys.exit(app.exec_())
