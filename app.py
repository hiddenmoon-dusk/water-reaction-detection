"""
反应管检测与分类系统 — Streamlit Web 界面
启动方式: streamlit run app.py
"""
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import io
import base64
import json

# ===== 检测错误启动方式 =====
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        print("=" * 60)
        print("  ❌ 请使用: streamlit run app.py")
        print("=" * 60)
        sys.exit(1)
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DETECTOR_PATH = ROOT / "models" / "detector.pt"
CLASSIFIER_PATH = ROOT / "models" / "classifier.h5"
CLASS_NAMES = {0: ("未反应 (Negative)", (255, 80, 80)), 1: ("已反应 (Positive)", (80, 200, 80))}

st.set_page_config(page_title="反应管检测与分类系统", page_icon="🧪", layout="wide")

# ===== session_state =====
for key, default in [("manual_boxes", {}), ("manual_results", {}), ("pending_manual", None),
                      ("_cached_images", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

# ===== 模型 =====
@st.cache_resource
def load_models():
    detector = YOLO(str(DETECTOR_PATH))
    classifier = tf.keras.models.load_model(str(CLASSIFIER_PATH))
    return detector, classifier, classifier.input_shape[1:3]


def tiled_detect(detector, image, tile_size=640, overlap=0.2, conf=0.3, progress_callback=None):
    h, w = image.shape[:2]
    if h <= tile_size and w <= tile_size:
        results = detector(image, conf=conf, verbose=False)
        all_boxes = []
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                all_boxes.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])
        return all_boxes

    sh = int(tile_size * (1 - overlap)); sw = int(tile_size * (1 - overlap))
    rows = max(1, (h - tile_size) // sh + 2); cols = max(1, (w - tile_size) // sw + 2)
    total = rows * cols; done = 0; all_boxes = []

    for row in range(rows):
        for col in range(cols):
            x = min(col * sw, w - tile_size); y = min(row * sh, h - tile_size)
            x2 = min(x + tile_size, w); y2 = min(y + tile_size, h)
            x = max(0, x2 - tile_size); y = max(0, y2 - tile_size)
            tile = image[y:y2, x:x2]
            results = detector(tile, conf=conf, verbose=False)
            for r in results:
                if r.boxes is None: continue
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    all_boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y, float(box.conf[0]), int(box.cls[0])])
            done += 1
            if progress_callback: progress_callback(done, total)

    if len(all_boxes) < 2: return all_boxes
    all_boxes = np.array(all_boxes); boxes_xyxy = all_boxes[:, :4]; scores = all_boxes[:, 4]
    order = scores.argsort()[::-1]; keep = []
    while len(order) > 0:
        keep.append(order[0])
        if len(order) == 1: break
        x1 = np.maximum(boxes_xyxy[order[0], 0], boxes_xyxy[order[1:], 0])
        y1 = np.maximum(boxes_xyxy[order[0], 1], boxes_xyxy[order[1:], 1])
        x2 = np.minimum(boxes_xyxy[order[0], 2], boxes_xyxy[order[1:], 2])
        y2 = np.minimum(boxes_xyxy[order[0], 3], boxes_xyxy[order[1:], 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes_xyxy[order[0], 2] - boxes_xyxy[order[0], 0]) * (boxes_xyxy[order[0], 3] - boxes_xyxy[order[0], 1])
        area2 = (boxes_xyxy[order[1:], 2] - boxes_xyxy[order[1:], 0]) * (boxes_xyxy[order[1:], 3] - boxes_xyxy[order[1:], 1])
        iou = inter / (area1 + area2 - inter + 1e-6)
        order = order[1:][iou < 0.5]
    return all_boxes[keep].tolist()


def classify_crop(cropped_rgb, classifier, img_size, cls_thresh):
    resized = cv2.resize(cropped_rgb, img_size)
    pred = classifier.predict(np.expand_dims(resized, axis=0), verbose=0)[0][0]
    pc = 1 if pred > cls_thresh else 0
    label, color = CLASS_NAMES[pc]
    return label, color, pred if pc == 1 else 1 - pred


def draw_manual_boxes_on_image(img_rgb, boxes, results):
    for i, (box, result) in enumerate(zip(boxes, results)):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        label = result.get("结果", ""); conf = result.get("置信度", "")
        color = (80, 200, 80) if "已反应" in label else (255, 80, 80)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)
        text = f"M#{i + 1} {label.split(' ')[0]} {conf}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img_rgb, (x1, y1 - th - 8), (x1 + tw + 5, y1), color, -1)
        cv2.putText(img_rgb, text, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


# ===== Canvas HTML（用 form 提交替代 postMessage） =====
CANVAS_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#0e1117;color:#fafafa}
.toolbar{display:flex;gap:8px;padding:8px 0;align-items:center;flex-wrap:wrap}
.toolbar button{padding:6px 14px;border:1px solid #555;border-radius:6px;cursor:pointer;font-size:13px;background:#262730;color:#fafafa}
.toolbar button:hover{background:#3a3a45}
.toolbar button.submit{background:#1a6b3c;border-color:#1a6b3c}
.toolbar button.submit:hover{background:#228b4a}
.toolbar button.danger{background:#7b1e1e;border-color:#7b1e1e}
.toolbar span{font-size:13px;color:#aaa;margin-left:auto}
.container{position:relative;display:inline-block;border:1px solid #333;border-radius:4px;overflow:hidden;cursor:crosshair}
canvas{display:block}
.legend{font-size:12px;color:#888;padding:4px 0}
</style></head><body>
<div class="toolbar">
  <button onclick="undoLast()">↩ 撤销</button>
  <button onclick="clearAll()" class="danger">✕ 清空</button>
  <span id="counter" style="color:#4caf50">🖱 拖拽鼠标框选反应管</span>
</div>
<div class="container"><canvas id="canvas"></canvas></div>
<form id="submitForm" method="GET" target="_top" style="display:none">
  <input type="hidden" name="manual_data" id="manualData">
  <input type="hidden" name="manual_file" id="manualFile">
</form>
<script>
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const IMG_W=%IW%,IMG_H=%IH%,MAX_DISP=%MD%;
let displayW,displayH,scale;
let rects=[];
let drawing=false,startX,startY;
let autoSubmitTimer=null;
const img=new Image();
img.onload=function(){
    displayW=Math.min(IMG_W,MAX_DISP);
    displayH=Math.round(IMG_H*(displayW/IMG_W));
    scale=IMG_W/displayW;
    canvas.width=displayW;canvas.height=displayH;
    redraw();
};
img.src="%IMG%";

function getPos(e){
    const r=canvas.getBoundingClientRect();
    return{x:(e.clientX-r.left)*(displayW/canvas.offsetWidth),y:(e.clientY-r.top)*(displayH/canvas.offsetHeight)};
}
function scheduleAutoSubmit(){
    if(autoSubmitTimer)clearTimeout(autoSubmitTimer);
    if(rects.length===0)return;
    autoSubmitTimer=setTimeout(function(){
        document.getElementById('counter').textContent='⏳ 正在分析 '+rects.length+' 个区域...';
        document.getElementById('counter').style.color='#ffaa00';
        var data=rects.map(function(r){return[r.ix1,r.iy1,r.ix2,r.iy2];});
        document.getElementById('manualData').value=JSON.stringify(data);
        document.getElementById('manualFile').value='%FNAME%';
        document.getElementById('submitForm').action=window.parent.location.pathname;
        document.getElementById('submitForm').submit();
    },1500);
}
canvas.onmousedown=function(e){
    if(autoSubmitTimer){clearTimeout(autoSubmitTimer);autoSubmitTimer=null;}
    const p=getPos(e);startX=p.x;startY=p.y;
    for(let i=rects.length-1;i>=0;i--){
        const r=rects[i];
        if(p.x>=r.cx1-6&&p.x<=r.cx2+6&&p.y>=r.cy1-6&&p.y<=r.cy2+6){
            rects.splice(i,1);redraw();updateCounter();
            scheduleAutoSubmit();
            return;
        }
    }
    drawing=true;
};
canvas.onmousemove=function(e){
    if(!drawing)return;
    const p=getPos(e);
    redraw();
    ctx.strokeStyle='#ffaa00';ctx.lineWidth=2;ctx.setLineDash([6,3]);
    ctx.strokeRect(Math.min(startX,p.x),Math.min(startY,p.y),Math.abs(p.x-startX),Math.abs(p.y-startY));
    ctx.setLineDash([]);
};
canvas.onmouseup=function(e){
    if(!drawing)return;drawing=false;
    const p=getPos(e),x1=Math.min(startX,p.x),y1=Math.min(startY,p.y),x2=Math.max(startX,p.x),y2=Math.max(startY,p.y);
    if(Math.abs(x2-x1)<10||Math.abs(y2-y1)<10){redraw();return;}
    rects.push({cx1:x1,cy1:y1,cx2:x2,cy2:y2,ix1:Math.round(x1*scale),iy1:Math.round(y1*scale),ix2:Math.round(x2*scale),iy2:Math.round(y2*scale)});
    redraw();updateCounter();
    scheduleAutoSubmit();
};
function redraw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(img,0,0,displayW,displayH);
    rects.forEach(function(r,i){
        ctx.fillStyle='rgba(100,200,100,0.12)';
        ctx.fillRect(r.cx1,r.cy1,r.cx2-r.cx1,r.cy2-r.cy1);
        ctx.strokeStyle='#4caf50';ctx.lineWidth=2.5;
        ctx.strokeRect(r.cx1,r.cy1,r.cx2-r.cx1,r.cy2-r.cy1);
        ctx.fillStyle='#4caf50';ctx.font='bold 14px sans-serif';
        ctx.fillText('#'+(i+1),r.cx1+4,r.cy1>22?r.cy1-6:r.cy1+18);
    });
}
function updateCounter(){
    if(rects.length===0){
        document.getElementById('counter').textContent='🖱 拖拽鼠标框选反应管';
        document.getElementById('counter').style.color='#4caf50';
    } else {
        document.getElementById('counter').textContent='已选: '+rects.length+' 个区域 (1.5秒后自动分析)';
        document.getElementById('counter').style.color='#4caf50';
    }
}
function undoLast(){
    if(rects.length){rects.pop();redraw();updateCounter();scheduleAutoSubmit();}
}
function clearAll(){
    rects=[];redraw();updateCounter();
    if(autoSubmitTimer){clearTimeout(autoSubmitTimer);autoSubmitTimer=null;}
}
</script></body></html>"""


def render_selection_canvas(img_rgb, filename, max_display=900):
    h, w = img_rgb.shape[:2]
    disp_w = min(w, max_display); disp_h = int(h * (disp_w / w))
    disp_img = cv2.resize(img_rgb, (disp_w, disp_h))
    buf = io.BytesIO(); Image.fromarray(disp_img).save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    html = CANVAS_HTML.replace("%IW%", str(w)).replace("%IH%", str(h))
    html = html.replace("%MD%", str(max_display))
    html = html.replace("%IMG%", f"data:image/jpeg;base64,{b64}")
    html = html.replace("%FNAME%", filename)
    components.html(html, height=disp_h + 75, scrolling=False)


with st.spinner("正在加载模型..."):
    try:
        detector, classifier, img_size = load_models()
        st.sidebar.success("模型加载成功")
    except Exception as e:
        st.error(f"模型加载失败: {e}"); st.stop()

st.sidebar.title("⚙️ 设置")
conf_threshold = st.sidebar.slider("目标检测置信度阈值", 0.1, 0.9, 0.3, 0.05)
cls_threshold = st.sidebar.slider("分类置信度阈值", 0.5, 0.95, 0.5, 0.05)
scan_mode = st.sidebar.checkbox("🔍 精细扫描模式", value=True)
st.sidebar.divider()
st.sidebar.caption("💡 自动检测 + 手动框选")
st.sidebar.caption("📊 YOLOv8n + CNN")
st.sidebar.caption("🧪 未反应 / 已反应")

st.title("🧪 反应管检测与分类系统")
st.caption("自动检测 → 手动框选 → 分类判断")

# ===== 处理 form 提交的手动框选数据 =====
manual_submitted = False
manual_submit_fname = None
try:
    qp = st.query_params
    if "manual_data" in qp and "manual_file" in qp:
        manual_submit_fname = qp["manual_file"]
        coords = json.loads(qp["manual_data"])
        if isinstance(coords, list) and len(coords) > 0:
            valid = all(isinstance(c, list) and len(c) == 4 for c in coords)
            if valid:
                st.session_state.manual_boxes[manual_submit_fname] = coords
                st.session_state.pending_manual = manual_submit_fname
                manual_submitted = True
        st.query_params.clear()
except Exception:
    pass

uploaded_files = st.file_uploader(
    "上传反应管图片", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# 缓存上传的图片（form 提交后页面刷新也能找回）
if uploaded_files:
    for uf in uploaded_files:
        st.session_state["_cached_images"][uf.name] = uf.read()

# 构建处理列表
files_to_process = {}
if uploaded_files:
    for uf in uploaded_files:
        b = st.session_state["_cached_images"].get(uf.name)
        if b is None:
            b = uf.read()
        files_to_process[uf.name] = b
# form 提交后找回缓存图片（文件上传器为空但有缓存）
if manual_submitted and manual_submit_fname:
    cached = st.session_state["_cached_images"].get(manual_submit_fname)
    if cached:
        files_to_process[manual_submit_fname] = cached

if files_to_process:
    for fname in list(files_to_process.keys()):
        file_bytes = files_to_process[fname]
        st.divider()
        file_bytes_np = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        col1, col2 = st.columns([3, 2])
        with col1: st.subheader(f"📷 {fname}")

        # ===== 处理手动提交 =====
        if st.session_state.pending_manual == fname and st.session_state.manual_boxes.get(fname):
            boxes = st.session_state.manual_boxes[fname]; results = []
            for (x1, y1, x2, y2) in boxes:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                y1 = max(0, y1); y2 = min(h, y2); x1 = max(0, x1); x2 = min(w, x2)
                if y2 - y1 <= 0 or x2 - x1 <= 0:
                    results.append({"结果": "无效", "置信度": "N/A"}); continue
                label, color, conf = classify_crop(img_rgb[y1:y2, x1:x2], classifier, img_size, cls_threshold)
                results.append({"结果": label.split(" ")[0], "置信度": f"{conf:.2%}", "位置": f"({x1},{y1})-({x2},{y2})"})
            st.session_state.manual_results[fname] = results
            st.session_state.pending_manual = None

        # ===== 自动检测 =====
        with st.status("正在自动检测...", expanded=False) as status:
            if scan_mode and (h > 640 or w > 640):
                all_boxes = tiled_detect(detector, img_rgb, tile_size=640, overlap=0.2, conf=conf_threshold)
                status.update(label=f"精细扫描完成，发现 {len(all_boxes)} 个候选目标")
            else:
                results = detector(img_rgb, conf=conf_threshold, verbose=False); all_boxes = []
                for r in results:
                    if r.boxes is None: continue
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        all_boxes.append([x1, y1, x2, y2, float(box.conf[0]), int(box.cls[0])])
                status.update(label=f"检测完成，发现 {len(all_boxes)} 个候选目标", state="complete")

        # ===== 分类标注 =====
        auto_boxes_data = []
        for bi in all_boxes:
            x1, y1, x2, y2 = int(bi[0]), int(bi[1]), int(bi[2]), int(bi[3])
            y1 = max(0, y1); y2 = min(h, y2); x1 = max(0, x1); x2 = min(w, x2)
            if y2 - y1 <= 0 or x2 - x1 <= 0: continue
            label, color, conf = classify_crop(img_rgb[y1:y2, x1:x2], classifier, img_size, cls_threshold)
            auto_boxes_data.append({"编号": len(auto_boxes_data) + 1, "结果": label.split(" ")[0], "置信度": f"{conf:.2%}", "位置": f"({x1},{y1})-({x2},{y2})"})
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)
            text = f"A{len(auto_boxes_data)} {label.split(' ')[0]} {conf:.1%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_rgb, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img_rgb, text, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        manual_results = st.session_state.manual_results.get(fname, [])
        if manual_results and st.session_state.manual_boxes.get(fname):
            draw_manual_boxes_on_image(img_rgb, st.session_state.manual_boxes[fname], manual_results)

        with col1: st.image(img_rgb, use_container_width=True)

        with col2:
            st.subheader("📊 检测结果")
            if auto_boxes_data:
                st.caption(f"🤖 自动检测: **{len(auto_boxes_data)}** 个")
                for bd in auto_boxes_data:
                    emoji = "🟢" if "已反应" in bd["结果"] else "🔴"
                    st.metric(label=f"A#{bd['编号']} — {emoji} {bd['结果']}", value=f"置信度: {bd['置信度']}")
            else:
                st.warning("🤖 自动检测: 未发现反应管")

            st.divider()
            if manual_results and st.session_state.manual_boxes.get(fname):
                st.caption(f"✂️ 手动框选: **{len(manual_results)}** 个")
                for i, mr in enumerate(manual_results):
                    emoji = "🟢" if "已反应" in mr.get("结果", "") else "🔴"
                    st.metric(label=f"M#{i + 1} — {emoji} {mr.get('结果', '')}", value=f"置信度: {mr.get('置信度', '')}")
            else:
                st.caption("✂️ 手动框选: 暂未框选")

            if auto_boxes_data or manual_results:
                buf = io.BytesIO(); Image.fromarray(img_rgb).save(buf, format="PNG")
                st.download_button("📥 下载标注结果图", buf.getvalue(),
                                   f"result_{fname.rsplit('.',1)[0]}.png", "image/png")

        # ===== 手动框选画布 =====
        st.divider()
        in_manual = st.session_state.get(f"_manual_mode_{fname}", False)
        if not in_manual:
            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("✂️ 进入手动框选", key=f"enter_m_{fname}"):
                    st.session_state[f"_manual_mode_{fname}"] = True
                    st.rerun()
            with c2:
                st.caption("点击后在下方图片上拖拽鼠标框选反应管区域。")
        else:
            st.info("🖱 在图片上**拖拽鼠标**框选反应管 → 画完 **1.5 秒后自动分析** → 结果显示在标注图中（M# 标签）→ 可继续框选更多。")
            render_selection_canvas(img_rgb, fname)
            ce1, ce2 = st.columns([1, 4])
            with ce1:
                if st.button("↩ 退出框选模式", key=f"exit_m_{fname}"):
                    st.session_state[f"_manual_mode_{fname}"] = False
                    st.rerun()

else:
    st.markdown("""
### 🚀 快速开始
1. 上传反应管图片
2. 系统自动检测并分类
3. 有遗漏？点击「✂️ 进入手动框选」→ 在图片上拖拽鼠标框选
4. 点击「✓ 分析选中区域」→ CNN 分类 → 结果显示

---
### 📋 三种检测方式
| 方式 | 说明 |
|------|------|
| 🤖 自动检测 | YOLO 全图检测 |
| 🔍 精细扫描 | 滑动窗口分块检测（侧边栏开启） |
| ✂️ 手动框选 | 鼠标拖拽框选，1.5秒后自动分析 |
""")
