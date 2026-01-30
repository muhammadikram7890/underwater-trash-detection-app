from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as U 
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.block import Bottleneck
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels % self.groups == 0, f"Channels {channels} must be divisible by factor {factor}"
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
    
class C2f_EMA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5, ema_factor=8):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        # print("DEBUG C2f_EMA:", c1, c2, n, type(n), e)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0),
                EMA(self.c, factor=ema_factor)  # EMA di dalam tiap block
            )
            for _ in range(n)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
# Register the custom modules
U.EMA = EMA
tasks.__dict__['EMA'] = EMA
print(f"Modul EMA terdaftar: {hasattr(U, 'EMA')}")

U.C2f_EMA = C2f_EMA
tasks.__dict__['C2f_EMA'] = C2f_EMA
print(f"Modul C2f_EMA terdaftar: {hasattr(U, 'C2f_EMA')}")

# model = YOLO('Model/yolov8_training_runs/underwater_trash_detection/weights/best.pt')
# model_base = YOLO('Model/Model YOLO/yolov8_training_runs/underwater_trash_detection/weights/best.pt')
# model_ema = YOLO('Model/Model YOLO EMA/yolov8_training_runs/underwater_trash_detection/weights/best.pt')
model_base = YOLO('C:/Skripsi/app/model 3.pt')
model_ema = YOLO('C:/Skripsi/app/model 2.pt')

def handle_preview(file, selected_model):
    if not file:
        return None, "Tidak ada file yang dipilih"
            
    if not allowed_file(file.filename):
        return None, "Format file tidak didukung! (Hanya jpg, jpeg, png, atau mp4)"
            
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)
            
    if file_length > 50 * 1024 * 1024:
        return None, "Ukuran file melebihi 50MB!"

    filename = secure_filename(file.filename)
    upload_path = os.path.join('static', 'uploads', filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(upload_path)
    
    preview_image = f"/uploads/{filename}"
    return preview_image, None

def handle_detection(image_path, selected_model):
    full_path = os.path.join('static', image_path.strip('/'))
    model = model_ema if selected_model == 'ema' else model_base
    model.predict(full_path, save=True, project='static', name='results', exist_ok=True)
        
    base_name = os.path.splitext(os.path.basename(full_path))[0]
    ext = os.path.splitext(full_path)[1].lower()
    
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        avi_path = os.path.join('static', 'results', f'{base_name}.avi')
        mp4_path = os.path.join('static', 'results', f'{base_name}.mp4')
               
        if os.path.exists(avi_path):
            subprocess.run([
                'ffmpeg', '-y', '-i', avi_path, '-vcodec', 'libx264', '-acodec', 'aac', mp4_path
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result_path = f"results/{base_name}.mp4"
        else:
            result_path = f"results/{base_name}.avi"
    else:
        result_path = f"results/{base_name}.jpg"

    return result_path

@app.route('/', methods=['GET', 'POST'])
def index():
    preview_image = None
    result_image = None
    selected_model = 'ema'
    message = None
    
    if request.method == 'POST':
        print("method post loh ini")
        
        if 'preview' in request.form:
            file = request.files['image']
            selected_model = request.form.get('model', 'ema')
            print(f"ini file : {file.filename}")
            preview_image, message = handle_preview(file, selected_model)
        
        if 'detect' in request.form:
            image_path = request.form.get('image_path') 
            selected_model = request.form.get('model', 'ema')
            result_image = handle_detection(image_path, selected_model)
            preview_image = image_path

    return render_template('index.jinja', preview_image=preview_image, result_image=result_image, selected_model=selected_model, message = message)

if __name__ == '__main__':
    app.run(debug=True)
