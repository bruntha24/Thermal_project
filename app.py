from flask import Flask, render_template_string, request, send_file
import pandas as pd
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor, Color
import math
import torch
import torch.nn as nn

app = Flask(__name__)

# ======================
# PINN MODEL DEFINITION
# ======================
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()
    
    def forward(self, x):
        a = x
        for i in range(len(self.layers)-1):
            a = self.activation(self.layers[i](a))
        return self.layers[-1](a)

def train_pinn(T_base, T_ambient, fin_length, h, k, A, P, epochs=500):
    x = torch.linspace(0, fin_length, 100).reshape(-1,1)
    x.requires_grad = True
    model = PINN([1, 20, 20, 20, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()
        T_pred = model(x)
        dTdx = torch.autograd.grad(T_pred, x, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0]
        d2Tdx2 = torch.autograd.grad(dTdx, x, grad_outputs=torch.ones_like(dTdx), create_graph=True)[0]

        # 1D fin differential equation: d²T/dx² - (h*P/k*A)*(T-T_inf) = 0
        f = d2Tdx2 - (h*P/(k*A))*(T_pred - T_ambient)

        # Boundary condition at base: T(0) = T_base
        bc_loss = (T_pred[0] - T_base)**2

        loss = torch.mean(f**2) + bc_loss
        loss.backward()
        optimizer.step()

    return x.detach().numpy(), T_pred.detach().numpy()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Thermal Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { margin:0; font-family: 'Segoe UI', sans-serif; background:#f4f6f8; }
        header { width:100%; background:#004d4d; color:white; padding:15px 30px; display:flex; align-items:center; box-sizing:border-box; margin:0; }
        header .title { font-size:1.4em; flex:1; }
        header .brand { font-size:1.1em; text-align:right; }
        .dashboard { display:flex; padding:20px; gap:20px; max-width:1200px; margin:auto; }
        .sidebar { width:280px; display:flex; flex-direction:column; gap:15px; }
        .card { background:white; border-radius:12px; padding:15px; box-shadow:0 6px 15px rgba(0,0,0,0.1); text-align:center; }
        .card h3 { color:#34495e; margin-bottom:10px; }
        .card p { font-size:1.2em; font-weight:bold; margin:0; }
        .fin-container { display:flex; justify-content:center; flex-wrap:wrap; gap:2px; padding-top:10px; }
        .fin { width:10px; border-radius:4px; transition: all 0.4s; }
        .main { flex:1; display:flex; flex-direction:column; gap:20px; }
        form { background:white; padding:20px; border-radius:12px; box-shadow:0 6px 15px rgba(0,0,0,0.1); display:flex; flex-direction:column; gap:15px; }
        input { padding:8px; border-radius:6px; border:1px solid #ccc; width:100%; }
    .download-form {
    text-align: center;  /* center the button horizontally */
    margin: 20px 0;      /* spacing above and below */
}

button { padding: 20px 20px;
 border: none;
   border-radius: 20px; 
   background: #3498db; 
   color: white;
     font-weight: bold;
       cursor: pointer;
         transition: 0.3s; 
         display: block; /* allows margin auto to center */
           width: auto; /* do NOT stretch full width */ 
           max-width: 280px; /* optional: limit  button width */
              margin: 20px auto 0; /* 20px top, auto left/right centers it */ }

.download-form .icon {
    font-size: 1.2em;   /* bigger icon */
}
        .chart-card { background:white; border-radius:12px; padding:15px; box-shadow:0 6px 15px rgba(0,0,0,0.1); }
        .export-btn { display:flex; gap:15px; justify-content:flex-start; flex-wrap:wrap; }
        .export-btn button { width:160px; }
    </style>
</head>
<body>
<header>
    <div class="title">Temperature Analysis Monitoring</div>
   
</header>

<div class="dashboard">
    <div class="sidebar">
        <form method="POST">
            <label>Processor Power (W)</label>
            <input type="number" name="Q" value="{{ Q }}" required>
            <label>Number of Fins</label>
            <input type="number" name="N_fins" value="{{ N_fins }}" required>
            <label>Air Velocity (m/s)</label>
            <input type="number" step="0.1" name="velocity" value="{{ velocity }}" required>
            <button type="submit">Calculate</button>
        </form>

        {% if results %}
        <div class="card">
            <h3>Heat Sink Resistance</h3>
            <p>{{ results.R_hs|round(3) }} °C/W</p>
        </div>
        <div class="card">
            <h3>Total Thermal Resistance</h3>
            <p>{{ results.R_total|round(3) }} °C/W</p>
        </div>
        <div class="card">
            <h3>Junction Temperature</h3>
            <p style="color: {% if results.T_junction>85 %}red{% elif results.T_junction>70 %}orange{% else %}green{% endif %};">
                {{ results.T_junction|round(2) }} °C
            </p>
        </div>

        <div class="card">
            <h3>Heat Sink Visualization</h3>
            <div class="fin-container">
            {% for i in range(results.N_fins) %}
                <div class="fin" style="height:{{50 + (results.T_junction-25)}}px; background:hsl({{120-(results.T_junction-25)*2}},70%,50%)"></div>
            {% endfor %}
            </div>
        </div>

        <div class="export-btn">
             <form method="POST" action="/export" style="display:flex; justify-content:center; margin-top:15px;">
    <input type="hidden" name="R_hs" value="{{ results.R_hs }}">
    <input type="hidden" name="R_total" value="{{ results.R_total }}">
    <input type="hidden" name="T_junction" value="{{ results.T_junction }}">
    <input type="hidden" name="N_fins" value="{{ results.N_fins }}">
    
    <input type="submit" value="⬇ Download CSV Report"
    style="
        background: linear-gradient(135deg,#27ae60,#2ecc71);
        color:white;
        border:none;
        padding:12px 25px;
        border-radius:10px;
        font-size:14px;
        font-weight:600;
        cursor:pointer;
        box-shadow:0 4px 10px rgba(0,0,0,0.2);
        transition:all 0.3s ease;
        letter-spacing:0.5px;
    ">
</form>
                <form method="POST" action="/export_pdf" style="
    display:flex; 
    flex-direction:column; 
    align-items:center; 
    justify-content:center; 
    background:#ffffff; 
    padding:25px 30px; 
    border-radius:12px; 
    box-shadow:0 6px 15px rgba(0,0,0,0.1); 
    margin-top:20px; 
    width:280px;
    text-align:center;
">

    <h3 style="
        color:#2980b9; 
        margin-bottom:15px; 
        font-family:'Segoe UI',sans-serif;
        letter-spacing:0.5px;
    ">Download PDF Report</h3>
     
    <input type="hidden" name="R_hs" value="{{ results.R_hs }}">
    <input type="hidden" name="R_total" value="{{ results.R_total }}">
    <input type="hidden" name="T_junction" value="{{ results.T_junction }}">
    <input type="hidden" name="N_fins" value="{{ results.N_fins }}">

    <input type="submit" value="📄 Download PDF"
    style="
        background: linear-gradient(135deg,#e74c3c,#ff6b6b);
        color:white;
        border:none;
        padding:12px 25px;
        border-radius:10px;
        font-size:14px;
        font-weight:600;
        cursor:pointer;
        box-shadow:0 4px 10px rgba(0,0,0,0.2);
        transition:all 0.3s ease;
        letter-spacing:0.5px;
        margin-top:10px;
        width:100%;
    ">
</form>
        </div>
        {% endif %}
    </div>

    <div class="main">
        {% if results %}
        <div class="chart-card" id="chart1" style="height:300px;"></div>
        <div class="chart-card" id="chart2" style="height:300px;"></div>
        <div class="chart-card" id="chart3" style="height:300px;"></div>
        <div class="chart-card" id="chart4" style="height:300px;"></div>
        
        <script>
            // Chart 1: Junction Temp vs Fins
            Plotly.newPlot('chart1', [{
                x: Array.from({length: 20}, (_, i) => i+10),
                y: Array.from({length: 20}, (_, i) => i*0.5 + {{ results.T_junction|round(2) }} -5),
                mode:'lines+markers', line:{color:'#3498db'}, name:'Junction Temp'
            }], {title:'Junction Temperature vs Number of Fins', xaxis:{title:'Fins'}, yaxis:{title:'Temp (°C)'}});

            // Chart 2: Heat Sink Resistance vs Fins
            Plotly.newPlot('chart2', [{
                x: Array.from({length: 20}, (_, i)=>i+10),
                y: Array.from({length: 20}, (_, i)=>{{ results.R_hs|round(2) }}*Math.random()+0.01),
                type:'bar', name:'Heat Sink Resistance', marker:{color:'#e67e22'}
            }], {title:'Heat Sink Resistance vs Number of Fins', xaxis:{title:'Fins'}, yaxis:{title:'Resistance (°C/W)'}});

            // Chart 3: Total Thermal Resistance vs Air Velocity
            Plotly.newPlot('chart3', [{
                x: Array.from({length: 20}, (_, i)=>i+0.5),
                y: Array.from({length: 20}, (_, i)=>{{ results.R_total|round(2) }}*Math.random()+0.02),
                mode:'lines+markers', line:{color:'#2ecc71'}, name:'Total Thermal Resistance'
            }], {title:'Total Thermal Resistance vs Air Velocity', xaxis:{title:'Velocity (m/s)'}, yaxis:{title:'Resistance (°C/W)'}});
            // Chart 4: Junction Temperature Distribution (Histogram)
    Plotly.newPlot('chart4', [{
        x: Array.from({length: 50}, () => {{ results.T_junction|round(2) }} + (Math.random()-0.5)*5),
        type:'histogram',
        marker:{color:'#9b59b6'},
        name:'Temp Distribution'
    }], {
        title:'Junction Temperature Distribution',
        xaxis:{title:'Temperature (°C)'},
        yaxis:{title:'Frequency'}
    });
        </script>
        <div class="chart-card" id="pinn-chart" style="height:300px;"></div>
<script>
    {% if results.fin_x and results.fin_T %}
    Plotly.newPlot('pinn-chart', [{
    x: {{ results.fin_x | safe }},
    y: {{ results.fin_T | safe }},
    mode: 'lines+markers',
    line: {color:'#c0392b'},
    name:'Fin Temperature'
}], {
    title:'Fin Temperature Profile (PINN)',
    xaxis:{title:'Fin Length (m)'},
    yaxis:{title:'Temperature (°C)'}
});
    {% endif %}
</script>
         
        
        {% endif %}
    </div>
</div>
</body>
<footer style="
background:#004d4d;
color:white;
text-align:center;
padding:15px 10px;
margin-top:20px;
font-family:'Segoe UI', sans-serif;
font-size:14px;
">

<div style="margin-bottom:5px;">
Thermal Analysis Dashboard
</div>

<div style="opacity:0.8;">
© 2026 All Rights Reserved
</div>

</footer>
</html>
"""
# =========================
# FLASK ROUTES
# =========================

# ======================
# FLASK ROUTES
# ======================
@app.route("/", methods=["GET", "POST"])
def dashboard():
    # Default input values
    Q = 150
    N_fins = 60
    velocity = 1
    results = None

    if request.method == "POST":
        Q = float(request.form.get("Q", 150))
        N_fins = int(request.form.get("N_fins", 60))
        velocity = float(request.form.get("velocity", 1))

        # --- Fixed Parameters ---
        T_ambient = 25
        A_die = 0.0023625
        k_al = 167
        base_thickness = 0.0025
        fin_thickness = 0.0008
        fin_height = 0.0245
        h = 23.2879824
        A_single_fin = 0.004482
        A_base_exposed = 0.00612
        R_internal_gap = 0.200095

        # --- Thermal Calculations ---
        m = math.sqrt((2 * h) / (k_al * fin_thickness))
        eta_fin = math.tanh(m * fin_height) / (m * fin_height)
        A_eff = A_base_exposed + (N_fins * A_single_fin * eta_fin)
        R_conv = 1 / (h * A_eff)
        R_cond_base = base_thickness / (k_al * A_die)
        R_hs_total = R_internal_gap + R_cond_base + R_conv
        R_total = R_hs_total
        T_junction = T_ambient + (Q * R_total)

        # --- PINN Fin Temperature Profile ---
        fin_x, fin_T = train_pinn(
            T_base=T_junction,
            T_ambient=T_ambient,
            fin_length=fin_height,
            h=h,
            k=k_al,
            A=A_single_fin,
            P=0.05,
            epochs=500
        )

        results = {
            "R_hs": R_hs_total,
            "R_total": R_total,
            "T_junction": T_junction,
            "N_fins": N_fins,
            "fin_x": fin_x.flatten().tolist(),   # flatten 2D to 1D
    "fin_T": fin_T.flatten().tolist()
        }

    return render_template_string(
        HTML_TEMPLATE,
        Q=Q,
        N_fins=N_fins,
        velocity=velocity,
        results=results
    )

@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    R_hs = float(request.form.get("R_hs"))
    T_junction = float(request.form.get("T_junction"))
    N_fins = int(request.form.get("N_fins"))
    T_ambient = 25

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # ------------------ Title ------------------
    c.setFont("Helvetica-Bold", 26)
    c.setFillColor(HexColor("#2c3e50"))  # dark blue
    c.drawCentredString(width / 2, height - 80, "Thermal Analysis Report")

    # ------------------ Details ------------------
    start_y = height - 150
    line_spacing = 35

    # Total Heat Sink Resistance
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(HexColor("#e74c3c"))  # red
    c.drawCentredString(width / 2, start_y, f"Total Heat Sink Resistance: {R_hs:.6f} °C/W")

    # Junction Temperature
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(HexColor("#2980b9"))  # blue
    c.drawCentredString(width / 2, start_y - line_spacing, f"Junction Temperature: {T_junction:.5f} °C")

    # Number of Fins
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(HexColor("#27ae60"))  # green
    c.drawCentredString(width / 2, start_y - 2*line_spacing, f"Number of Fins: {N_fins}")

    # ------------------ Heat Sink Thermal Gradient ------------------
    gradient_title_y = start_y - 3*line_spacing - 10
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(HexColor("#34495e"))  # dark gray
    c.drawCentredString(width / 2, gradient_title_y, "Heat Sink Thermal Gradient:")

    # Gradient Bars
    fin_width = 12
    fin_height = 50
    spacing = 3
    max_fins = min(N_fins, 50)  # Limit to avoid overflow
    total_gradient_width = max_fins * (fin_width + spacing) - spacing
    start_x = (width - total_gradient_width) / 2  # center bars
    start_y = gradient_title_y - 60  # below title

    for i in range(max_fins):
        ratio = min(1, max(0, (T_junction - T_ambient)/100))
        color = Color(max(0, 1 - ratio), min(1, ratio), 0)  # green → yellow → red
        c.setFillColor(color)
        c.rect(start_x + i * (fin_width + spacing), start_y, fin_width, fin_height, fill=1, stroke=0)

    # Temperature scale below bars
    c.setFont("Helvetica", 10)
    c.setFillColor(HexColor("#7f8c8d"))
    c.drawString(start_x, start_y - 15, f"{T_ambient}°C")
    c.drawRightString(start_x + total_gradient_width, start_y - 15, f"{T_junction:.1f}°C")

    # ------------------ Footer ------------------
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(HexColor("#7f8c8d"))
    c.drawCentredString(width / 2, 30, "Generated by Thermal Analysis Dashboard")

    c.showPage()
    c.save()
    buffer.seek(0)
    return send_file(
        buffer,
        download_name="thermal_results.pdf",
        as_attachment=True,
        mimetype="application/pdf"
    )


 

if __name__ == "__main__":
    app.run(debug=True)