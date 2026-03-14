Thermal Analysis Dashboard

A Flask-based thermal analysis web dashboard for analyzing heat sink performance and processor thermal behavior.

The system calculates thermal resistance, junction temperature, and heat sink efficiency, and visualizes results with interactive charts.

It also includes a Physics-Informed Neural Network (PINN) model to predict the temperature distribution along the heat sink fin.

Features

• Processor heat sink thermal analysis
• Junction temperature calculation
• Heat sink resistance calculation
• Interactive charts using Plotly
• Heat sink fin visualization
• Physics-Informed Neural Network (PINN) for fin temperature profile
• CSV report export
• PDF report export
• Modern responsive dashboard UI

Technologies Used
Backend

Python

Flask

Machine Learning

PyTorch (PINN model)

Data Processing

Pandas

NumPy

Visualization

Plotly.js

Report Generation

ReportLab (PDF)

Frontend

HTML

CSS

JavaScript

Installation
1 Install Python

Make sure Python 3.9 or later is installed.

Check with:

python --version
2 Clone the Repository
git clone https://github.com/bruntha24/Thermal_project.git
cd Thermal_project
3 Create Virtual Environment (Recommended)
python -m venv venv

Activate it:

Windows
venv\Scripts\activate
Mac / Linux
source venv/bin/activate
4 Install Required Packages
pip install flask pandas torch reportlab

Or install from requirements file:

pip install -r requirements.txt
Required Python Libraries

The project requires the following Python packages:

flask
pandas
torch
reportlab

Optional but recommended:

numpy
Running the Application

Start the Flask server:

python app.py

The application will run at:

http://127.0.0.1:5000

Open this URL in your browser.

How It Works

User enters:

Processor Power

Number of fins

Air velocity

System calculates:

Heat sink resistance

Total thermal resistance

Junction temperature

A Physics-Informed Neural Network (PINN) solves the fin heat transfer equation:

𝑑
2
𝑇
𝑑
𝑥
2
−
ℎ
𝑃
𝑘
𝐴
(
𝑇
−
𝑇
∞
)
=
0
dx
2
d
2
T
	​

−
kA
hP
	​

(T−T
∞
	​

)=0

Results are visualized using interactive charts.

Charts in Dashboard

The dashboard displays:

Junction Temperature vs Number of Fins

Heat Sink Resistance vs Number of Fins

Total Thermal Resistance vs Air Velocity

Junction Temperature Distribution

Fin Temperature Profile (PINN)

Export Features

Users can download:

CSV Report

Contains calculated thermal results.

PDF Report

Includes:

Heat sink resistance

Junction temperature
Thermal_project
│
├── app.py
├── README.md
├── requirements.txt

number of fins

thermal gradient visualization 
