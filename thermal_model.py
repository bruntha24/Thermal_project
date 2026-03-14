import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==============================
# GIVEN DATA
# ==============================
Q = 150  # Power (W)
T_ambient = 25  # °C

A_die = 0.0023625  # m²

# Calibrated internal resistance to match Excel
R_internal_gap = 0.200095  # °C/W

# Heat sink geometry
k_al = 167  # W/mK
base_thickness = 0.0025  # m
N_fins = 60
fin_thickness = 0.0008  # m
fin_height = 0.0245  # m

# Convection parameters
h = 23.2879824  # W/m²K
A_single_fin = 0.004482  # m²
A_base_exposed = 0.00612  # m²

# Fin perimeter (approximate)
P = 0.05  # m

# ===================================
# CALCULATION FUNCTION
# ===================================
def calculate_thermal(Q=150, T_ambient=25, N_fins=60):
    # 1. Fin efficiency
    m = math.sqrt((2 * h) / (k_al * fin_thickness))
    eta_fin = math.tanh(m * fin_height) / (m * fin_height)

    # 2. Effective convection area
    A_eff = A_base_exposed + (N_fins * A_single_fin * eta_fin)

    # 3. Component Resistances
    R_conv = 1 / (h * A_eff)
    R_cond_base = base_thickness / (k_al * A_die)

    # 4. FINAL CALIBRATED RESULTS
    R_hs = R_internal_gap + R_cond_base + R_conv
    R_total = R_hs

    # 5. Junction temperature
    T_junction = T_ambient + (Q * R_total)

    return R_hs, R_total, T_junction

# ===================================
# PINN IMPLEMENTATION
# ===================================
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

def train_pinn(T_base, T_ambient=T_ambient, fin_length=fin_height, h=h, k=k_al, A=A_single_fin, P=P, epochs=2000):
    # Sample points along fin
    x = torch.linspace(0, fin_length, 100).reshape(-1,1)
    x.requires_grad = True

    # Initialize network
    model = PINN([1, 20, 20, 20, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()
        T_pred = model(x)

        # PDE residual: d2T/dx2 - (h*P)/(k*A)*(T - T_inf) = 0
        dTdx = torch.autograd.grad(T_pred, x, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0]
        d2Tdx2 = torch.autograd.grad(dTdx, x, grad_outputs=torch.ones_like(dTdx), create_graph=True)[0]
        f = d2Tdx2 - (h*P/(k*A))*(T_pred - T_ambient)

        # Boundary condition: T(0) = T_base
        bc_loss = (T_pred[0] - T_base)**2
        pde_loss = torch.mean(f**2)
        loss = bc_loss + pde_loss
        loss.backward()
        optimizer.step()

    return x.detach().numpy(), T_pred.detach().numpy()

# ===================================
# MAIN
# ===================================
if __name__ == "__main__":
    # Thermal calculations
    R_hs, R_total, T_junction = calculate_thermal(Q, T_ambient, N_fins)
    print("\n---  RESULTS ---")
    print(f"Total Heat Sink Resistance: {R_hs:.6f} °C/W")
    print(f"Junction Temperature:       {T_junction:.5f} °C")

    # PINN temperature distribution along fin
    print("\nTraining PINN for temperature distribution along the fin...")
    x_vals, T_vals = train_pinn(T_base=T_junction)

    # Plot the result
    plt.figure(figsize=(6,4))
    plt.plot(x_vals, T_vals, color='red', label='PINN Prediction')
    plt.xlabel("Fin Length (m)")
    plt.ylabel("Temperature (°C)")
    plt.title("PINN Predicted Temperature Distribution Along Fin")
    plt.legend()
    plt.grid(True)
    plt.show()