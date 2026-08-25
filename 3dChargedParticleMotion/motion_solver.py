import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import streamlit as st

st.set_page_config(page_title="B-field Inhomogeneity & Particle Trajectory", layout="wide")
st.title("Field Inhomogeneity (κ = L/r_L) + Charged Particle Trajectory")

st.markdown(
    "Field model: `B = Curvdrift·Bθ(r)·θ̂ + graddrift·B0_strength·x·ẑ`, "
    "with `Bθ(r) = μ0·I/(2πr)` (azimuthal, from a straight wire) and a **linear** "
    "axial gradient `Bz(x) = B0_strength·x` (models the field near a neutral line / "
    "current sheet, where the gradient scale length `L = |B|/|∇B| = x`)."
)

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Particle initial state")
x0x = st.sidebar.number_input("x0 (position, sets L = x0)", value=5.0, min_value=0.1, step=0.5)
x0y = st.sidebar.number_input("y0", value=0.0, step=0.5)
x0z = st.sidebar.number_input("z0", value=0.0, step=0.5)
theta = st.sidebar.slider("pitch angle θ (rad)", 0.0, np.pi, np.pi/2, 0.05)
phi = st.sidebar.slider("gyrophase φ (rad)", 0.0, 2*np.pi, np.pi/4, 0.05)
KE = st.sidebar.number_input("kinetic energy KE", value=1.0, min_value=1e-6, step=0.1)

st.sidebar.header("Particle properties")
q = st.sidebar.number_input("charge q", value=1.0, step=0.1)
m = st.sidebar.number_input("mass m", value=1.0, min_value=1e-6, step=0.1)
mu0 = st.sidebar.number_input("μ0 (permeability)", value=1.0, step=0.1)

st.sidebar.header("Inhomogeneity / adiabaticity")
st.sidebar.caption(
    "κ = L / r_L at x0, where L = x0 (gradient scale length) and r_L is the "
    "Larmor radius evaluated at x0. κ ≫ 1 → smooth adiabatic drift. "
    "κ ~ 1 → chaotic, non-adiabatic (Speiser-like) scattering."
)
kappa = st.sidebar.slider("κ (adiabaticity parameter)", 0.1, 50.0, 5.0, 0.1)

st.sidebar.header("Field components")
I_wire = st.sidebar.number_input("I (wire current, sets Bθ)", value=5.0, step=0.5)
Curvdrift = st.sidebar.checkbox("Include azimuthal Bθ (curvature term)", value=False)
graddrift = st.sidebar.checkbox("Include axial Bz gradient (grad-B term)", value=True)

st.sidebar.header("Domain / grid")
x_i, x_y = st.sidebar.slider("X range", 0.0, 20.0, (0.0, 10.0), 0.5)
y_i, y_y = st.sidebar.slider("Y range", 0.0, 20.0, (0.0, 10.0), 0.5)
z_i, z_y = st.sidebar.slider("Z range", -5.0, 5.0, (0.0, 1.0), 0.1)
resolution_xy = st.sidebar.slider("XY grid resolution", 20, 150, 100, 10)
resolution_z = st.sidebar.slider("Z grid resolution", 20, 300, 100, 10)

st.sidebar.header("Electric field (uniform)")
Ex = st.sidebar.number_input("E_x", value=0.0, step=0.1)
Ey = st.sidebar.number_input("E_y", value=0.0, step=0.1)
Ez = st.sidebar.number_input("E_z", value=0.0, step=0.1)

st.sidebar.header("Pusher / integration")
pusher = st.sidebar.radio("Pusher", ["Boris (recommended)", "Simple (explicit Euler)"])
boris_pusher = pusher.startswith("Boris")
timesteps = st.sidebar.slider("timesteps", 100, 200000, 20000, 100)
dt = st.sidebar.select_slider(
    "dt", options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2], value=1e-3,
)

run = st.sidebar.button("Run simulation", type="primary")

# ----------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------
def r_of(x, y):
    return np.sqrt(x**2 + y**2) + 1e-5

def Bth_of(r_, mu0, I_wire):
    return I_wire * mu0 / (2 * np.pi * r_)

def Bx_of(Bth_, r_, y):
    return -Bth_ * y / r_

def By_of(Bth_, r_, x):
    return Bth_ * x / r_

def make_B_field(B0_strength, mu0, I_wire, Curvdrift, graddrift):
    def B_field(x, y):
        rr = r_of(x, y)
        bth = Bth_of(rr, mu0, I_wire)
        return np.array([
            Curvdrift * Bx_of(bth, rr, y),
            Curvdrift * By_of(bth, rr, x),
            graddrift * B0_strength * x,
        ])
    return B_field

def make_v0(x0, B_field, theta, phi, KE, m):
    B_vec = np.asarray(B_field(x0[0], x0[1]), dtype=float)
    b_hat = B_vec / np.linalg.norm(B_vec)

    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, b_hat)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(b_hat, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(b_hat, e1)

    speed = np.sqrt(2 * KE / m)
    v_par = speed * np.cos(theta)
    v_perp = speed * np.sin(theta)

    v0 = v_par * b_hat + v_perp * (np.cos(phi) * e1 + np.sin(phi) * e2)
    return v0, v_perp

@st.cache_data(show_spinner=False)
def compute_grid(B0_strength, mu0, I_wire, Curvdrift, graddrift,
                  x_i, x_y, y_i, y_y, z_i, z_y, resolution_xy, resolution_z):
    x_dense = np.linspace(x_i, x_y, resolution_xy)
    y_dense = np.linspace(y_i, y_y, resolution_xy)
    z_dense = np.linspace(z_i, z_y, resolution_z)

    X_d, Y_d, Z_d = np.meshgrid(x_dense, y_dense, z_dense)  # shape (ny, nx, nz)
    rr = r_of(X_d, Y_d)
    bth = Bth_of(rr, mu0, I_wire)
    U_d = Curvdrift * Bx_of(bth, rr, Y_d)
    V_d = Curvdrift * By_of(bth, rr, X_d)
    W_d = graddrift * B0_strength * X_d
    return x_dense, y_dense, z_dense, X_d, Y_d, Z_d, U_d, V_d, W_d

@st.cache_data(show_spinner=False)
def simulate(B0_strength, mu0, I_wire, Curvdrift, graddrift, x0, theta, phi, KE,
             q, m, E, timesteps, dt, boris_pusher, bound):
    B_field = make_B_field(B0_strength, mu0, I_wire, Curvdrift, graddrift)
    E = np.asarray(E, dtype=float)

    v0, v_perp = make_v0(x0, B_field, theta, phi, KE, m)
    v = v0.copy()
    x = np.array(x0, dtype=float)
    xmem = [x.copy()]
    diverged_at = None

    if not boris_pusher:
        for _ in range(timesteps):
            a = q / m * np.cross(v, B_field(x[0], x[1]))
            v = v + a * dt
            x = x + v * dt
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > bound:
                diverged_at = len(xmem)
                break
            xmem.append(x.copy())
    else:
        for _ in range(timesteps):
            B = np.asarray(B_field(x[0], x[1]))
            t = q / m * B * 0.5 * dt
            t_mag2 = np.dot(t, t)
            s = 2. * t / (1. + t_mag2)

            v_minus = v + q / m * E * 0.5 * dt
            v_prime = v_minus + np.cross(v_minus, t)
            v_plus = v_minus + np.cross(v_prime, s)
            v = v_plus + q / m * E * 0.5 * dt
            x = x + v * dt

            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > bound:
                diverged_at = len(xmem)
                break
            xmem.append(x.copy())

    matrix = np.vstack(xmem)
    return matrix[:, 0], matrix[:, 1], matrix[:, 2], diverged_at, v_perp

def colored_line(x, y, c, ax=None, **lc_kwargs):
    xy = np.stack((x, y), axis=-1)
    xy_mid = np.concatenate(
        (xy[0, :][None, :], (xy[:-1, :] + xy[1:, :]) / 2, xy[-1, :][None, :]), axis=0
    )
    segments = np.stack((xy_mid[:-1, :], xy, xy_mid[1:, :]), axis=-2)
    lc_kwargs["array"] = c
    lc = LineCollection(segments, **lc_kwargs)
    ax = ax or plt.gca()
    ax.add_collection(lc)
    return lc

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if run:
    x0 = np.array([x0x, x0y, x0z])

    # --- solve B0_strength from the requested kappa ---
    speed = np.sqrt(2 * KE / m)
    v_perp_est = speed * np.sin(theta)
    L = x0x  # gradient scale length at x0, since B = B0_strength*x -> L = x
    if v_perp_est <= 0:
        st.error("v_perp = 0 (theta is 0 or pi): κ cannot control the field via a "
                 "purely perpendicular Larmor radius. Choose a pitch angle away from 0/π.")
        st.stop()

    B0_strength = kappa * v_perp_est * m / (q * x0x**2)
    B_at_x0 = B0_strength * x0x
    r_L = v_perp_est * m / (q * B_at_x0)   # should equal L/kappa, sanity-checked below

    st.info(
        f"Solved **B0_strength = {B0_strength:.5f}** to realize κ = {kappa:.2f} "
        f"at x0 = {x0x:.2f}  (L = {L:.3f}, r_L = {r_L:.4f}, check L/r_L = {L/r_L:.3f})"
    )

    with st.spinner("Computing field grid and trajectory..."):
        x_dense, y_dense, z_dense, X_d, Y_d, Z_d, U_d, V_d, W_d = compute_grid(
            B0_strength, mu0, I_wire, Curvdrift, graddrift,
            x_i, x_y, y_i, y_y, z_i, z_y, resolution_xy, resolution_z,
        )

        ix0 = np.argmin(np.abs(x_dense))
        iy0 = np.argmin(np.abs(y_dense))
        iz0 = np.argmin(np.abs(z_dense))

        E = (Ex, Ey, Ez)
        bound = 5 * max(x_y, y_y, abs(z_i), abs(z_y), 1.0)

        x1, y1, z1, diverged_at, v_perp = simulate(
            B0_strength, mu0, I_wire, Curvdrift, graddrift, x0, theta, phi, KE,
            q, m, E, timesteps, dt, boris_pusher, bound,
        )

        if diverged_at is not None:
            st.warning(
                f"Trajectory diverged / left the domain after {diverged_at} steps "
                f"and was truncated. Try a smaller dt or larger κ."
            )

    n_steps = len(x1)

    # ---------------- 2D planes ----------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    img = ax1.imshow((U_d[:, :, iz0]**2 + V_d[:, :, iz0]**2)**.5, cmap='RdPu',
                      extent=[x_i, x_y, y_i, y_y], origin='lower')
    fig.colorbar(img, ax=ax1, label='|B| (XY)')
    ax1.streamplot(X_d[:, :, iz0], Y_d[:, :, iz0], U_d[:, :, iz0], V_d[:, :, iz0],
                    linewidth=0.5, color='k')
    colored_line(x1, y1, range(n_steps), ax=ax1, cmap='viridis')
    ax1.set_xlim(x_i, x_y); ax1.set_ylim(y_i, y_y)
    ax1.set_title('XY plane'); ax1.set_xlabel('X'); ax1.set_ylabel('Y')

    Xxz, Zxz = X_d[iy0, :, :].T, Z_d[iy0, :, :].T
    Uxz, Wxz = U_d[iy0, :, :].T, W_d[iy0, :, :].T
    img = ax2.imshow((Uxz**2 + Wxz**2)**.5, cmap='RdPu',
                      extent=[x_i, x_y, z_i, z_y], origin='lower')
    fig.colorbar(img, ax=ax2, label='|B| (XZ)')
    ax2.streamplot(Xxz, Zxz, Uxz, Wxz, linewidth=0.5, color='k')
    colored_line(x1, z1, range(n_steps), ax=ax2, cmap='viridis')
    ax2.set_xlim(x_i, x_y); ax2.set_ylim(z_i, z_y)
    ax2.set_title('XZ plane'); ax2.set_xlabel('X'); ax2.set_ylabel('Z')
    ax2.set_aspect('auto')

    Yyz, Zyz = Y_d[:, ix0, :].T, Z_d[:, ix0, :].T
    Vyz, Wyz = V_d[:, ix0, :].T, W_d[:, ix0, :].T
    img = ax3.imshow((Vyz**2 + Wyz**2)**.5, cmap='RdPu',
                      extent=[y_i, y_y, z_i, z_y], origin='lower')
    fig.colorbar(img, ax=ax3, label='|B| (YZ)')
    ax3.streamplot(Yyz, Zyz, Vyz, Wyz, linewidth=0.5, color='k')
    colored_line(y1, z1, range(n_steps), ax=ax3, cmap='viridis')
    ax3.set_xlim(y_i, y_y); ax3.set_ylim(z_i, z_y)
    ax3.set_title('YZ plane'); ax3.set_xlabel('Y'); ax3.set_ylabel('Z')
    ax3.set_aspect('auto')

    fig.tight_layout()
    st.pyplot(fig)

    # ---------------- 3D trajectory ----------------
    fig3d = plt.figure(figsize=(7, 6))
    ax = fig3d.add_subplot(111, projection='3d')
    ax.plot3D(x1, y1, z1, 'green')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title('3D particle trajectory')
    st.pyplot(fig3d)

    with st.expander("Trajectory data / stats"):
        st.write(f"Steps completed: {n_steps} / {timesteps}")
        st.write(f"Final position: ({x1[-1]:.3f}, {y1[-1]:.3f}, {z1[-1]:.3f})")
        st.write(f"v_perp (from θ, KE): {v_perp:.4f}")
        st.write(f"κ = {kappa:.3f}, L = {L:.3f}, r_L(x0) = {r_L:.4f}, B0_strength = {B0_strength:.5f}")
        st.write(f"Pusher used: {'Boris' if boris_pusher else 'Simple (Euler)'}")
else:
    st.info("Set your parameters in the sidebar, then click **Run simulation**.")