import numpy as np
import matplotlib.pyplot as plt
from mechanical_lib.mechanical_system import mechanical_system, model_params


#simulation parameters

N         = 4          # number of metronomes
M_cart    = 8.0        # cart mass (kg)
m_bob     = 0.25       # bob mass (kg)
g         = 9.81
epsilon   = 0.01

T_END     = 120        # max simulation time (s)
N_STEPS   = 6000       # RK4 steps
COH_THRESH = 0.9       # r(t) moving-average threshold for synchronisation

def generate_oscillators(N: int, seed: float = 42) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    lengths = np.array([0.1, 0.3, 0.1, 1.0, 0.0])
    initial_angles = np.random.uniform(-2, 2, N)
    initial_conditions = np.vstack((
        np.append(initial_angles, 0.0),   # angles + cart position
        np.zeros(N + 1)                    # all velocities zero
    ))
    return initial_conditions, lengths

SEED = 41
INITIAL_CONDITIONS, lengths = generate_oscillators(N, SEED)

params = model_params(m_bob, M_cart, lengths, g, epsilon)


# energy tracking

def run_with_energy(params, initial_conditions, tau_func, t_end, n_steps, coh_thresh):
    sim = mechanical_system(params, initial_conditions, tau_func)

    t0 = 0.0
    tf = t_end
    h  = (tf - t0) / n_steps

    sim.times  = [t0]
    sim.Y      = [initial_conditions]
    sim.orders = [sim.get_order(initial_conditions)]

    total_energy  = 0.0
    tau_prev      = sim.tau(initial_conditions[0], initial_conditions[1])[-1]
    x_dot_prev    = initial_conditions[1, -1]

    for _ in range(n_steps):
        t   = sim.times[-1]
        y_k = sim.Y[-1]

        m1 = sim.step(t,       y_k)
        m2 = sim.step(t + h/2, y_k + m1 * h/2)
        m3 = sim.step(t + h/2, y_k + m2 * h/2)
        m4 = sim.step(t + h,   y_k + m3 * h)
        y_kp1 = y_k + h/6 * (m1 + 2*m2 + 2*m3 + m4)

        sim.Y.append(y_kp1)
        sim.times.append(t + h)
        sim.orders.append(sim.get_order(y_kp1))

        tau_now   = sim.tau(y_kp1[0], y_kp1[1])[-1]
        x_dot_now = y_kp1[1, -1]
        total_energy += h / 2 * (tau_prev * x_dot_prev + tau_now * x_dot_now)
        tau_prev, x_dot_prev = tau_now, x_dot_now

    # computing moving average and finding coherence time
    sim.moving_average(n_steps // 4)
    avg = np.abs(sim.average_orders)

    if np.any(avg >= coh_thresh):
        coh_time = sim.times[np.argwhere(avg >= coh_thresh)[0][0]]
    else:
        coh_time = float(t_end)   # didnt synchronise within window

    return coh_time, abs(total_energy)


# PD control law (from cici's code)

def make_pd_tau(Kp, Kd, A=0.05, omega=3.13):
    """Returns a PD tau function with the given gains.
    omega=3.13 rad/s is the natural frequency of the longest pendulum (l=1m),
    chosen to maximise energy transfer through resonance (see Section 5.3.4)."""
    def tau(self, q, dqdt):
        t_vec = np.zeros(self.n)
        t_cur = self.times[-1]
        x      = q[-1]
        x_dot  = dqdt[-1]
        x_d    = A * np.sin(omega * t_cur)
        error  = x_d - x
        t_vec[-1] = Kp * error - Kd * x_dot
        return t_vec
    return tau


# FL control law (from cici's code)

def make_fl_tau(Kp_met, Kd_met, Kp_cart=2.0, Kd_cart=0.1, scale=0.1):
    def tau(self, q, dqdt):
        t_vec = np.zeros(self.n)
        q1  = q[0]
        dq1 = dqdt[0]

        v = np.zeros(self.n)
        for i in range(self.n - 1):
            v[i] = Kp_met * (q1 - q[i]) + Kd_met * (dq1 - dqdt[i])
        v[-1] = Kp_cart * (-scale * q1 - q[-1]) + Kd_cart * (-scale * dq1 - dqdt[-1])

        M_last = self.M(q)[-1, :]
        C_last = self.C(q, dqdt)[-1, :]
        G_last = self.G(q)[-1]

        t_vec[-1] = np.dot(M_last, v) + np.dot(C_last, dqdt) + G_last
        return t_vec
    return tau


# parameter sweeps

# PD, sweep over Kp, Kd, and A for 800 runs
pd_kp_values = np.linspace(5, 150, 10)
pd_kd_values = np.linspace(1, 80, 10)
pd_A_values = np.linspace(0.01, 0.2, 8)

# FL, 1024 runs
fl_Kp_met_vals = np.linspace(1, 50, 5)   # metronome proportional gain
fl_Kd_met_vals = np.linspace(1, 30, 5)   # metronome derivative gain
fl_Kp_cart_vals = np.linspace(0.5, 15, 5)   # cart proportional gain
fl_Kd_cart_vals = np.linspace(0.05, 2, 5)   # cart derivative gain
fl_scale_vals = np.linspace(0.01, 0.5, 5)  # cart reference scale

print("Running PD")
pd_results = []  # list of (Kp, Kd, A, t_sync, energy)
total_pd = len(pd_kp_values) * len(pd_kd_values) * len(pd_A_values)
pd_count = 0
for Kp in pd_kp_values:
    for Kd in pd_kd_values:
        for A in pd_A_values:
            tau_func = make_pd_tau(Kp, Kd, A=A)
            t_sync, E = run_with_energy(params, INITIAL_CONDITIONS.copy(), tau_func, T_END, N_STEPS, COH_THRESH)
            pd_results.append((Kp, Kd, A, t_sync, E))
            pd_count += 1
            if pd_count % 100 == 0:
                print(f"  {pd_count}/{total_pd} runs complete...")


total_fl = len(fl_Kp_met_vals) * len(fl_Kd_met_vals) * len(fl_Kp_cart_vals) * \
           len(fl_Kd_cart_vals) * len(fl_scale_vals)
print(f"\nRunning FL")
fl_results = []  # list of (Kp_met, Kd_met, Kp_cart, Kd_cart, scale, t_sync, energy)
count = 0
for Kp_met in fl_Kp_met_vals:
    for Kd_met in fl_Kd_met_vals:
        for Kp_cart in fl_Kp_cart_vals:
            for Kd_cart in fl_Kd_cart_vals:
                for scale in fl_scale_vals:
                    tau_func = make_fl_tau(Kp_met, Kd_met, Kp_cart, Kd_cart, scale)
                    t_sync, E = run_with_energy(params, INITIAL_CONDITIONS.copy(), tau_func, T_END, N_STEPS, COH_THRESH)
                    fl_results.append((Kp_met, Kd_met, Kp_cart, Kd_cart, scale, t_sync, E))
                    count += 1
                    if count % 100 == 0:
                        print(f"  {count}/{total_fl} runs complete...")

print(f"  {total_fl}/{total_fl} runs complete.")


# filtering out points that never synchronised

pd_synced = [r for r in pd_results if r[3] < T_END]   # r[3] is t_sync
fl_synced = [r for r in fl_results if r[5] < T_END]   # r[5] is t_sync

print(f"\nPD:  {len(pd_synced)}/{total_pd} runs synchronised")
print(f"FL:  {len(fl_synced)}/{total_fl} runs synchronised")


if pd_synced:
    pd_times_f    = [r[3] for r in pd_synced]
    pd_energies_f = [r[4] for r in pd_synced]
else:
    pd_times_f, pd_energies_f = [], [], []

if fl_synced:
    fl_times_f  = [r[5] for r in fl_synced]
    fl_energies_f = [r[6] for r in fl_synced]
else:
    fl_times_f, fl_energies_f = [], []


# finding the optimal points for the different priorities

def pareto_front_pd(points):
    pareto = []
    for r in points:
        t, e = r[3], r[4]
        dominated = any(
            r2[3] <= t and r2[4] <= e and (r2[3] < t or r2[4] < e)
            for r2 in points)
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda x: x[4])
    return pareto

def pareto_front_fl(points):
    pareto = []
    for r in points:
        t, e = r[5], r[6]
        dominated = any(
            r2[5] <= t and r2[6] <= e and (r2[5] < t or r2[6] < e)
            for r2 in points)
        if not dominated:
            pareto.append(r)
    pareto.sort(key=lambda x: x[6])
    return pareto

pd_pareto = pareto_front_pd(list(pd_synced))
fl_pareto = pareto_front_fl(fl_synced)


# printing recommendations

def normalised_distance(t, e, all_t, all_e):
    """Normalised Euclidean distance from origin after scaling both axes to [0,1],
    lower is better, rewards being simultaneously fast and energy efficient """
    t_range = max(all_t) - min(all_t)
    e_range = max(all_e) - min(all_e)
    t_norm = (t - min(all_t)) / t_range if t_range > 0 else 0
    e_norm = (e - min(all_e)) / e_range if e_range > 0 else 0
    return np.sqrt(t_norm**2 + e_norm**2)

def print_pd_recommendations(pareto):
    print(f"\n{'='*75}")
    print(f"  PD Control — Optimal Parameter Recommendations")
    print(f"{'='*75}")
    print(f"  {'Kp':>6}  {'Kd':>6}  {'A':>6}  {'T_sync (s)':>12}  {'Energy (J)':>12}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*12}")
    for r in pareto:
        print(f"  {r[0]:>6.1f}  {r[1]:>6.1f}  {r[2]:>6.3f}  {r[3]:>12.2f}  {r[4]:>12.4f}")

    all_t = [r[3] for r in pareto]
    all_e = [r[4] for r in pareto]
    best     = min(pareto, key=lambda x: normalised_distance(x[3], x[4], all_t, all_e))
    fastest  = min(pareto, key=lambda x: x[3])
    cheapest = min(pareto, key=lambda x: x[4])

    print(f"\n   Best balanced (min normalised distance from origin):")
    print(f"    Kp={best[0]:.1f}, Kd={best[1]:.1f}, A={best[2]:.3f}"
          f"  T={best[3]:.2f}s, E={best[4]:.4f} J")
    print(f"   Fastest:                  Kp={fastest[0]:.1f}, Kd={fastest[1]:.1f}, A={fastest[2]:.3f}"
          f"  T={fastest[3]:.2f}s, E={fastest[4]:.4f} J")
    print(f"   Lowest energy:            Kp={cheapest[0]:.1f}, Kd={cheapest[1]:.1f}, A={cheapest[2]:.3f}"
          f"  T={cheapest[3]:.2f}s, E={cheapest[4]:.4f} J")

def print_fl_recommendations(pareto):
    print(f"\n{'='*85}")
    print(f"  FL Control — Optimal Parameter Recommendations")
    print(f"{'='*85}")
    print(f"  {'Kp_met':>7}  {'Kd_met':>7}  {'Kp_cart':>8}  {'Kd_cart':>8}"
          f"  {'scale':>6}  {'T_sync (s)':>12}  {'Energy (J)':>12}")
    print(f"  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*12}  {'-'*12}")
    for r in pareto:
        print(f"  {r[0]:>7.2f}  {r[1]:>7.2f}  {r[2]:>8.3f}  {r[3]:>8.3f}"
              f"  {r[4]:>6.3f}  {r[5]:>12.2f}  {r[6]:>12.4f}")

    all_t = [r[5] for r in pareto]
    all_e = [r[6] for r in pareto]
    best = min(pareto, key=lambda x: normalised_distance(x[5], x[6], all_t, all_e))
    fastest = min(pareto, key=lambda x: x[5])
    cheapest = min(pareto, key=lambda x: x[6])

    print(f"\n   Best balanced (min normalised distance from origin):")
    print(f"    Kp_met={best[0]:.2f}, Kd_met={best[1]:.2f}, Kp_cart={best[2]:.3f},"
          f" Kd_cart={best[3]:.3f}, scale={best[4]:.3f}"
          f"  T={best[5]:.2f}s, E={best[6]:.4f} J")
    print(f"   Fastest:")
    print(f"    Kp_met={fastest[0]:.2f}, Kd_met={fastest[1]:.2f}, Kp_cart={fastest[2]:.3f},"
          f" Kd_cart={fastest[3]:.3f}, scale={fastest[4]:.3f}"
          f"  T={fastest[5]:.2f}s, E={fastest[6]:.4f} J")
    print(f"   Lowest energy:")
    print(f"    Kp_met={cheapest[0]:.2f}, Kd_met={cheapest[1]:.2f}, Kp_cart={cheapest[2]:.3f},"
          f" Kd_cart={cheapest[3]:.3f}, scale={cheapest[4]:.3f}"
          f"  T={cheapest[5]:.2f}s, E={cheapest[6]:.4f} J")

print_pd_recommendations(pd_pareto)
print_fl_recommendations(fl_pareto)


# plotting

fig, ax = plt.subplots(figsize=(9, 6))

# all synchronised points (faded background)
if pd_times_f:
    ax.scatter(pd_energies_f, pd_times_f, color="steelblue", alpha=0.25,
               s=35, zorder=2, label="_nolegend_")
if fl_times_f:
    ax.scatter(fl_energies_f, fl_times_f, color="tomato", alpha=0.15,
               s=25, marker="^", zorder=2, label="_nolegend_")

# optimal points in bold
if pd_pareto:
    ax.scatter([r[4] for r in pd_pareto], [r[3] for r in pd_pareto],
               color="#1A3E6B", s=90, zorder=4,
               label=f"PD Control — Optimal points ({len(pd_pareto)} pts)")

if fl_pareto:
    fl_p_t = [r[5] for r in fl_pareto]
    fl_p_e = [r[6] for r in fl_pareto]
    ax.scatter(fl_p_e, fl_p_t, color="#C0271A", s=90, marker="^", zorder=4,
               label=f"FL Control — Optimal points ({len(fl_pareto)} pts)")

# best balanced points using normalised distance metric
if pd_pareto:
    all_t_pd = [r[3] for r in pd_pareto]
    all_e_pd = [r[4] for r in pd_pareto]
    best_pd = min(pd_pareto, key=lambda x: normalised_distance(x[3], x[4], all_t_pd, all_e_pd))
    ax.scatter(best_pd[4], best_pd[3], color="#0A1628", s=280, marker="*", zorder=5,
               label=f"PD best balanced")

if fl_pareto:
    all_t_fl = [r[5] for r in fl_pareto]
    all_e_fl = [r[6] for r in fl_pareto]
    best_fl = min(fl_pareto, key=lambda x: normalised_distance(x[5], x[6], all_t_fl, all_e_fl))
    ax.scatter(best_fl[6], best_fl[5], color="#5C0A04", s=280, marker="*", zorder=5, label=f"FL best balanced point")


ax.set_xlabel("Control Energy  $E = \\int \\tau \\dot{x}\\, dt$  (J)", fontsize=12)
ax.set_ylabel("Synchronisation Time  $T_{\\mathrm{sync}}$ (s)", fontsize=12)
ax.set_title("Synchronisation Time vs Control Energy\n", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
plt.tight_layout()
plt.show()