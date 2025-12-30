# 3-DOF CubeSat Detumble Simulation

A physics-based **3-degree-of-freedom CubeSat detumbling simulation** using **magnetorquer-only control**, a **B-dot controller**, **IGRF geomagnetic field modeling**, and **numerical integration of coupled orbital and attitude dynamics**.

This project models the post-deployment detumble phase of a CubeSat, where high initial body rates are damped using Earth’s magnetic field as the only control authority.

---

## Overview

After deployment, small satellites often experience large uncontrolled angular rates. This project simulates the detumbling process by combining:

* **Rigid-body rotational dynamics** (Euler equations)
* **Quaternion-based attitude kinematics**
* **Magnetorquer control** using a B-dot–based law (as described in the reference paper)
* **Earth magnetic field modeling** via IGRF (`ppigrf`)
* **Two-body orbital propagation** initialized from TLE data

The simulation integrates **orbit dynamics**, **geomagnetic field evaluation**, **control law execution**, and **attitude propagation** in a closed loop, producing time histories of body rates, commanded dipoles, actuator on-times, and orbital states.

---

## Key Features

* **3-DOF attitude dynamics** with quaternion state propagation (RK4)
* **Magnetorquer-only B-dot detumble controller**

  * Finite-difference estimate of magnetic field direction rate
  * Duty-cycled actuator on-time mapping with dipole saturation and polarity
* **Realistic geomagnetic field model**

  * IGRF via `ppigrf`
  * Proper ECI ↔ ECEF transformations using GMST
* **Orbit propagation**

  * Two-body gravitational model (RK4)
  * Initialization from **TLE → Kepler Elements → ECI r,v**
* **Logging + plotting**

  * Body rates, magnetic field magnitude, commanded dipoles, on-times
  * ECI position/velocity components and magnitudes
* **Performance instrumentation**

  * Timing breakdown for field evaluation, control, attitude, and orbit propagation

---

## Physical Model

### Attitude Dynamics (3-DOF)

The spacecraft is modeled as a rigid body with inertia matrix **I**. The attitude state is:

* Body angular rate **ω_B** (3×1)
* Quaternion **q_IB** (4×1, scalar-first)

Euler’s rotational equation is implemented as:

* **I ω̇ = −ω × (I ω) + τ_c**

Quaternion kinematics follow:

* **q̇ = 0.5 Ω(ω) q**, with normalization applied after propagation.

---

### Orbit Dynamics (Two-Body)

The orbit propagator integrates:

* **ṙ = v**
* **v̇ = −μ r / ‖r‖³**

Initialization path:

* **TLE → Kepler elements → ECI position/velocity**

---

### Magnetic Field Model (IGRF)

The magnetic field is computed using IGRF in a realistic frame chain:

1. **ECI position → ECEF** (GMST)
2. **ECEF → geocentric spherical** (r, θ, φ)
3. IGRF field evaluation in spherical components: *(B_r, B_θ, B_φ)*
4. Convert spherical → **ECEF Cartesian**
5. **ECEF → ECI**
6. Rotate **ECI → body** using quaternion transforms

The control law uses the body-frame magnetic field **B_E_B**.

---

## Control Law (B-dot Detumbling)

The controller implements a B-dot–style law consistent with the referenced paper:

* Compute unit vector **b̂ = B / ‖B‖**
* Estimate **b̂̇** via finite differences across the sample period **T_s**
* Command desired magnetic dipole moment:

Practical magnetorquer constraints are modeled via:

* Per-axis max dipole moments **m̄**
* Duty cycle **d**
* On-time computation **t_on** per axis
* Polarity-aware direction vector

Effective dipole over the sample is:

* **m_eff = direction ⊙ m̄ ⊙ (t_on / T_s)**

Torque is computed as:

* **τ_c = m_eff × B**

---

## Detumble Stop Condition

Detumbling is declared complete when:

* **‖ω_B‖ ≤ ω_stop**

for **N_w consecutive control samples**, where:

* ω_stop defaults to **2 deg/s**
* N_w is configured as a “confirmation window” length (in samples)

This prevents false-positive detumble detection due to transient dips in rate magnitude.

---

## Installation

### Dependencies

| Dependency   | Purpose                                 | Notes                                         |
| ------------ | --------------------------------------- | --------------------------------------------- |
| `numpy`      | Vector/matrix math, integration support | Core numeric dependency                       |
| `matplotlib` | Plotting simulation results             | Used in `sim.py`                              |
| `ppigrf`     | IGRF geomagnetic field model            | Used in `magnetic_field/model.py`             |
| `requests`   | Optional fetching of TLE text           | Present in code; not required for default run |

Install with:

```bash
pip install -r requirements.txt
```

---

## Running the Simulation

Default run executes a sample TLE, builds the orbit + attitude model, runs the detumble loop, and plots results.

```bash
python sim.py
```

If you use `maintest.py` as an entry point:

```bash
python maintest.py
```

---

## Output & Visualization

The run produces plots for:

* **‖ω_B‖** (body-rate magnitude) vs time
* ω_Bx, ω_By, ω_Bz vs time
* **‖B‖** vs time
* Commanded dipole **m_cmd** components vs time
* Duty-cycle on-time **t_on** components vs time
* ECI position/velocity components and magnitudes (|r|, |v|)

These time histories are also logged for analysis.

---

## Example Usage (Minimal)

```python
from datetime import datetime, timezone, timedelta
import numpy as np

from simulator.sim import build_sim

t0 = datetime.now(timezone.utc).replace(tzinfo=None)
sim, x_orbit0 = build_sim()

t_final = t0 + timedelta(seconds=3000)
result = sim.run(t0=t0, t_final=t_final, x_orbit0=x_orbit0)

print("Detumbled:", result["detumbled"])
print("Detumble time:", result["t_detumbled"])
```

---

## Validation Notes

Practical correctness checks built into the implementation:

* Quaternion normalization applied after each integration step to maintain **unit norm**
* Torque computed using **τ = m × B**, ensuring torque is orthogonal to the magnetic field direction
* Stop rule enforces *sustained* low-rate behavior rather than a single threshold crossing
* Logging of orbit state (**r, v**) enables sanity checking of propagation (e.g., |r|, |v| stability over short runs)

---

## References

* Example reference paper (controller formulation and detumble approach):
  [https://pure.tudelft.nl/ws/portalfiles/portal/47149549/IAC_18_C1_3_11_x46290.pdf](https://pure.tudelft.nl/ws/portalfiles/portal/47149549/IAC_18_C1_3_11_x46290.pdf)
* Geomagnetic field model: **International Geomagnetic Reference Field (IGRF)** via `ppigrf`
