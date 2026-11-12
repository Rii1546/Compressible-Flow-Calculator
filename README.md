# Compressible Flow Calculator: Design, Verification and Extension

## 1. Project Overview

The Compressible Flow Calculator is a Python-based engineering application developed to calculate and analyse different compressible flow properties for a calorically perfect gas.

The project implements important gas dynamics relations used in aerospace engineering, including isentropic flow, normal shocks, oblique shocks, Prandtl–Meyer expansion, Rayleigh flow and Fanno flow.

The calculator provides numerical solutions, inverse calculations, validation checks and graphical representations of compressible flow behaviour.

## 2. Objectives

- To develop a reliable Python-based compressible flow calculator.
- To review and correct errors in the original implementation.
- To implement additional compressible flow relations and inverse solvers.
- To analyse shock waves and expansion fans.
- To implement Rayleigh and Fanno flow calculations.
- To verify numerical results using published reference tables and physical identities.
- To generate graphical representations of different compressible flow properties.

## 3. Features

The project includes the following modules:

### A. Isentropic Flow
- Stagnation-to-static temperature ratio.
- Stagnation-to-static pressure ratio.
- Stagnation-to-static density ratio.
- Area–Mach number relation.
- Inverse area–Mach calculation.

### B. Normal Shock
- Downstream Mach number.
- Pressure ratio.
- Density ratio.
- Temperature ratio.
- Stagnation pressure loss.
- Entropy rise.
- Pitot pressure ratio.

### C. Oblique Shock
- Theta–Beta–Mach relation.
- Weak and strong shock solutions.
- Maximum flow deflection angle.
- Shock detachment detection.

### D. Prandtl–Meyer Expansion
- Prandtl–Meyer function.
- Maximum expansion angle.
- Inverse Mach number calculation.
- Expansion fan solver.

### E. Rayleigh Flow
- Heat addition and removal analysis.
- Flow property ratios.
- Thermal choking detection.
- Inverse calculations.

### F. Fanno Flow
- Frictional flow analysis.
- Flow property ratios.
- Friction parameter calculation.
- Friction choking detection.

## 4. Technologies and Libraries Used

- Python
- NumPy
- SciPy
- Matplotlib
- Pytest

## 5. Project Structure

The repository is organised into the following files:

```text
Compressible-Flow-Calculator/
│
├── compressible_flow.py
├── main.py
├── test_compressible_flow.py
├── test_main.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── figures/
    ├── Isentropic flow plots
    ├── Normal shock plots
    ├── Oblique shock plots
    ├── Prandtl-Meyer plots
    ├── Rayleigh flow plots
    └── Fanno flow plots
```

## 6. Installation and Requirements

Make sure Python is installed on your system.

Install the required libraries using:

```bash
pip install -r requirements.txt
```

Alternatively, install the dependencies individually:

```bash
pip install numpy scipy matplotlib pytest
```

## 7. How to Run the Program

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### Step 2: Open the Project Directory

```bash
cd <REPOSITORY_NAME>
```

### Step 3: Run Validation

```bash
python main.py --validate
```

### Step 4: Generate Graphical Outputs

```bash
python main.py --plots
```

### Step 5: Run a Calculation

For example, to calculate compressible flow properties at Mach 2.5:

```bash
python main.py --mach 2.5
```

### Step 6: Launch the Interactive Program

```bash
python main.py
```

## 8. Testing and Verification

The implementation was verified using automated testing and comparisons with published compressible flow reference tables.

The test suite contains:

- 108 library tests.
- 8 command-line interface tests.
- 116 total automated tests.

All 116 tests passed during verification.

The numerical results were checked against established compressible flow relations and physical identities.

## 9. Graphical Outputs

The program generates six graphical representations:

1. Isentropic flow properties.
2. Normal shock relations.
3. Theta–Beta–Mach diagram.
4. Prandtl–Meyer function.
5. Rayleigh flow properties.
6. Fanno flow properties.

These visualisations help understand how compressible flow properties vary with Mach number.

## 10. Scope and Limitations

The calculator is based on steady, one-dimensional or locally planar two-dimensional flow of a calorically perfect gas.

The current implementation does not include:

- Real-gas effects.
- Temperature-dependent specific heat ratio.
- Viscous boundary-layer effects.
- Shock–boundary-layer interactions.
- Conical shock solutions.

## 11. References

The theoretical relations and verification methods are based on established gas dynamics references, including:

1. J. D. Anderson – Modern Compressible Flow.
2. NACA Report 1135 – Equations, Tables, and Charts for Compressible Flow.
3. H. W. Liepmann and A. Roshko – Elements of Gasdynamics.
4. A. H. Shapiro – The Dynamics and Thermodynamics of Compressible Fluid Flow.

## 12. Project Report

The complete project report, including theoretical background, methodology, numerical results, graphical outputs and source-code appendices, is available in this repository as a PDF.

---

**Project Domain:** Aerospace Engineering – Gas Dynamics and Compressible Flow

**Programming Language:** Python