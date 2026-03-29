# 🏙️ AI-Driven Smart Campus Energy Management System

## Overview

A **MATLAB/Simulink Digital Twin** and **ESP32 hardware prototype** of an AI-powered Smart Campus Energy Management System (EMS), developed for **Hitachi INNOTHON 3.0** at NIT Warangal. The system integrates **ANN-based load and solar forecasting**, **intelligent BESS dispatch**, **real-time anomaly and fault detection**, and a **live financial impact dashboard** — validated on both simulation and physical hardware.

> 🏆 **4th Place (Consolation Prize) | ₹10,000 Cash Award | 28 Competing Teams**
> Team Jayanova | Hitachi INNOTHON 3.0 | NIT Warangal | March 2026

---

## 🛠️ Tools & Environment

| Tool | Version | Purpose |
|------|---------|---------|
| MATLAB | R2025a | ANN training, EMS logic, KPI computation |
| Simulink | R2025a | Digital Twin modeling & real-time dashboard |
| ESP32 | — | Hardware prototype & physical EMS validation |
| Arduino IDE | — | ESP32 firmware development |

---

## 🏗️ System Architecture

| Module | Description |
|--------|-------------|
| **ANN Load Forecaster** | 15-10 neuron feedforward network; features: 3-lag + sin/cos(hour) + day-of-week |
| **ANN Solar Forecaster** | 8-neuron feedforward network; features: 3-lag + sin/cos(hour) |
| **AI BESS Dispatch** | Rule-based intelligent dispatch using dynamic 75th-percentile peak threshold |
| **Battery Model** | 2 MWh Li-ion BESS; SOC limits 20–90%; η_charge=0.95, η_discharge=0.97 |
| **Anomaly Detector** | Statistical 3-sigma fault detection with 3-step sustained confirmation |
| **Financial Engine** | Peak demand savings, energy savings, payback period, CO₂ offset computation |
| **Hardware Prototype** | ESP32-based physical EMS validating Digital Twin dispatch decisions |
| **Real-time Dashboard** | 6-panel MATLAB live dashboard: forecasting, grid import, SOC, anomalies, financials |

---

## 📁 Project Structure

```
AI-Smart-Campus-EMS/
│
├── MATLAB/                  # MATLAB simulation scripts
│   └── Jayanova_Innothon.m  # Main EMS simulation file
├── Hardware/                # ESP32 hardware files
│   ├── firmware/            # Arduino IDE firmware (.ino)
│   ├── circuit_diagram/     # ESP32 wiring schematic
│   └── demo_video/          # Hardware working demo
├── Results/                 # Simulation outputs and plots
│   ├── AI_EMS_Dashboard.png # 6-panel KPI dashboard
│   ├── 24hr_operational_view.png
│   └── fault_detection_plot.png
└── Documentation/           # Report and presentation slides
```

---

## ▶️ How to Run

1. Open **MATLAB R2025a** (or compatible version)
2. Navigate to the `MATLAB/` folder
3. Open and run `Jayanova_Innothon.m`
4. MATLAB will automatically train ANNs, run EMS loop, and generate 4 figures
5. Final KPI summary prints to console on completion

> ⚠️ Requires **Deep Learning Toolbox** for ANN training (`fitnet`). Ensure it is installed via MATLAB Add-Ons Manager.

---

## 📊 Simulation Results

### ANN Forecast Performance
| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|------|
| Load Forecaster (15-10 ANN) | < 0.05 MW | < 0.07 MW | > 0.97 | < 3% |
| Solar Forecaster (8 ANN) | < 0.03 MW | < 0.04 MW | > 0.96 | < 4% |

- ✅ Load ANN: 3-lag + sinusoidal time encoding + day-of-week features
- ✅ Solar ANN: Captures diurnal generation profile accurately
- ✅ 70/15/15 train/validation/test split with fixed random seed (rng=42) for reproducibility

---

### AI EMS Technical Performance
| Metric | Value |
|--------|-------|
| Campus Load Range | 1.5 – 6.0 MW |
| Rooftop Solar Capacity | 0 – 1.0 MW |
| BESS Capacity | 2.0 MWh (SOC: 20–90%) |
| Peak Demand Reduction | ~18–22% |
| Battery Round-trip Efficiency | 92.15% (η_c × η_d) |
| Solar Utilization | 100% (solar < load at all times) |
| Simulation Duration | 60 days, hourly resolution (1440 points) |

- ✅ AI dispatch reduces grid peak demand through intelligent charge/discharge scheduling
- ✅ Dynamic peak threshold adapts to load profile (75th percentile)
- ✅ Battery SOC maintained within safe limits (20–90%) at all times
- ⚠️ Round-trip losses over 60 days computed and reported in KPI summary

---

### Anomaly & Fault Detection
| Metric | Value |
|--------|-------|
| Detection Method | 3-sigma statistical threshold on ANN prediction error |
| Fault Confirmation | 3-step sustained deviation check (reduces false positives) |
| Anomalies Detected | Reported per simulation run |
| Sustained Fault Events | Reported per simulation run |

- ✅ Detects sudden load spikes and abnormal consumption events
- ✅ 3-step confirmation avoids false alarms from momentary disturbances
- ✅ Fault events flagged and visualized on dedicated dashboard panel

---

### Financial & Carbon Impact
| Metric | Value |
|--------|-------|
| Electricity Tariff | Rs 400/kW (demand) + Rs 8/kWh (energy) |
| BESS Capital Cost | Rs 1.2 Crore |
| Peak Demand Savings | Computed over 60-day simulation |
| Annual Net Savings | Extrapolated from 60-day results |
| BESS Payback Period | Reported in KPI summary (years) |
| Solar CO₂ Offset | Computed using CEA emission factor: 0.82 kg CO₂/kWh |
| Battery-shift CO₂ Saving | Additional saving from BESS energy shifting |

- ✅ All financial metrics computed using Indian electricity tariff structure
- ✅ CO₂ impact quantified using India 2023 Central Electricity Authority emission factor
- ✅ Payback period calculated against realistic BESS capital investment

---

### Hardware Prototype Validation
| Component | Specification |
|-----------|--------------|
| Microcontroller | ESP32 (dual-core, Wi-Fi enabled) |
| Purpose | Physical validation of AI EMS dispatch decisions |
| Dashboard | Real-time display of EMS status, SOC, and power flow |
| Validation | Hardware results consistent with MATLAB Digital Twin outputs |

- ✅ ESP32 prototype validates the Digital Twin's energy dispatch logic in real hardware
- ✅ Real-time dashboard displays live EMS decisions and system status
- ✅ Hardware demo available in `Hardware/demo_video/`

---

## 🔑 Key Concepts Demonstrated

- Artificial Neural Network (ANN) design for time-series load and solar forecasting
- AI-driven Battery Energy Storage System (BESS) dispatch with dynamic peak shaving
- Statistical anomaly detection and sustained fault confirmation using 3-sigma method
- Digital Twin modeling in MATLAB/Simulink with real-time 6-panel KPI dashboard
- ESP32 hardware prototyping and firmware development for embedded EMS
- Financial modeling — demand charge savings, payback period, ROI analysis
- Carbon impact quantification using CEA India emission factors
- Smart campus energy optimization — peak demand reduction and solar integration

---

## 🏆 Competition

| Detail | Info |
|--------|------|
| Event | Hitachi INNOTHON 3.0 |
| Organizer | Hitachi Energy |
| Venue | NIT Warangal |
| Teams | 28 participating teams |
| Result | **4th Place — Consolation Prize** |
| Prize | **₹10,000 Cash Award** |
| Team Name | Jayanova |

---

## 👤 Author

**Sanmith G S**

- 🎓 M.Tech – Smart Electric Grid, **NIT Warangal**
- 🎓 B.E – Electrical & Electronics Engineering, **BMS College of Engineering, Bengaluru**
- 🔗 [LinkedIn](https://www.linkedin.com/in/sanmith-g-s)
- 🐙 [GitHub](https://github.com/SanmithGS)
- 📧 sannysanmith@gmail.com
