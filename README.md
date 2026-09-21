# Contactless Gesture-Based Access Control System
### V-Model Systems Engineering, Multi-Node Interfacing & Embedded Vision

**Course:** ME561 - Mechatronics Systems Design & Interfacing  
**Instructor:** Dr. Ghaith Al-refai  
**Institution:** German Jordanian University (GJU)  
**Engineering Report:** `docs/GJU_Detailed_V_Model_Gesture_Access_System.docx`

---

## System Architecture & V-Model Methodology

This project implements a complete cyber-physical contactless security access system designed and verified strictly according to the **V-Model development lifecycle** (Stakeholder Needs $\to$ System Requirements $\to$ Architectural Decomposition $\to$ Verification & Validation).

```
        Stakeholder Needs                   System Acceptance
               \                                   /
       System Requirements                  System Integration
               \                                   /
       Subsystem Architecture              Subsystem Verification
               \                                   /
               Detailed Design -------- Unit Testing
```

### Hardware Subsystems

1. **Presence Sensing Subsystem (NI myRIO):**
   - Interfaced ultrasonic and infrared proximity sensors.
   - Operates as a low-power wake-up trigger, ensuring the camera vision subsystem runs only when a user is in close range.
2. **Vision & Authentication Node (Raspberry Pi):**
   - Real-time OpenCV video capture and hand gesture sequence recognition.
   - Matches dynamic gesture sequences against encrypted user access credentials.
   - Communicates via TCP/IP socket protocol to central access control servers.
3. **Actuation & Interlocking Subsystem (Industrial PLC & Relays):**
   - Receives deterministic digital triggers from the central controller.
   - Drives industrial door-strike relays with fail-safe timing and safety interlocks.

---

## Repository Structure

```text
├── docs/
│   └── GJU_Detailed_V_Model_Gesture_Access_System.docx # Full 20+ page V-model report
├── src/
│   ├── main_system.py           # Central Raspberry Pi orchestrator
│   ├── gesture_camera.py        # OpenCV camera capture and image preprocessing
│   ├── gesture_password.py      # Dynamic gesture sequence comparator
│   ├── plc_output.py            # Industrial PLC relay trigger interface
│   └── password_client.py       # TCP socket client communication
├── simulink/
│   ├── EV_HW.slx                # Electric vehicle longitudinal dynamic model
│   ├── Acceleration.slx         # Powertrain acceleration simulation
│   └── Torque_Function.mat      # Motor torque curve LUT
└── README.md
```