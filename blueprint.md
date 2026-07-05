# Role & Objective
You are an expert Python GUI Developer. Your task is to build a Desktop Application using PySide6 based on the specifications below. You must adhere to strict software engineering principles, modular code design, and comprehensive documentation.

---

## 🚫 STRICT CODE LAWS (CRITICAL)

1. **COMPREHENSIVE DOCUMENTATION REQUIRED**
   Every time you generate or modify code, you MUST provide or update a `README.md` or a comprehensive documentation block. This documentation must explicitly cover:
   - **Design Architecture:** High-level overview of how components interact.
   - **Tech Stack & Dependencies:** List of libraries used and their versions.
   - **Environment Setup:** Step-by-step guide to prepare the python environment.
   - **How to Run/Deploy:** Clear instructions on how to execute and package the application.

2. **MODULAR ARCHITECTURE (NO MONOLITHIC CODE)**
   - DO NOT write all code in a single, long file.
   - Code must be cleanly separated into distinct components/modules (e.g., Views/Pages, Custom Widgets, Business Logic/Model Mock, Utilities).
   - Adhere to the Single Responsibility Principle (SRP). Keep files concise and focused.

---

## 🛠 TECH STACK & CORE MODULES
- **Language:** Python 3.10+
- **GUI Framework:** PySide6 (Qt for Python)
- **Image Processing:** Pillow (PIL)
- **Core Qt Modules to use:** - `QWizard` & `QWizardPage` for the workflow.
  - `QGraphicsView` & `QGraphicsScene` for interactive image canvas.
  - `QGraphicsEllipseItem` (with `ItemIsMovable` flag) for draggable control points.

---

## 🗺 PROJECT DIRECTORY STRUCTURE GUIDE
You must structure the project cleanly, for example:
├── app.py                  # Application entry point
├── config.py               # Configuration and constants
├── README.md               # Mandatory Documentation
├── modules/
│   ├── __init__.py
│   ├── wizard.py           # Main QWizard implementation
│   ├── pages.py            # Individual QWizardPages (Page 1 to 4)
│   ├── canvas.py           # Custom QGraphicsView / QGraphicsScene for interaction
│   ├── model_mock.py       # Mock function for Image Model (Segmentation/Points)
│   └── utils.py            # File I/O, JSON export handlers

---

## 🎛 FEATURE & WORKFLOW REQUIREMENTS

The application must operate as a Step-by-Step Wizard with 4 distinct pages:

### Page 1: Image Loader
- A page containing a prominent area for loading an image.
- **Must support:** A "Browse" button AND native **Drag and Drop** functionality to drop an image file into the window.
- Dynamically display the loaded image. Enable the "Next" button only after a valid image is loaded.

### Page 2: Model Inference & Visualization
- Upon entering this page, trigger a mock inference function (simulating an AI model) that inputs the image and outputs:
  1. Segmentation Area data (a mask).
  2. Control Points data (a list of X, Y coordinates).
- **Visualization requirements using Graphics View Framework:**
  - Base Layer: The original image.
  - Overlay Layer: The segmentation area rendered with a semi-transparent color (adjustable opacity/alpha).
  - Interaction Layer: The control points rendered as distinct visual circles overlaying the image.

### Page 3: Interactive Adjustment
- Display the same canvas from Page 2.
- **User Interaction:** The user must be able to click and drag the control point circles using the mouse to adjust their positions manually.
- The system must track the updated coordinates in real-time.

### Page 4: Export Data
- Display a summary of the finalized control points.
- Provide an "Export" button to save the control point coordinates into a structured **JSON file**.
- Show a success confirmation message upon successful export.

---

## 🚀 START CODING
Please begin by generating the **Project Documentation (`README.md`)** first as requested in the laws, followed by the modular code structure step-by-step.