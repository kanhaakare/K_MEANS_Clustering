
"""
Health Diagnosis Predictor - Desktop App
----------------------------------------
Install:   pip install customtkinter pandas numpy scikit-learn
Run:       python health_predictor_desktop.py
 
Place model.pkl and data.scaling.pkl in the same folder as this script.
"""
 
import pickle
import sys
from pathlib import Path
from tkinter import messagebox
 
import customtkinter as ctk
import pandas as pd
 
# =====================================================
# THEME
# =====================================================
ctk.set_appearance_mode("light")
 
BG = "#f0f7ff"
CARD = "#ffffff"
CARD_BORDER = "#dce9fb"
NAVY = "#15346b"
TITLE = "#173b75"
MUTED = "#7385a2"
LABEL = "#244577"
ACCENT = "#627cf6"
ACCENT_HOVER = "#4f67e0"
SIDEBAR_BG = "#e3f2ff"
 
HEALTHY_BG, HEALTHY_BORDER, HEALTHY_FG = "#d9f8eb", "#b9eed2", "#159765"
RISK_BG, RISK_BORDER, RISK_FG = "#ffebe8", "#ffd1c7", "#d85c48"
 
FONT = "Segoe UI"
 
BASE_DIR = Path(__file__).resolve().parent
 
FEATURE_NAMES = ["age", "bmi", "glucose", "insulin", "homa",
                 "leptin", "adiponectin", "resistin", "mcp_1"]
 
# (key, label, min, max, default)
SECTIONS = [
    ("👤  Basic Measurements", [
        ("age", "Age (years)", 1, 120, 50),
        ("bmi", "BMI", 5.0, 60.0, 25.0),
        ("glucose", "Glucose", 10.0, 300.0, 90.0),
    ]),
    ("🧬  Metabolic Measurements", [
        ("insulin", "Insulin", 0.1, 100.0, 6.0),
        ("homa", "HOMA", 0.1, 30.0, 1.5),
        ("leptin", "Leptin", 1.0, 150.0, 15.0),
    ]),
    ("🔬  Additional Biomarkers", [
        ("adiponectin", "Adiponectin", 1.0, 50.0, 10.0),
        ("resistin", "Resistin", 1.0, 100.0, 10.0),
        ("mcp_1", "MCP-1", 10.0, 2000.0, 400.0),
    ]),
]
 
 
# =====================================================
# MODEL LOADING
# =====================================================
def load_artifacts():
    model_path = BASE_DIR / "model.pkl"
    scaler_path = BASE_DIR / "data.scaling.pkl"
 
    if not model_path.is_file():
        raise FileNotFoundError(f"model.pkl is missing in {BASE_DIR}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"data.scaling.pkl is missing in {BASE_DIR}")
 
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler
 
 
def run_prediction(model, scaler, values):
    """Same pipeline as the Streamlit app. Returns (prediction, classes, probs)."""
    input_data = pd.DataFrame([values], columns=FEATURE_NAMES)
 
    n_expected = getattr(scaler, "n_features_in_", None)
    if n_expected is not None and n_expected != len(FEATURE_NAMES):
        raise ValueError(
            f"The scaler expects {n_expected} features, "
            f"but the app supplies {len(FEATURE_NAMES)}."
        )
 
    scaler_features = None
    if hasattr(scaler, "feature_names_in_"):
        scaler_features = list(scaler.feature_names_in_)
        missing = [n for n in scaler_features if n not in input_data.columns]
        if missing:
            raise ValueError(f"Missing scaler features: {missing}")
        scaler_input = input_data[scaler_features]
    else:
        scaler_input = input_data.to_numpy(dtype=float)
 
    scaled = scaler.transform(scaler_input)
 
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        if len(model_features) != scaled.shape[1]:
            raise ValueError(
                "The model and scaler expect different numbers of features. "
                "Check training."
            )
        cols = scaler_features if scaler_features else FEATURE_NAMES
        pred_input = pd.DataFrame(scaled, columns=cols)
        missing = [n for n in model_features if n not in pred_input.columns]
        if missing:
            raise ValueError(
                f"The model expects different feature names: {missing}."
            )
        pred_input = pred_input[model_features]
    else:
        pred_input = scaled
 
    prediction = model.predict(pred_input)[0]
 
    classes, probs = None, None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(pred_input)[0]
        classes = [str(c) for c in model.classes_]
 
    return prediction, classes, probs
 
 
# =====================================================
# APP
# =====================================================
class HealthApp(ctk.CTk):
    def __init__(self, model, scaler):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.entries = {}
        self.ranges = {}
 
        self.title("💙 Health Diagnosis Predictor")
        self.geometry("1280x820")
        self.minsize(1050, 700)
        self.configure(fg_color=BG)
 
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
 
        self._build_sidebar()
        self._build_main()
 
    # ---------------- Sidebar ----------------
    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=270, corner_radius=0,
                          fg_color=SIDEBAR_BG)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
 
        ctk.CTkLabel(sb, text="👋  Welcome!", font=(FONT, 22, "bold"),
                     text_color=TITLE).pack(anchor="w", padx=22, pady=(28, 6))
        ctk.CTkLabel(
            sb, justify="left", wraplength=225, text_color=LABEL,
            font=(FONT, 13),
            text="Explore clinical measurements using a machine-learning "
                 "classification model.",
        ).pack(anchor="w", padx=22)
 
        self._divider(sb)
 
        for title, caption in [
            ("🛡️  Health Insights", "Explore patterns in clinical measurements."),
            ("🧠  Machine Learning", "Obtain a prediction from your trained model."),
            ("📋  Easy to Use", "Enter values and view your model's output."),
        ]:
            ctk.CTkLabel(sb, text=title, font=(FONT, 14, "bold"),
                         text_color=TITLE).pack(anchor="w", padx=22, pady=(10, 0))
            ctk.CTkLabel(sb, text=caption, font=(FONT, 12), text_color=MUTED,
                         wraplength=225, justify="left"
                         ).pack(anchor="w", padx=22)
 
        self._divider(sb)
 
        ctk.CTkLabel(sb, text="💚  Healthy Habits", font=(FONT, 15, "bold"),
                     text_color=TITLE).pack(anchor="w", padx=22, pady=(0, 6))
        for line in ["🥗  Balanced nutrition", "🏃  Regular physical activity",
                     "💧  Adequate hydration", "🩺  Routine medical checkups"]:
            ctk.CTkLabel(sb, text=line, font=(FONT, 13),
                         text_color=LABEL).pack(anchor="w", padx=22, pady=1)
 
        ctk.CTkLabel(
            sb, justify="left", wraplength=225, font=(FONT, 11),
            text_color=MUTED,
            text="Educational use only. Avoid entering identifying "
                 "personal information.",
        ).pack(side="bottom", anchor="w", padx=22, pady=20)
 
    @staticmethod
    def _divider(parent):
        ctk.CTkFrame(parent, height=1, fg_color="#c5d9ee"
                     ).pack(fill="x", padx=22, pady=16)
 
    # ---------------- Main area ----------------
    def _build_main(self):
        self.main = ctk.CTkScrollableFrame(self, fg_color=BG,
                                           corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew", padx=(0, 4))
        self.main.grid_columnconfigure(0, weight=1)
 
        # Hero
        hero = ctk.CTkFrame(self.main, fg_color="#eaf5ff", corner_radius=22,
                            border_width=1, border_color="#dceafe")
        hero.grid(row=0, column=0, sticky="ew", padx=26, pady=(24, 14))
        ctk.CTkLabel(hero, text="💙  Health Diagnosis Predictor",
                     font=(FONT, 30, "bold"), text_color=NAVY
                     ).pack(anchor="w", padx=28, pady=(22, 2))
        ctk.CTkLabel(
            hero, font=(FONT, 14), text_color="#62789d",
            text="Better Insights   •   Smarter Health Screening   •   "
                 "Healthier Tomorrow",
        ).pack(anchor="w", padx=28, pady=(0, 22))
 
        # Form card
        card = ctk.CTkFrame(self.main, fg_color=CARD, corner_radius=20,
                            border_width=1, border_color=CARD_BORDER)
        card.grid(row=1, column=0, sticky="ew", padx=26, pady=8)
        card.grid_columnconfigure(0, weight=1)
 
        ctk.CTkLabel(card, text="🩺  Enter Your Clinical Details",
                     font=(FONT, 21, "bold"), text_color=TITLE
                     ).grid(row=0, column=0, sticky="w", padx=26, pady=(22, 0))
        ctk.CTkLabel(
            card, font=(FONT, 12), text_color=MUTED, justify="left",
            text="Enter all nine measurements using the same units and "
                 "definitions used by your training dataset.",
        ).grid(row=1, column=0, sticky="w", padx=26, pady=(2, 8))
 
        row = 2
        for section_title, fields in SECTIONS:
            ctk.CTkLabel(card, text=section_title, font=(FONT, 15, "bold"),
                         text_color=LABEL
                         ).grid(row=row, column=0, sticky="w",
                                padx=26, pady=(14, 4))
            row += 1
 
            grid = ctk.CTkFrame(card, fg_color="transparent")
            grid.grid(row=row, column=0, sticky="ew", padx=20, pady=(0, 4))
            for c in range(3):
                grid.grid_columnconfigure(c, weight=1, uniform="cols")
            row += 1
 
            for col, (key, label, lo, hi, default) in enumerate(fields):
                self._make_field(grid, col, key, label, lo, hi, default)
 
        # Buttons
        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.grid(row=row, column=0, sticky="ew", padx=26, pady=(18, 24))
        btn_row.grid_columnconfigure(0, weight=1)
 
        self.predict_btn = ctk.CTkButton(
            btn_row, text="💙  Predict Classification", height=52,
            corner_radius=12, font=(FONT, 17, "bold"),
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=self.on_predict,
        )
        self.predict_btn.grid(row=0, column=0, sticky="ew")
 
        ctk.CTkButton(
            btn_row, text="Reset", width=110, height=52, corner_radius=12,
            font=(FONT, 15, "bold"), fg_color="#eef2fb",
            hover_color="#dde5f7", text_color=LABEL,
            command=self.on_reset,
        ).grid(row=0, column=1, padx=(12, 0))
 
        # Result container
        self.result_frame = ctk.CTkFrame(self.main, fg_color="transparent")
        self.result_frame.grid(row=2, column=0, sticky="ew",
                               padx=26, pady=(8, 0))
        self.result_frame.grid_columnconfigure(0, weight=1)
 
        # Footer
        ctk.CTkLabel(
            self.main, font=(FONT, 11), text_color="#8191aa",
            text="💙 Health Diagnosis Predictor   |   Machine Learning   |   "
                 "Data-Driven Insights\n\nFor educational purposes only. "
                 "Not a substitute for professional medical advice or "
                 "diagnosis.",
        ).grid(row=3, column=0, pady=(24, 16))
 
    def _make_field(self, parent, col, key, label, lo, hi, default):
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=0, column=col, sticky="ew", padx=6, pady=4)
 
        ctk.CTkLabel(box, text=label, font=(FONT, 13, "bold"),
                     text_color=LABEL).pack(anchor="w")
        entry = ctk.CTkEntry(box, height=40, corner_radius=10,
                             border_color="#cfdcf1", fg_color="#fbfdff",
                             text_color=NAVY, font=(FONT, 14))
        entry.insert(0, str(default))
        entry.pack(fill="x", pady=(3, 1))
        entry.bind("<Return>", lambda e: self.on_predict())
 
        ctk.CTkLabel(box, text=f"Range: {lo:g} – {hi:g}", font=(FONT, 11),
                     text_color=MUTED).pack(anchor="w")
 
        self.entries[key] = entry
        self.ranges[key] = (label, lo, hi, default)
 
    # ---------------- Actions ----------------
    def on_reset(self):
        for key, entry in self.entries.items():
            entry.delete(0, "end")
            entry.insert(0, str(self.ranges[key][3]))
            entry.configure(border_color="#cfdcf1")
        self._clear_results()
 
    def _clear_results(self):
        for w in self.result_frame.winfo_children():
            w.destroy()
 
    def _read_values(self):
        """Validate inputs; returns list of values or None."""
        values, errors = [], []
        for key in FEATURE_NAMES:
            label, lo, hi, _ = self.ranges[key]
            entry = self.entries[key]
            entry.configure(border_color="#cfdcf1")
            try:
                v = float(entry.get().strip())
                if not lo <= v <= hi:
                    raise ValueError
                values.append(v)
            except ValueError:
                entry.configure(border_color="#e0584a")
                errors.append(f"• {label}: enter a number from {lo:g} to {hi:g}")
        if errors:
            messagebox.showwarning("Please check your inputs",
                                   "\n".join(errors))
            return None
        return values
 
    def on_predict(self):
        values = self._read_values()
        if values is None:
            return
        self._clear_results()
 
        try:
            prediction, classes, probs = run_prediction(
                self.model, self.scaler, values)
        except Exception as err:
            self._show_error(err)
            return
 
        self._show_result(prediction)
        if probs is not None:
            self._show_probabilities(classes, probs)
 
        # Scroll to the results
        self.after(100, lambda: self.main._parent_canvas.yview_moveto(0.85))
 
    # ---------------- Result rendering ----------------
    def _show_error(self, err):
        card = ctk.CTkFrame(self.result_frame, fg_color=RISK_BG,
                            border_width=1, border_color=RISK_BORDER,
                            corner_radius=18)
        card.grid(row=0, column=0, sticky="ew", pady=10)
        ctk.CTkLabel(card, text="❌  Prediction could not be completed.",
                     font=(FONT, 18, "bold"), text_color=RISK_FG
                     ).pack(anchor="w", padx=24, pady=(18, 4))
        ctk.CTkLabel(card, text=f"{type(err).__name__}: {err}",
                     font=(FONT, 12), text_color="#865950",
                     wraplength=850, justify="left"
                     ).pack(anchor="w", padx=24)
        ctk.CTkLabel(
            card, font=(FONT, 12), text_color="#865950",
            text="Check the feature names, feature order, scaler, and "
                 "model training pipeline.",
        ).pack(anchor="w", padx=24, pady=(8, 18))
 
    def _show_result(self, prediction):
        # NOTE: assumes class 1 = Healthy, anything else = Patient.
        # Confirm this mapping against your training dataset.
        healthy = str(prediction) == "1"
        bg, border, fg = ((HEALTHY_BG, HEALTHY_BORDER, HEALTHY_FG) if healthy
                          else (RISK_BG, RISK_BORDER, RISK_FG))
        sub = "#18764f" if healthy else "#a8483b"
        body_c = "#416c5c" if healthy else "#865950"
 
        ctk.CTkLabel(self.result_frame, text="📊  Prediction Result",
                     font=(FONT, 22, "bold"), text_color=TITLE
                     ).grid(row=0, column=0, sticky="w", pady=(8, 8))
 
        card = ctk.CTkFrame(self.result_frame, fg_color=bg, corner_radius=18,
                            border_width=1, border_color=border)
        card.grid(row=1, column=0, sticky="ew")
 
        ctk.CTkLabel(card, text="PREDICTION OUTCOME", font=(FONT, 12, "bold"),
                     text_color="#45617d").pack(anchor="w", padx=26,
                                                pady=(20, 0))
        ctk.CTkLabel(
            card, font=(FONT, 32, "bold"), text_color=fg,
            text="✅  Healthy Person" if healthy else "⚠️  Patient",
        ).pack(anchor="w", padx=26, pady=(4, 0))
        ctk.CTkLabel(card, text=f"Model predicted Class {prediction}",
                     font=(FONT, 14, "bold"), text_color=sub
                     ).pack(anchor="w", padx=26)
 
        if healthy:
            msg = (f"The model assigned your measurements to Class "
                   f"{prediction}. This is not a medical diagnosis. "
                   "Consult a healthcare professional for interpretation "
                   "of your results.")
        else:
            msg = (f"The model assigned your measurements to Class "
                   f"{prediction}. This does not confirm a disease. Please "
                   "consult a healthcare professional for further "
                   "evaluation.")
        ctk.CTkLabel(card, text=msg, font=(FONT, 13), text_color=body_c,
                     wraplength=880, justify="left"
                     ).pack(anchor="w", padx=26, pady=(10, 22))
 
    def _show_probabilities(self, classes, probs):
        card = ctk.CTkFrame(self.result_frame, fg_color=CARD,
                            corner_radius=18, border_width=1,
                            border_color=CARD_BORDER)
        card.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        card.grid_columnconfigure(1, weight=1)
 
        ctk.CTkLabel(card, text="📈  Model Class Probabilities",
                     font=(FONT, 18, "bold"), text_color=TITLE
                     ).grid(row=0, column=0, columnspan=3, sticky="w",
                            padx=24, pady=(18, 10))
 
        for i, (cls, p) in enumerate(zip(classes, probs), start=1):
            ctk.CTkLabel(card, text=f"Class {cls}", font=(FONT, 13, "bold"),
                         text_color=LABEL, width=80, anchor="w"
                         ).grid(row=i, column=0, padx=(24, 8), pady=6)
            bar = ctk.CTkProgressBar(card, height=16, corner_radius=8,
                                     progress_color=ACCENT,
                                     fg_color="#e6edfa")
            bar.set(float(p))
            bar.grid(row=i, column=1, sticky="ew", pady=6)
            ctk.CTkLabel(card, text=f"{p * 100:.1f}%", font=(FONT, 13),
                         text_color=LABEL, width=70, anchor="e"
                         ).grid(row=i, column=2, padx=(8, 24), pady=6)
 
        ctk.CTkLabel(
            card, font=(FONT, 11), text_color=MUTED, justify="left",
            text="These probabilities describe the model output; they are "
                 "not validated clinical risk estimates.",
        ).grid(row=len(classes) + 1, column=0, columnspan=3, sticky="w",
               padx=24, pady=(6, 18))
 
 
# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    try:
        model, scaler = load_artifacts()
    except Exception as exc:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Unable to load model",
                             f"Unable to load the model or scaler.\n\n{exc}")
        sys.exit(1)
 
    HealthApp(model, scaler).mainloop()
