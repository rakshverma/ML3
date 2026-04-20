from __future__ import annotations

import json
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from ml3.continuous import run_continuous_monitoring
from ml3.workflows import (
    discover_open_industrial_workflow,
    inspect_kgis_workflow,
    run_real_workflow,
)


def launch_desktop_ui() -> None:
    app = MonitoringDesktopApp()
    app.mainloop()


class MonitoringDesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.repo_root = Path(__file__).resolve().parent.parent
        self.demo_boundary = self.repo_root / "outputs" / "open_udyambag" / "industrial_candidates.geojson"
        self.demo_before_npz = self.repo_root / "data" / "open" / "site_001" / "before_scene.npz"
        self.demo_after_npz = self.repo_root / "data" / "open" / "site_001" / "after_scene.npz"
        self.demo_real_config = self.repo_root / "configs" / "open_candidate_run.json"
        self.demo_output_dir = self.repo_root / "outputs" / "desktop_run"
        self.demo_discovery_dir = self.repo_root / "outputs" / "open_ui_discovery"
        self.demo_history_path = self.demo_output_dir / "continuous_history.json"

        self.title("ML3 Environmental Monitoring Studio")
        self.geometry("1380x900")
        self.minsize(1120, 760)
        self.configure(bg="#edf2f7")

        self._setup_style()

        shell = ttk.Frame(self, style="App.TFrame", padding=(18, 14))
        shell.pack(fill="both", expand=True)

        hero = ttk.Frame(shell, style="Card.TFrame", padding=(18, 14))
        hero.pack(fill="x", pady=(0, 10))
        ttk.Label(
            hero,
            text="Industrial Compliance Desktop Studio",
            style="HeroTitle.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            hero,
            text=(
                "No JSON editing in UI. Choose boundary and satellite inputs, run analysis, and view human-readable output. "
                "Designed for desktop packaging (EXE/App)."
            ),
            style="HeroSub.TLabel",
            wraplength=1060,
        ).pack(anchor="w", pady=(4, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(shell, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(0, 8))

        notebook = ttk.Notebook(shell)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook

        self.inspect_text = None
        self.discovery_text = None
        self.analysis_text = None
        self.analysis_alerts_tree = None
        self.analysis_image_label = None
        self.analysis_image_caption_var = tk.StringVar(value="No evidence image generated yet.")
        self.analysis_evidence_paths: dict[str, str] = {}
        self.analysis_preview_image = None
        self.analysis_preview_choice_var = tk.StringVar(value="comparison_panel")
        self.analysis_preview_selector = None
        self.results_text = None
        self.results_alerts_tree = None
        self.results_image_label = None
        self.results_image_caption_var = tk.StringVar(value="No results loaded yet.")
        self.results_evidence_paths: dict[str, str] = {}
        self.results_preview_image = None
        self.results_preview_choice_var = tk.StringVar(value="comparison_panel")
        self.results_preview_selector = None
        self.result_site_var = tk.StringVar(value="-")
        self.result_window_var = tk.StringVar(value="-")
        self.result_threshold_var = tk.StringVar(value="-")
        self.result_mode_var = tk.StringVar(value="-")
        self.result_ml_var = tk.StringVar(value="-")
        self.result_before_var = tk.StringVar(value="-")
        self.result_after_var = tk.StringVar(value="-")
        self.result_delta_var = tk.StringVar(value="-")
        self.result_alerts_var = tk.StringVar(value="-")
        self.result_green_loss_var = tk.StringVar(value="-")
        self.result_new_const_var = tk.StringVar(value="-")
        self.results_narrative_text = None
        self.results_data_text = None
        self.results_ml_text = None
        self.results_tab = None
        self.results_canvas = None
        self.results_scrollable_frame = None
        self.continuous_text = None
        self.continuous_tree = None

        inspect_tab = ttk.Frame(notebook, style="App.TFrame", padding=12)
        discover_tab = ttk.Frame(notebook, style="App.TFrame", padding=12)
        analysis_tab = ttk.Frame(notebook, style="App.TFrame", padding=12)
        results_tab = ttk.Frame(notebook, style="App.TFrame", padding=12)
        continuous_tab = ttk.Frame(notebook, style="App.TFrame", padding=12)

        notebook.add(inspect_tab, text="Boundary Inspection")
        notebook.add(discover_tab, text="OSM Discovery")
        notebook.add(analysis_tab, text="Compliance Analysis")
        notebook.add(results_tab, text="Compliance Results")
        notebook.add(continuous_tab, text="Continuous Monitoring")

        self._build_inspect_tab(inspect_tab)
        self._build_discovery_tab(discover_tab)
        self._build_analysis_tab(analysis_tab)
        self._build_results_tab(results_tab)
        self._build_continuous_tab(continuous_tab)
        self.after(250, self._load_results_if_available)

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("App.TFrame", background="#edf2f7")
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("SectionTitle.TLabel", background="#ffffff", foreground="#0f172a", font=("SF Pro Text", 12, "bold"))
        style.configure("HeroTitle.TLabel", background="#ffffff", foreground="#0b3b5a", font=("SF Pro Display", 20, "bold"))
        style.configure("HeroSub.TLabel", background="#ffffff", foreground="#4b5563", font=("SF Pro Text", 10))
        style.configure("Body.TLabel", background="#ffffff", foreground="#111827", font=("SF Pro Text", 10))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#64748b", font=("SF Pro Text", 9))
        style.configure("Status.TLabel", background="#edf2f7", foreground="#0b3b5a", font=("SF Pro Text", 10, "bold"))
        style.configure("KPIValue.TLabel", background="#ffffff", foreground="#0f172a", font=("SF Pro Display", 13, "bold"))
        style.configure("KPITitle.TLabel", background="#ffffff", foreground="#64748b", font=("SF Pro Text", 9))

        style.configure("TNotebook", background="#edf2f7", borderwidth=0)
        style.configure("TNotebook.Tab", font=("SF Pro Text", 10, "bold"), padding=(14, 8))

        style.configure("Accent.TButton", foreground="#ffffff", background="#0b7a75", font=("SF Pro Text", 10, "bold"), borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#0d9488")])

    def _build_inspect_tab(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.pack(fill="both", expand=True)

        ttk.Label(card, text="Upload KGIS Boundary and Validate", style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(
            card,
            text="Required: .zip shapefile OR .shp OR .geojson/.json. Use polygon boundaries for compliance analysis.",
            style="Hint.TLabel",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 10))

        self.inspect_path_var = tk.StringVar()
        self.inspect_json_out_var = tk.StringVar(value=str(self.repo_root / "outputs" / "inspect_boundary_summary.json"))
        if self.demo_boundary.exists():
            self.inspect_path_var.set(str(self.demo_boundary))

        self._add_entry_row(card, 2, "Boundary file", self.inspect_path_var, self._pick_boundary_file)
        self._add_entry_row(card, 3, "Optional summary output", self.inspect_json_out_var, self._pick_json_output)

        ttk.Button(card, text="Inspect Boundary", style="Accent.TButton", command=self._run_inspection).grid(
            row=4, column=0, sticky="w", pady=(8, 10)
        )

        self.inspect_text = tk.Text(card, height=26, wrap="word", bg="#f8fafc", fg="#0f172a", relief="flat")
        self.inspect_text.grid(row=5, column=0, columnspan=4, sticky="nsew")
        card.rowconfigure(5, weight=1)
        card.columnconfigure(1, weight=1)

    def _build_discovery_tab(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.pack(fill="both", expand=True)

        ttk.Label(card, text="Discover Industrial Candidates from OpenStreetMap", style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(
            card,
            text="Search by region name. This creates GeoJSON and CSV for candidate industrial premises.",
            style="Hint.TLabel",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 10))

        self.discover_query_var = tk.StringVar(value="Udyambag, Belagavi, Karnataka, India")
        self.discover_output_var = tk.StringVar(value=str(self.demo_discovery_dir))
        self.discover_limit_var = tk.StringVar(value="25")
        self.discover_include_buildings_var = tk.BooleanVar(value=True)

        self._add_entry_row(card, 2, "Region query", self.discover_query_var)
        self._add_entry_row(card, 3, "Output folder", self.discover_output_var, self._pick_output_folder)
        self._add_entry_row(card, 4, "Candidate limit", self.discover_limit_var)

        ttk.Checkbutton(
            card,
            text="Include building=industrial features",
            variable=self.discover_include_buildings_var,
            style="Body.TLabel",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 6))

        ttk.Button(card, text="Run OSM Discovery", style="Accent.TButton", command=self._run_discovery).grid(
            row=6, column=0, sticky="w", pady=(6, 10)
        )

        self.discovery_text = tk.Text(card, height=24, wrap="word", bg="#f8fafc", fg="#0f172a", relief="flat")
        self.discovery_text.grid(row=7, column=0, columnspan=4, sticky="nsew")
        card.rowconfigure(7, weight=1)
        card.columnconfigure(1, weight=1)

    def _build_analysis_tab(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.pack(fill="both", expand=True)

        ttk.Label(card, text="Run Compliance Analysis (No JSON Editing)", style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=6, sticky="w")
        ttk.Label(
            card,
            text=(
                "Required inputs: boundary file (.zip/.shp/.geojson), before and after NPZ scenes with bands blue/green/red/nir/(optional swir), "
                "acquisition dates, and raster geotransform values."
            ),
            style="Hint.TLabel",
            wraplength=1100,
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(4, 10))

        self.site_id_var = tk.StringVar(value="SITE_001")
        self.site_name_var = tk.StringVar(value="Industrial Premises")
        self.required_green_var = tk.StringVar(value="15.0")
        self.enable_ml_var = tk.BooleanVar(value=True)
        self.boundary_path_var = tk.StringVar(value=str(self.demo_boundary) if self.demo_boundary.exists() else "")
        self.feature_index_var = tk.StringVar(value="0")
        self.boundary_crs_var = tk.StringVar(value="EPSG:4326")

        self.before_npz_var = tk.StringVar(value=str(self.demo_before_npz) if self.demo_before_npz.exists() else "")
        self.after_npz_var = tk.StringVar(value=str(self.demo_after_npz) if self.demo_after_npz.exists() else "")
        self.before_date_var = tk.StringVar(value="2024-01-24")
        self.after_date_var = tk.StringVar(value="2025-01-13")
        self.origin_x_var = tk.StringVar(value="74.48")
        self.origin_y_var = tk.StringVar(value="15.83")
        self.pixel_w_var = tk.StringVar(value="0.0001")
        self.pixel_h_var = tk.StringVar(value="0.0001")
        self.raster_crs_var = tk.StringVar(value="EPSG:4326")
        self.output_dir_var = tk.StringVar(value=str(self.demo_output_dir))

        self._add_entry_row(card, 2, "Site ID", self.site_id_var)
        self._add_entry_row(card, 3, "Site name", self.site_name_var)
        self._add_entry_row(card, 4, "Required green cover (%)", self.required_green_var)
        ttk.Checkbutton(
            card,
            text="Enable ML mode (train models from current run)",
            variable=self.enable_ml_var,
            style="Body.TLabel",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(2, 4))
        self._add_entry_row(card, 6, "Boundary file", self.boundary_path_var, self._pick_boundary_file_analysis)
        self._add_entry_row(card, 7, "Boundary feature index", self.feature_index_var)
        self._add_entry_row(card, 8, "Boundary CRS", self.boundary_crs_var)

        self._add_entry_row(card, 9, "Before NPZ", self.before_npz_var, self._pick_before_npz)
        self._add_entry_row(card, 10, "After NPZ", self.after_npz_var, self._pick_after_npz)
        self._add_entry_row(card, 11, "Before date (YYYY-MM-DD)", self.before_date_var)
        self._add_entry_row(card, 12, "After date (YYYY-MM-DD)", self.after_date_var)
        self._add_entry_row(card, 13, "Origin X", self.origin_x_var)
        self._add_entry_row(card, 14, "Origin Y", self.origin_y_var)
        self._add_entry_row(card, 15, "Pixel width", self.pixel_w_var)
        self._add_entry_row(card, 16, "Pixel height", self.pixel_h_var)
        self._add_entry_row(card, 17, "Raster CRS", self.raster_crs_var)
        self._add_entry_row(card, 18, "Output folder", self.output_dir_var, self._pick_output_folder_analysis)

        ttk.Button(card, text="Run Compliance Analysis", style="Accent.TButton", command=self._run_analysis).grid(
            row=19, column=0, sticky="w", pady=(8, 10)
        )

        evidence_panel = ttk.Frame(card, style="Card.TFrame")
        evidence_panel.grid(row=20, column=0, columnspan=6, sticky="ew", pady=(0, 10))
        ttk.Label(evidence_panel, text="Evidence Images", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(evidence_panel, textvariable=self.analysis_image_caption_var, style="Hint.TLabel").grid(
            row=0, column=1, columnspan=4, sticky="w", padx=(10, 0)
        )

        self.analysis_preview_selector = ttk.Combobox(
            evidence_panel,
            textvariable=self.analysis_preview_choice_var,
            values=["comparison_panel", "before_annotated_image", "after_annotated_image"],
            state="readonly",
            width=26,
        )
        self.analysis_preview_selector.grid(row=1, column=0, sticky="w", pady=(6, 8))
        self.analysis_preview_selector.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._set_analysis_image_preview(self.analysis_evidence_paths.get(self.analysis_preview_choice_var.get(), "")),
        )

        self.analysis_image_label = ttk.Label(evidence_panel, style="Body.TLabel")
        self.analysis_image_label.grid(row=2, column=0, columnspan=5, sticky="w")

        splitter = ttk.Panedwindow(card, orient="vertical")
        splitter.grid(row=21, column=0, columnspan=6, sticky="nsew")
        card.rowconfigure(21, weight=1)
        card.columnconfigure(1, weight=1)

        upper = ttk.Frame(splitter, style="Card.TFrame")
        lower = ttk.Frame(splitter, style="Card.TFrame")
        splitter.add(upper, weight=3)
        splitter.add(lower, weight=2)

        self.analysis_text = tk.Text(upper, wrap="word", bg="#f8fafc", fg="#0f172a", relief="flat")
        self.analysis_text.pack(fill="both", expand=True)

        ttk.Label(lower, text="Violation Alerts", style="SectionTitle.TLabel").pack(anchor="w")
        self.analysis_alerts_tree = ttk.Treeview(lower, columns=("severity", "type", "message", "coord"), show="headings", height=6)
        self.analysis_alerts_tree.heading("severity", text="Severity")
        self.analysis_alerts_tree.heading("type", text="Alert Type")
        self.analysis_alerts_tree.heading("message", text="Message")
        self.analysis_alerts_tree.heading("coord", text="Geo Coordinate")
        self.analysis_alerts_tree.column("severity", width=90, anchor="center")
        self.analysis_alerts_tree.column("type", width=180, anchor="w")
        self.analysis_alerts_tree.column("message", width=560, anchor="w")
        self.analysis_alerts_tree.column("coord", width=220, anchor="w")
        self.analysis_alerts_tree.pack(fill="both", expand=True)

    def _build_continuous_tab(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.pack(fill="both", expand=True)

        ttk.Label(card, text="Continuous Monitoring Runs", style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(
            card,
            text="Use a real-run config. The system repeats analysis and stores run history in a JSON file automatically.",
            style="Hint.TLabel",
            wraplength=1000,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 10))

        self.cont_config_var = tk.StringVar(value=str(self.demo_real_config) if self.demo_real_config.exists() else "")
        self.cont_iterations_var = tk.StringVar(value="3")
        self.cont_interval_var = tk.StringVar(value="0")
        self.cont_history_var = tk.StringVar(value=str(self.demo_history_path))

        self._add_entry_row(card, 2, "Real-run config", self.cont_config_var, self._pick_real_config)
        self._add_entry_row(card, 3, "Iterations", self.cont_iterations_var)
        self._add_entry_row(card, 4, "Interval seconds", self.cont_interval_var)
        self._add_entry_row(card, 5, "History output (optional)", self.cont_history_var, self._pick_history_output)

        ttk.Button(card, text="Run Continuous Monitoring", style="Accent.TButton", command=self._run_continuous).grid(
            row=6, column=0, sticky="w", pady=(8, 10)
        )

        self.continuous_text = tk.Text(card, height=10, wrap="word", bg="#f8fafc", fg="#0f172a", relief="flat")
        self.continuous_text.grid(row=7, column=0, columnspan=4, sticky="nsew", pady=(0, 8))

        self.continuous_tree = ttk.Treeview(
            card,
            columns=("seq", "site", "before", "after", "delta", "alerts"),
            show="headings",
            height=10,
        )
        for col, text, width in [
            ("seq", "Run #", 70),
            ("site", "Site", 180),
            ("before", "Before %", 100),
            ("after", "After %", 100),
            ("delta", "Delta pp", 100),
            ("alerts", "Alerts", 80),
        ]:
            self.continuous_tree.heading(col, text=text)
            self.continuous_tree.column(col, width=width, anchor="center")
        self.continuous_tree.grid(row=8, column=0, columnspan=4, sticky="nsew")

        card.rowconfigure(8, weight=1)
        card.columnconfigure(1, weight=1)

    def _build_results_tab(self, parent: ttk.Frame) -> None:
        self.results_tab = parent
        shell = ttk.Frame(parent, style="App.TFrame")
        shell.pack(fill="both", expand=True)

        canvas = tk.Canvas(shell, bg="#edf2f7", highlightthickness=0)
        v_scroll = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=v_scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        v_scroll.pack(side="right", fill="y")

        card = ttk.Frame(canvas, style="Card.TFrame", padding=14)
        canvas_window = canvas.create_window((0, 0), window=card, anchor="nw")
        self.results_canvas = canvas
        self.results_scrollable_frame = card

        def on_frame_configure(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        card.bind("<Configure>", on_frame_configure)
        canvas.bind("<Configure>", on_canvas_configure)

        def _on_wheel(event) -> None:
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        canvas.bind_all("<MouseWheel>", _on_wheel)
        canvas.bind_all("<Button-4>", _on_wheel)
        canvas.bind_all("<Button-5>", _on_wheel)

        ttk.Label(card, text="Compliance Results", style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=6, sticky="w")
        ttk.Label(
            card,
            text="This tab shows complete report details and generated images from the selected output folder.",
            style="Hint.TLabel",
        ).grid(row=1, column=0, columnspan=6, sticky="w", pady=(4, 8))

        ttk.Button(
            card,
            text="Load Latest From Output Folder",
            style="Accent.TButton",
            command=self._load_results_from_output_folder,
        ).grid(row=2, column=0, sticky="w", pady=(0, 10))

        kpi_card = ttk.Frame(card, style="Card.TFrame")
        kpi_card.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(0, 10))
        ttk.Label(kpi_card, text="Compliance Snapshot", style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=6, sticky="w")
        self._add_kpi(kpi_card, 1, 0, "Site", self.result_site_var)
        self._add_kpi(kpi_card, 1, 1, "Time Window", self.result_window_var)
        self._add_kpi(kpi_card, 1, 2, "Threshold", self.result_threshold_var)
        self._add_kpi(kpi_card, 1, 3, "Method", self.result_mode_var)
        self._add_kpi(kpi_card, 2, 0, "Before Green", self.result_before_var)
        self._add_kpi(kpi_card, 2, 1, "After Green", self.result_after_var)
        self._add_kpi(kpi_card, 2, 2, "Delta", self.result_delta_var)
        self._add_kpi(kpi_card, 2, 3, "ML Status", self.result_ml_var)
        self._add_kpi(kpi_card, 3, 0, "Alert Count", self.result_alerts_var)
        self._add_kpi(kpi_card, 3, 1, "Green Loss Area", self.result_green_loss_var)
        self._add_kpi(kpi_card, 3, 2, "New Construction", self.result_new_const_var)
        for col in range(4):
            kpi_card.columnconfigure(col, weight=1)

        image_card = ttk.Frame(card, style="Card.TFrame")
        image_card.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(0, 10))
        ttk.Label(image_card, text="Evidence Images", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(image_card, textvariable=self.results_image_caption_var, style="Hint.TLabel").grid(
            row=0, column=1, columnspan=5, sticky="w", padx=(10, 0)
        )

        self.results_preview_selector = ttk.Combobox(
            image_card,
            textvariable=self.results_preview_choice_var,
            values=["comparison_panel", "before_annotated_image", "after_annotated_image"],
            state="readonly",
            width=26,
        )
        self.results_preview_selector.grid(row=1, column=0, sticky="w", pady=(6, 8))
        self.results_preview_selector.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._set_results_image_preview(self.results_evidence_paths.get(self.results_preview_choice_var.get(), "")),
        )

        self.results_image_label = ttk.Label(image_card, style="Body.TLabel")
        self.results_image_label.grid(row=2, column=0, columnspan=6, sticky="w")

        splitter = ttk.Panedwindow(card, orient="vertical")
        splitter.grid(row=5, column=0, columnspan=6, sticky="nsew")
        card.rowconfigure(5, weight=1)
        card.columnconfigure(1, weight=1)

        upper = ttk.Frame(splitter, style="Card.TFrame")
        lower = ttk.Frame(splitter, style="Card.TFrame")
        splitter.add(upper, weight=3)
        splitter.add(lower, weight=2)

        upper_split = ttk.Panedwindow(upper, orient="horizontal")
        upper_split.pack(fill="both", expand=True)
        report_panel = ttk.Frame(upper_split, style="Card.TFrame")
        narrative_panel = ttk.Frame(upper_split, style="Card.TFrame")
        data_panel = ttk.Frame(upper_split, style="Card.TFrame")
        upper_split.add(report_panel, weight=2)
        upper_split.add(narrative_panel, weight=3)
        upper_split.add(data_panel, weight=2)

        ttk.Label(report_panel, text="Structured Report", style="SectionTitle.TLabel").pack(anchor="w")
        self.results_text = ScrolledText(report_panel, wrap="word", bg="#f8fafc", fg="#0f172a", relief="flat")
        self.results_text.pack(fill="both", expand=True, pady=(4, 0))

        ttk.Label(narrative_panel, text="Narrative Report", style="SectionTitle.TLabel").pack(anchor="w")
        self.results_narrative_text = ScrolledText(narrative_panel, wrap="word", bg="#f8fafc", fg="#0f172a", relief="flat")
        self.results_narrative_text.pack(fill="both", expand=True, pady=(4, 0))
        self.results_narrative_text.configure(state="disabled")

        ttk.Label(data_panel, text="ML Approach and Data View", style="SectionTitle.TLabel").pack(anchor="w")
        self.results_ml_text = ScrolledText(data_panel, wrap="word", bg="#f8fafc", fg="#0f172a", relief="flat", height=10)
        self.results_ml_text.pack(fill="both", expand=True, pady=(4, 6))
        self.results_ml_text.configure(state="disabled")
        self.results_data_text = ScrolledText(data_panel, wrap="word", bg="#f8fafc", fg="#0f172a", relief="flat", height=10)
        self.results_data_text.pack(fill="both", expand=True)
        self.results_data_text.configure(state="disabled")

        ttk.Label(lower, text="Violation Alerts", style="SectionTitle.TLabel").pack(anchor="w")
        alerts_frame = ttk.Frame(lower, style="Card.TFrame")
        alerts_frame.pack(fill="both", expand=True)
        self.results_alerts_tree = ttk.Treeview(alerts_frame, columns=("severity", "type", "message", "coord"), show="headings", height=8)
        self.results_alerts_tree.heading("severity", text="Severity")
        self.results_alerts_tree.heading("type", text="Alert Type")
        self.results_alerts_tree.heading("message", text="Message")
        self.results_alerts_tree.heading("coord", text="Geo Coordinate")
        self.results_alerts_tree.column("severity", width=90, anchor="center")
        self.results_alerts_tree.column("type", width=180, anchor="w")
        self.results_alerts_tree.column("message", width=560, anchor="w")
        self.results_alerts_tree.column("coord", width=220, anchor="w")

        results_alert_y_scroll = ttk.Scrollbar(alerts_frame, orient="vertical", command=self.results_alerts_tree.yview)
        results_alert_x_scroll = ttk.Scrollbar(alerts_frame, orient="horizontal", command=self.results_alerts_tree.xview)
        self.results_alerts_tree.configure(yscrollcommand=results_alert_y_scroll.set, xscrollcommand=results_alert_x_scroll.set)
        self.results_alerts_tree.grid(row=0, column=0, sticky="nsew")
        results_alert_y_scroll.grid(row=0, column=1, sticky="ns")
        results_alert_x_scroll.grid(row=1, column=0, sticky="ew")
        alerts_frame.rowconfigure(0, weight=1)
        alerts_frame.columnconfigure(0, weight=1)

    def _add_entry_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        browse_action=None,
    ) -> None:
        ttk.Label(parent, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, columnspan=3, sticky="ew", pady=4)
        if browse_action is not None:
            ttk.Button(parent, text="Browse", command=browse_action).grid(row=row, column=4, sticky="w", padx=(8, 0), pady=4)
        parent.columnconfigure(1, weight=1)

    def _add_kpi(self, parent: ttk.Frame, row: int, column: int, label: str, variable: tk.StringVar) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=(8, 6))
        frame.grid(row=row, column=column, sticky="ew", padx=4, pady=3)
        ttk.Label(frame, text=label, style="KPITitle.TLabel").pack(anchor="w")
        ttk.Label(frame, textvariable=variable, style="KPIValue.TLabel").pack(anchor="w")

    def _pick_boundary_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select boundary file",
            filetypes=[
                ("Boundary files", "*.zip *.shp *.geojson *.json"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.inspect_path_var.set(path)

    def _pick_json_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Summary output JSON", defaultextension=".json")
        if path:
            self.inspect_json_out_var.set(path)

    def _pick_output_folder(self) -> None:
        path = filedialog.askdirectory(title="Select discovery output folder")
        if path:
            self.discover_output_var.set(path)

    def _pick_boundary_file_analysis(self) -> None:
        path = filedialog.askopenfilename(
            title="Select analysis boundary",
            filetypes=[("Boundary files", "*.zip *.shp *.geojson *.json"), ("All files", "*.*")],
        )
        if path:
            self.boundary_path_var.set(path)

    def _pick_before_npz(self) -> None:
        path = filedialog.askopenfilename(title="Select before NPZ", filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")])
        if path:
            self.before_npz_var.set(path)

    def _pick_after_npz(self) -> None:
        path = filedialog.askopenfilename(title="Select after NPZ", filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")])
        if path:
            self.after_npz_var.set(path)

    def _pick_output_folder_analysis(self) -> None:
        path = filedialog.askdirectory(title="Select analysis output folder")
        if path:
            self.output_dir_var.set(path)

    def _pick_real_config(self) -> None:
        path = filedialog.askopenfilename(title="Select real-run config", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.cont_config_var.set(path)

    def _pick_history_output(self) -> None:
        path = filedialog.asksaveasfilename(title="History output JSON", defaultextension=".json")
        if path:
            self.cont_history_var.set(path)

    def _run_inspection(self) -> None:
        boundary = self.inspect_path_var.get().strip()
        summary_out = self.inspect_json_out_var.get().strip() or None
        if not boundary:
            messagebox.showerror("Missing input", "Please choose a boundary file first.")
            return

        self._set_status("Inspecting boundary...")

        def worker() -> None:
            result = inspect_kgis_workflow(boundary, summary_out)
            lines = [
                "Boundary Inspection Result",
                "",
                f"Source path: {result['source_path']}",
                f"Layer path: {result['extracted_layer_path']}",
                f"Geometry type: {result['geometry_type']}",
                f"Record count: {result['record_count']}",
                f"BBox: {tuple(result['bbox'])}",
                f"Field count: {len(result['fields'])}",
                f"Fields: {', '.join(result['fields'])}",
                "",
                "Sample record:",
            ]
            sample = result.get("sample_record", {})
            if sample:
                for key, value in sample.items():
                    lines.append(f"- {key}: {value}")
            else:
                lines.append("- No non-empty sample record available.")
            notes = result.get("notes") or []
            if notes:
                lines.append("")
                lines.append("Notes:")
                for note in notes:
                    lines.append(f"- {note}")
            if result.get("json_summary_path"):
                lines.append("")
                lines.append(f"Saved summary: {result['json_summary_path']}")
            self._update_text(self.inspect_text, "\n".join(lines))
            self._set_status("Boundary inspection completed")

        self._run_threaded(worker)

    def _run_discovery(self) -> None:
        query = self.discover_query_var.get().strip()
        output_dir = self.discover_output_var.get().strip()
        limit_raw = self.discover_limit_var.get().strip()
        include_buildings = self.discover_include_buildings_var.get()

        if not query:
            messagebox.showerror("Missing input", "Please enter a region query.")
            return
        if not output_dir:
            messagebox.showerror("Missing input", "Please select an output folder.")
            return
        try:
            limit = int(limit_raw)
        except ValueError:
            messagebox.showerror("Invalid value", "Candidate limit must be an integer.")
            return

        self._set_status("Running OSM discovery...")

        def worker() -> None:
            result = discover_open_industrial_workflow(
                query=query,
                bbox=None,
                output_dir=output_dir,
                limit=limit,
                include_buildings=include_buildings,
            )
            region = result["region"]
            lines = [
                "OSM Discovery Completed",
                "",
                f"Region: {region['display_name']}",
                f"Center: {region['lat']:.6f}, {region['lon']:.6f}",
                f"BBox: {tuple(region['bbox'])}",
                f"Total candidates: {result['candidate_count']}",
                "",
                "Top candidates:",
            ]
            for row in result["candidates"][:15]:
                lines.append(
                    f"- #{row['candidate_index']} {row['display_name']} "
                    f"(osm:{row['osm_type']}/{row['osm_id']}, area_hint={row['area_hint_deg2']})"
                )
            lines.append("")
            lines.append("Generated files:")
            for label, path in result["artifacts"].items():
                lines.append(f"- {label}: {path}")
            self._update_text(self.discovery_text, "\n".join(lines))
            self._set_status("OSM discovery completed")

        self._run_threaded(worker)

    def _run_analysis(self) -> None:
        try:
            datetime.strptime(self.before_date_var.get().strip(), "%Y-%m-%d")
            datetime.strptime(self.after_date_var.get().strip(), "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid date", "Dates must be in YYYY-MM-DD format.")
            return

        required_fields = {
            "Site ID": self.site_id_var.get().strip(),
            "Site name": self.site_name_var.get().strip(),
            "Boundary file": self.boundary_path_var.get().strip(),
            "Before NPZ": self.before_npz_var.get().strip(),
            "After NPZ": self.after_npz_var.get().strip(),
            "Output folder": self.output_dir_var.get().strip(),
            "Origin X": self.origin_x_var.get().strip(),
            "Origin Y": self.origin_y_var.get().strip(),
        }
        missing = [name for name, value in required_fields.items() if not value]
        if missing:
            messagebox.showerror("Missing inputs", "Please fill required fields:\n" + "\n".join(missing))
            return

        try:
            required_green = float(self.required_green_var.get().strip())
            feature_index = int(self.feature_index_var.get().strip())
            origin_x = float(self.origin_x_var.get().strip())
            origin_y = float(self.origin_y_var.get().strip())
            pixel_w = float(self.pixel_w_var.get().strip())
            pixel_h = float(self.pixel_h_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid numeric value", "Please review numeric fields (threshold, index, transform values).")
            return

        config = {
            "output_dir": self.output_dir_var.get().strip(),
            "site": {
                "site_id": self.site_id_var.get().strip(),
                "site_name": self.site_name_var.get().strip(),
                "metadata": {
                    "env_category": "Unknown",
                    "industrial_area_name": "Desktop UI run",
                },
            },
            "rule": {
                "rule_name": "Industrial green-belt compliance baseline",
                "required_green_cover_pct": required_green,
            },
            "ml_models": {
                "train_from_current_run": bool(self.enable_ml_var.get()),
                "bundle_path": str((Path(self.output_dir_var.get().strip()).expanduser().resolve() / "trained_monitoring_models.json")),
                "max_samples_per_scene": 30000,
            },
            "boundary": {
                "path": self.boundary_path_var.get().strip(),
                "feature_index": feature_index,
                "crs": self.boundary_crs_var.get().strip(),
            },
            "before_scene": {
                "npz_path": self.before_npz_var.get().strip(),
                "acquired_on": self.before_date_var.get().strip(),
                "origin_x": origin_x,
                "origin_y": origin_y,
                "pixel_width": pixel_w,
                "pixel_height": pixel_h,
                "crs": self.raster_crs_var.get().strip(),
                "source": "Provided by desktop UI",
                "sensor": "sentinel-2",
            },
            "after_scene": {
                "npz_path": self.after_npz_var.get().strip(),
                "acquired_on": self.after_date_var.get().strip(),
                "origin_x": origin_x,
                "origin_y": origin_y,
                "pixel_width": pixel_w,
                "pixel_height": pixel_h,
                "crs": self.raster_crs_var.get().strip(),
                "source": "Provided by desktop UI",
                "sensor": "sentinel-2",
            },
        }

        out_dir = Path(self.output_dir_var.get().strip()).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        generated_config = out_dir / "desktop_generated_run_config.json"
        generated_config.write_text(json.dumps(config, indent=2), encoding="utf-8")

        self._set_status("Running compliance analysis...")

        def worker() -> None:
            result = run_real_workflow(str(generated_config))
            report = result["report"]
            evidence = report.get("evidence_paths", {})
            self._update_text(self.analysis_text, "\n".join(self._format_report_lines(report, result.get("output_dir"))))
            self._update_evidence_panel(evidence)
            self._fill_alert_tree(report.get("alerts", []))
            self._apply_results_report(report=report, output_dir=result.get("output_dir"))
            self.after(0, lambda: self.notebook.select(self.results_tab))
            self._set_status("Compliance analysis completed")

        self._run_threaded(worker)

    def _run_continuous(self) -> None:
        config_path = self.cont_config_var.get().strip()
        if not config_path:
            messagebox.showerror("Missing input", "Please select a real-run config file.")
            return

        try:
            iterations = int(self.cont_iterations_var.get().strip())
            interval = int(self.cont_interval_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid value", "Iterations and interval must be integers.")
            return

        history_path = self.cont_history_var.get().strip() or None
        self._set_status("Running continuous monitoring...")

        def worker() -> None:
            result = run_continuous_monitoring(
                config_path=config_path,
                iterations=iterations,
                interval_seconds=interval,
                history_path=history_path,
            )
            lines = [
                "Continuous Monitoring Completed",
                "",
                f"Total runs in this batch: {result.run_count}",
                f"History file: {result.history_path}",
            ]
            self._update_text(self.continuous_text, "\n".join(lines))
            self._fill_continuous_tree(result.runs)
            self._set_status("Continuous monitoring completed")

        self._run_threaded(worker)

    def _fill_alert_tree(self, alerts: list[dict[str, Any]]) -> None:
        tree = self.analysis_alerts_tree
        for item in tree.get_children():
            tree.delete(item)
        for alert in alerts:
            coord = ""
            if isinstance(alert.get("geo_coordinate"), list) and len(alert["geo_coordinate"]) == 2:
                coord = f"{alert['geo_coordinate'][0]}, {alert['geo_coordinate'][1]}"
            tree.insert(
                "",
                "end",
                values=(
                    alert.get("severity", ""),
                    alert.get("alert_type", ""),
                    alert.get("message", ""),
                    coord,
                ),
            )

    def _fill_results_alert_tree(self, alerts: list[dict[str, Any]]) -> None:
        tree = self.results_alerts_tree
        for item in tree.get_children():
            tree.delete(item)
        for alert in alerts:
            coord = ""
            if isinstance(alert.get("geo_coordinate"), list) and len(alert["geo_coordinate"]) == 2:
                coord = f"{alert['geo_coordinate'][0]}, {alert['geo_coordinate'][1]}"
            tree.insert(
                "",
                "end",
                values=(
                    alert.get("severity", ""),
                    alert.get("alert_type", ""),
                    alert.get("message", ""),
                    coord,
                ),
            )

    def _fill_continuous_tree(self, runs: list[dict[str, Any]]) -> None:
        tree = self.continuous_tree
        for item in tree.get_children():
            tree.delete(item)
        for run in runs:
            report = run.get("report", {})
            tree.insert(
                "",
                "end",
                values=(
                    run.get("sequence", ""),
                    report.get("site_id", ""),
                    report.get("green_cover_before_pct", ""),
                    report.get("green_cover_after_pct", ""),
                    report.get("green_cover_delta_pct_points", ""),
                    report.get("alert_count", ""),
                ),
            )

    def _update_text(self, widget: tk.Text, message: str) -> None:
        def update() -> None:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("end", message)
            widget.configure(state="disabled")
        self.after(0, update)

    def _format_report_lines(self, report: dict[str, Any], output_dir: str | None) -> list[str]:
        before_metrics = report.get("before_metrics", {})
        after_metrics = report.get("after_metrics", {})
        lines = [
            "Compliance Analysis Completed",
            "",
            f"Site: {report.get('site_name', '')} ({report.get('site_id', '')})",
            f"Output folder: {output_dir or self.output_dir_var.get().strip()}",
            f"Rule threshold: {float(report.get('required_green_cover_pct', 0.0)):.2f}%",
            f"Time window: {report.get('before_date', '')} -> {report.get('after_date', '')}",
            "",
            "Vegetation Metrics",
            f"- Before green cover: {float(before_metrics.get('green_cover_pct', 0.0)):.2f}%",
            f"- After green cover: {float(after_metrics.get('green_cover_pct', 0.0)):.2f}%",
            f"- Delta: {float(report.get('green_cover_delta_pct_points', 0.0)):.2f} percentage points",
            f"- Green-loss area: {float(report.get('green_loss_area_sq_m', 0.0)):.2f}",
            "",
            "Construction Metrics",
            f"- New construction area: {float(report.get('new_construction_area_sq_m', 0.0)):.2f}",
            f"- New construction regions: {len(report.get('new_construction_regions', []))}",
            f"- Total alerts: {len(report.get('alerts', []))}",
            "",
            "Evidence Files",
        ]
        evidence = report.get("evidence_paths", {}) if isinstance(report.get("evidence_paths", {}), dict) else {}
        for label, path in evidence.items():
            lines.append(f"- {label}: {path}")
        return lines

    def _apply_results_report(self, report: dict[str, Any], output_dir: str | None = None) -> None:
        lines = self._format_report_lines(report, output_dir)
        self._update_text(self.results_text, "\n".join(lines))
        self._update_result_kpis(report)
        self._fill_results_alert_tree(report.get("alerts", []))
        self._update_results_evidence_panel(
            evidence=report.get("evidence_paths", {}),
            output_dir=output_dir or self.output_dir_var.get().strip(),
        )
        self._update_results_narrative(output_dir or self.output_dir_var.get().strip())
        self._update_results_ml_panel(report)
        self._update_results_data_panel(report)

    def _load_results_from_output_folder(self) -> None:
        output_dir = Path(self.output_dir_var.get().strip()).expanduser().resolve()
        report_path = output_dir / "compliance_report.json"
        if not report_path.exists():
            messagebox.showerror("Report not found", f"No compliance_report.json found in: {output_dir}")
            return
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self._apply_results_report(report=payload, output_dir=str(output_dir))
        self._set_status("Loaded compliance results from output folder")

    def _update_results_evidence_panel(self, evidence: dict[str, Any], output_dir: str) -> None:
        def update() -> None:
            evidence_paths = {
                key: str(value)
                for key, value in evidence.items()
                if isinstance(value, str)
            }
            self.results_evidence_paths = evidence_paths

            before_path = self.results_evidence_paths.get("before_annotated_image", "")
            after_path = self.results_evidence_paths.get("after_annotated_image", "")
            comparison_path = self.results_evidence_paths.get("comparison_panel", "")
            preferred_key = self.results_preview_choice_var.get().strip() or "comparison_panel"
            preview_path = self.results_evidence_paths.get(preferred_key, "")
            if not preview_path:
                preview_path = comparison_path or before_path or after_path
            self._set_results_image_preview(preview_path)

        self.after(0, update)

    def _set_results_image_preview(self, image_path: str) -> None:
        if self.results_image_label is None:
            return
        if not image_path or not Path(image_path).exists():
            self.results_image_label.configure(image="")
            self.results_preview_image = None
            self.results_image_caption_var.set("No evidence image generated yet.")
            return

        try:
            raw_img = tk.PhotoImage(file=image_path)
        except tk.TclError:
            self.results_image_label.configure(image="")
            self.results_preview_image = None
            self.results_image_caption_var.set(f"Preview unavailable for: {Path(image_path).name}. Use open buttons.")
            return

        max_w = 960
        max_h = 280
        sx = max(1, math.ceil(raw_img.width() / max_w))
        sy = max(1, math.ceil(raw_img.height() / max_h))
        scaled = raw_img.subsample(sx, sy)
        self.results_preview_image = scaled
        self.results_image_label.configure(image=self.results_preview_image)
        self.results_image_caption_var.set(f"Preview: {Path(image_path).name}")

    def _update_result_kpis(self, report: dict[str, Any]) -> None:
        before_metrics = report.get("before_metrics", {})
        after_metrics = report.get("after_metrics", {})
        metadata = report.get("metadata", {}) if isinstance(report.get("metadata"), dict) else {}
        ml_models = metadata.get("ml_models", {}) if isinstance(metadata.get("ml_models"), dict) else {}
        classification_mode = str(metadata.get("classification_mode", "spectral_rules")).replace("_", " ").title()
        self.result_site_var.set(f"{report.get('site_name', '-')} ({report.get('site_id', '-')})")
        self.result_window_var.set(f"{report.get('before_date', '-')} -> {report.get('after_date', '-')}")
        self.result_threshold_var.set(f"{float(report.get('required_green_cover_pct', 0.0)):.2f}%")
        self.result_mode_var.set(classification_mode)
        ml_enabled = bool(ml_models.get("enabled", False))
        if ml_enabled:
            self.result_ml_var.set("Enabled")
        else:
            self.result_ml_var.set("Disabled (spectral baseline)")
        self.result_before_var.set(f"{float(before_metrics.get('green_cover_pct', 0.0)):.2f}%")
        self.result_after_var.set(f"{float(after_metrics.get('green_cover_pct', 0.0)):.2f}%")
        self.result_delta_var.set(f"{float(report.get('green_cover_delta_pct_points', 0.0)):.2f} pp")
        self.result_alerts_var.set(str(len(report.get('alerts', []))))
        self.result_green_loss_var.set(f"{float(report.get('green_loss_area_sq_m', 0.0)):.2f}")
        self.result_new_const_var.set(f"{float(report.get('new_construction_area_sq_m', 0.0)):.2f}")

    def _update_results_narrative(self, output_dir: str) -> None:
        if self.results_narrative_text is None:
            return
        md_path = Path(output_dir).expanduser().resolve() / "compliance_report.md"
        if md_path.exists():
            content = md_path.read_text(encoding="utf-8")
        else:
            content = "Narrative report not found in output folder yet. Run analysis first."
        self.results_narrative_text.configure(state="normal")
        self.results_narrative_text.delete("1.0", "end")
        self.results_narrative_text.insert("end", content)
        self.results_narrative_text.configure(state="disabled")

    def _update_results_ml_panel(self, report: dict[str, Any]) -> None:
        if self.results_ml_text is None:
            return
        metadata = report.get("metadata", {}) if isinstance(report.get("metadata"), dict) else {}
        ml_models = metadata.get("ml_models", {}) if isinstance(metadata.get("ml_models"), dict) else {}
        classification_mode = str(metadata.get("classification_mode", "spectral_rules"))

        lines = [
            "ML / Algorithm Insights",
            "",
            f"Classification mode: {classification_mode}",
            f"ML enabled: {bool(ml_models.get('enabled', False))}",
            "",
        ]

        if bool(ml_models.get("enabled", False)):
            lines.extend(
                [
                    "Active approach:",
                    "- Trainable pixel logistic models for vegetation and built-up classes",
                    "- Inference applied over premises mask for before/after scenes",
                    f"- Bundle path: {ml_models.get('bundle_path', 'not provided')}",
                    f"- Trained in current run: {ml_models.get('trained_in_run', False)}",
                ]
            )
        else:
            lines.extend(
                [
                    "Active approach:",
                    "- Spectral-rule baseline (NDVI/NDBI-style thresholds)",
                    "- Deterministic mask differencing for green loss and new construction",
                    "- Reason ML shows disabled: this run did not load/train model bundle",
                    "  (Enable 'ML mode' checkbox in Compliance Analysis tab to train and use ML)",
                ]
            )

        self.results_ml_text.configure(state="normal")
        self.results_ml_text.delete("1.0", "end")
        self.results_ml_text.insert("end", "\n".join(lines))
        self.results_ml_text.configure(state="disabled")

    def _update_results_data_panel(self, report: dict[str, Any]) -> None:
        if self.results_data_text is None:
            return
        self.results_data_text.configure(state="normal")
        self.results_data_text.delete("1.0", "end")
        self.results_data_text.insert("end", json.dumps(report, indent=2))
        self.results_data_text.configure(state="disabled")

    def _load_results_if_available(self) -> None:
        output_dir = Path(self.output_dir_var.get().strip()).expanduser().resolve()
        report_path = output_dir / "compliance_report.json"
        if not report_path.exists():
            return
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self._apply_results_report(report=payload, output_dir=str(output_dir))
            self._set_status("Loaded latest compliance results")
        except Exception:
            # Keep startup robust even if saved report content is partially corrupted.
            return

    def _update_evidence_panel(self, evidence: dict[str, Any]) -> None:
        def update() -> None:
            self.analysis_evidence_paths = {
                key: str(value)
                for key, value in evidence.items()
                if isinstance(value, str)
            }

            before_path = self.analysis_evidence_paths.get("before_annotated_image", "")
            after_path = self.analysis_evidence_paths.get("after_annotated_image", "")
            comparison_path = self.analysis_evidence_paths.get("comparison_panel", "")
            preferred_key = self.analysis_preview_choice_var.get().strip() or "comparison_panel"
            preview_path = self.analysis_evidence_paths.get(preferred_key, "")
            if not preview_path:
                preview_path = comparison_path or before_path or after_path
            self._set_analysis_image_preview(preview_path)

        self.after(0, update)

    def _set_analysis_image_preview(self, image_path: str) -> None:
        if self.analysis_image_label is None:
            return
        if not image_path or not Path(image_path).exists():
            self.analysis_image_label.configure(image="")
            self.analysis_preview_image = None
            self.analysis_image_caption_var.set("No evidence image generated yet.")
            return

        try:
            raw_img = tk.PhotoImage(file=image_path)
        except tk.TclError:
            self.analysis_image_label.configure(image="")
            self.analysis_preview_image = None
            self.analysis_image_caption_var.set(f"Preview unavailable for: {Path(image_path).name}. Use open buttons.")
            return

        max_w = 960
        max_h = 280
        sx = max(1, math.ceil(raw_img.width() / max_w))
        sy = max(1, math.ceil(raw_img.height() / max_h))
        scaled = raw_img.subsample(sx, sy)
        self.analysis_preview_image = scaled
        self.analysis_image_label.configure(image=self.analysis_preview_image)
        self.analysis_image_caption_var.set(f"Preview: {Path(image_path).name}")

    def _set_status(self, message: str) -> None:
        self.after(0, lambda: self.status_var.set(message))

    def _run_threaded(self, callback) -> None:
        def wrapped() -> None:
            try:
                callback()
            except Exception as exc:
                error_message = str(exc)
                self._set_status("Operation failed")
                self.after(0, self._show_operation_error, error_message)

        threading.Thread(target=wrapped, daemon=True).start()

    def _show_operation_error(self, message: str) -> None:
        messagebox.showerror("Operation failed", message)
