# SCARA report

Upload this folder to Overleaf and compile `main.tex` with **pdfLaTeX**, or run `tectonic main.tex` locally. The ZIP contains the source, figures, tables, and measured experiment data.

To rebuild the figures and rerun all experiments from the repository:

```powershell
.\.venv\Scripts\python.exe scripts/report/build_assets.py --refresh
```

Install `report/requirements.txt` as well as the repository requirements before rebuilding figures. Compiling the supplied LaTeX does not require Python.

The RRT extension was implemented **after feedback from the final presentation**. It lives on `feature/rrt-path-planning`; `main` retains the original implementation, including its existing trajectory timing code.
