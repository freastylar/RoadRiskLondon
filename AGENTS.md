# RoadRisk London Agent Notes

- Keep the project data-first: no Streamlit UI work should depend on raw-data downloads, raw joins, or model training at runtime.
- Use `scripts/01_*` through `scripts/07_*` to generate artifacts before running `streamlit run app.py`.
- Do not invent DfT columns. Use schema inspection output before adding features or filters.
- Treat provisional data as excluded unless a user explicitly requests it.
- Run `ruff check .` and `pytest -q` before handing off changes.
