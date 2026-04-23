# HW_3 Streamlit GUI Deployment Guide

This folder is designed to live inside `HW_3/streamlit_gui/`.

## Local run

From this folder:

```bash
streamlit run app/streamlit_app.py
```

or double-click `run_gui.bat` on Windows.

## Streamlit Community Cloud deployment

Repository: `Stardust-math/Mathematical_Modeling`

Branch: `main`

Main file path:

```text
HW_3/streamlit_gui/app/streamlit_app.py
```

The dependency file is intentionally placed in the same directory as the entrypoint:

```text
HW_3/streamlit_gui/app/environment.yml
```

## After deployment

Assume your app URL is:

```text
https://your-hw3-gui.streamlit.app
```

Then open `HW_3/index_with_streamlit_embed.html` and replace all occurrences of:

```text
https://your-hw3-gui.streamlit.app/
```

with your actual deployed Streamlit URL.

Then use the modified HTML to update `HW_3/index.html` in your repository.
