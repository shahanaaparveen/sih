# Single-Pass Drone Video → 3D Model (Prototype UI)

This is a lightweight single-page prototype that visualizes the pipeline steps for the SIH problem statement. It is intended for demonstrations and prototyping only.

Files added:

- `index.html` — main demo page
- `assets/css/styles.css` — styles
- `assets/js/app.js` — populates the pipeline cards and toggles details

How to run

1. Open the project folder and double-click `index.html` in your browser, or serve it with a static server.

Example (Python 3 built-in server):

```bash
python -m http.server 8000
# then open http://localhost:8000 in your browser
```

Next steps (optional):

- Replace summary/details content with richer text or images from the dataset.
- Add thumbnails or embedded video from a recorded flight.
- Hook each stage to demo scripts or small visualizations.

Sample 4K drone videos (royalty-free)

You can download any of these free 4K clips for testing. Example commands below use `curl`, `wget`, or the included scripts in `scripts/`.

- https://cdn.pixabay.com/video/2024/03/10/203678-922748476_large.mp4  (sea / seascape, 4K)
- https://cdn.pixabay.com/video/2022/12/09/142199-779684572_large.mp4 (mountains / forest, 4K)
- https://cdn.pixabay.com/video/2017/06/22/10213-222917614_large.mp4 (Grindelwald, Switzerland, 4K)

Download examples:

```bash
# using curl
curl -L -o assets/media/sample_4k.mp4 https://cdn.pixabay.com/video/2024/03/10/203678-922748476_large.mp4

# or using wget
wget -O assets/media/sample_4k.mp4 https://cdn.pixabay.com/video/2024/03/10/203678-922748476_large.mp4

# or use the included scripts
./scripts/download_sample.sh
powershell -File scripts\download_sample.ps1
```

Notes:
- These files are large (tens to hundreds of MB). Keep an eye on disk space and Git history if you add them to the repository — consider storing large binaries in a release or external storage instead of committing to Git.

