"""Load the browser-side camera and GIF controls for the Streamlit 3D view."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def gif_download_html() -> str:
    """Inline the local scripts so GIF export works offline and needs no Python renderer."""
    assets = Path(__file__).with_name("web")
    template = (assets / "gif_download.html").read_text(encoding="utf-8")
    # gifenc 1.0.3 is vendored with its MIT licence next to the script.
    encoder = (assets / "gifenc.js").read_text(encoding="utf-8").split("//# sourceMappingURL=")[0]
    camera_sync = (assets / "river_camera.js").read_text(encoding="utf-8")
    exporter = (assets / "gif_download.js").read_text(encoding="utf-8")
    return template.replace(
        "<!-- GIF exporter scripts -->",
        f"""<script>
{camera_sync}
const gifenc = (() => {{
    const module = {{ exports: {{}} }};
    const exports = module.exports;
    {encoder}
    return module.exports;
}})();
{exporter}
</script>""",
    )
