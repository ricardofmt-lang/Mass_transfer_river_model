(() => {
    const app = window.parent.document;

    // Dragging or zooming updates the WebGL camera. When Plotly redraws
    // an animation frame, it reads the camera stored in the plot layout.
    function preserveView(event) {
        const control = event.target.closest?.(".updatemenu-button, .slider-container, .slider-group");
        const plot = control?.closest(".st-key-result_river_3d .js-plotly-plot");
        const camera = plot?._fullLayout?.scene?._scene?.getCamera?.();
        if (camera?.eye && plot.layout?.scene) {
            plot.layout.scene.camera = camera;
        }
    }

    // A case change can destroy this iframe while leaving the parent page
    // intact. Replace its old callback whenever the new iframe loads.
    const previousListener = app.__riverCameraListener;
    for (const event of ["pointerdown", "click"]) {
        if (previousListener) app.removeEventListener(event, previousListener, true);
        app.addEventListener(event, preserveView, true);
    }
    app.__riverCameraListener = preserveView;
})();
