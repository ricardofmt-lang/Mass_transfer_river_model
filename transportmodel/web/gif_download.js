(() => {
    const { GIFEncoder, quantize, applyPalette } = gifenc;
    const button = document.getElementById("gif-download");
    const status = document.getElementById("gif-status");
    const app = window.parent.document;
    const maxImageDimension = 900;
    let pendingExport = null;

    // The iframe has its own stylesheet. Inherit the text colour from the app.
    const appView = app.querySelector('[data-testid="stAppViewContainer"]') ?? app.body;
    document.body.style.color = window.parent.getComputedStyle(appView).color;

    button.addEventListener("click", async () => {
        if (pendingExport) {
            pendingExport.cancelled = true;
            button.disabled = true;
            status.value = "Cancelling GIF export…";
            return;
        }

        const job = { cancelled: false };
        pendingExport = job;
        button.textContent = "Cancel GIF export";
        status.value = "Preparing GIF…";

        try {
            const { Plotly } = window.parent;
            const plot = app.querySelector(".st-key-result_river_3d .js-plotly-plot");
            if (!Plotly?.toImage || !plot?._fullLayout?.scene?._scene) {
                throw new Error("Wait for the 3D plot to finish loading, then try again.");
            }
            const animationFrames = plot._transitionData?._frames;
            if (!animationFrames?.length) {
                throw new Error("Run the simulation to create the animation frames.");
            }

            // Snapshot the current camera and theme without stepping the visible chart.
            const display = plot._fullLayout;
            const layout = structuredClone(plot.layout);
            layout.scene.camera = display.scene._scene.getCamera();
            layout.font = { ...display.font };
            layout.paper_bgcolor = display.paper_bgcolor;
            layout.plot_bgcolor = display.plot_bgcolor;

            // Use the same frames, time labels, and speed as the live Play control.
            // Plotly frames update the colours of all six faces, plus the title.
            const slider = layout.sliders[0];
            const frames = new Map(animationFrames.map(frame => [frame.name, frame]));
            const surfaces = structuredClone(plot.data);
            const property = layout.meta?.propertyName ?? "property";
            const delay = layout.updatemenus[0].buttons[0].args[1].frame.duration;
            const { width, height } = display;
            const scale = Math.min(1, maxImageDimension / Math.max(width, height));
            const canvas = document.createElement("canvas");
            canvas.width = Math.round(width * scale);
            canvas.height = Math.round(height * scale);
            const context = canvas.getContext("2d", { willReadFrequently: true });
            const gif = GIFEncoder();

            for (let index = 0; index < slider.steps.length; index++) {
                const [frameName] = slider.steps[index].args[0];
                const frame = frames.get(frameName);
                if (!frame) throw new Error("An animation frame is missing. Run the simulation again.");
                slider.active = index;
                layout.title = { ...layout.title, ...frame.layout?.title };
                const traces = surfaces.map(trace => ({ ...trace }));
                frame.data.forEach((patch, traceIndex) => {
                    const target = frame.traces?.[traceIndex] ?? traceIndex;
                    Object.assign(traces[target], patch);
                });
                status.value = `Rendering frame ${index + 1} of ${slider.steps.length}…`;
                const image = new Image();
                image.src = await Plotly.toImage({ data: traces, layout }, {
                    format: "png", width, height, scale,
                });
                if (job.cancelled) break;
                await image.decode();
                if (job.cancelled) break;

                // GIF frames have at most 256 colours. Quantize each image
                // independently so the mesh, text and colour scale stay clear.
                context.clearRect(0, 0, canvas.width, canvas.height);
                context.drawImage(image, 0, 0, canvas.width, canvas.height);
                const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
                const palette = quantize(data, 256);
                const indexed = applyPalette(data, palette);
                gif.writeFrame(indexed, canvas.width, canvas.height, { palette, delay, repeat: 0 });
            }

            if (job.cancelled) {
                status.value = "GIF export cancelled.";
                return;
            }
            gif.finish();
            const url = URL.createObjectURL(new Blob([gif.bytesView()], { type: "image/gif" }));
            const link = document.createElement("a");
            link.href = url;
            const suffix = property.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
            link.download = `river-${suffix || "property"}.gif`;
            document.body.append(link);
            link.click();
            link.remove();
            setTimeout(() => URL.revokeObjectURL(url), 30000);
            status.value = "GIF ready to save.";
        } catch (error) {
            status.value = `Could not export the GIF. ${error.message}`;
        } finally {
            pendingExport = null;
            button.textContent = "Download 3D GIF";
            button.disabled = false;
        }
    });
})();
