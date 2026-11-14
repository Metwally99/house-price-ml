function predictPrice(artifact, model, features) {
    let values = features.map((value, i) => (value - artifact.mean[i]) / artifact.scale[i]);
    const layers = artifact.models[model];
    for (const [index, layer] of layers.entries()) {
        values = layer.weight.map((row, unit) => {
            const output = row.reduce((sum, weight, i) => sum + weight * values[i], 0) + layer.bias[unit];
            return index === layers.length - 1 ? output : Math.max(0, output);
        });
    }
    return values[0] * artifact.target_scale + artifact.target_mean;
}

async function startDemo() {
    const form = document.getElementById("prediction");
    const button = form.querySelector("button");
    const model = document.getElementById("model_type");
    const status = document.getElementById("status");
    const result = document.getElementById("result");
    try {
        const responses = await Promise.all([fetch("./models.json"), fetch("./evaluation.json")]);
        if (responses.some(response => !response.ok)) throw new Error("Missing model files");
        const [artifact, report] = await Promise.all(responses.map(response => response.json()));
        for (const [code, entry] of Object.entries(report.models)) {
            if (code === "mean") continue;
            const label = code + " · " + entry.name + (code === artifact.selected_model ? " (selected)" : "");
            model.add(new Option(label, code));
        }
        model.value = artifact.selected_model;
        const scores = report.models[report.selected_model].test;
        document.getElementById("benchmark").textContent =
            "Held-out test (" + report.split.test + " sales) · MAE " + scores.mae.toFixed(2) +
            " · RMSE " + scores.rmse.toFixed(2) + " · R² " + scores.r2.toFixed(2);
        button.disabled = false;

        form.addEventListener("submit", event => {
            event.preventDefault();
            const values = artifact.features.map(name => form.elements.namedItem(name).valueAsNumber);
            const price = predictPrice(artifact, model.value, values);
            if (!Number.isFinite(price)) {
                result.hidden = true;
                status.textContent = "Enter values within the displayed ranges.";
                return;
            }
            status.textContent = "";
            document.getElementById("priceValue").textContent = price.toFixed(2);
            result.hidden = false;
        });
        form.requestSubmit();
    } catch {
        status.textContent = "The model could not load. Refresh the page to try again.";
    }
}

if (typeof document !== "undefined") startDemo();
if (typeof module !== "undefined") module.exports = { predictPrice };
