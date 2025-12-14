const RDD_LOOKUP_TABLE = [
    // 1. B2_Full: Baseline
    { name: "B2_Full", miou: 0.8908, gflops: 25.24, latency_ms: 178.6, channels: 1024, mask: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] },
    // 2. B2_a: Optimized Baseline
    { name: "B2_a",    miou: 0.8819, gflops: 23.92, latency_ms: 112.4, channels: 980,  mask: [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0] },
    // 3. B2_b: Balanced
    { name: "B2_b",    miou: 0.8819, gflops: 23.92, latency_ms: 108.7, channels: 920,  mask: [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0] },
    // 4. B2_c: Efficiency Step
    { name: "B2_c",    miou: 0.8635, gflops: 22.88, latency_ms: 107.5, channels: 900,  mask: [0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0] },
    // 5. B2_d: Aggressive Optimization
    { name: "B2_d",    miou: 0.8555, gflops: 21.49, latency_ms: 106.4, channels: 880,  mask: [0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0] },
    // 6. B2_e: Super Eco Test
    { name: "B2_e",    miou: 0.8540, gflops: 20.61, latency_ms: 105.3, channels: 832,  mask: [0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1] },
    // 7. B2_f: Super Eco Floor
    { name: "B2_f",    miou: 0.8505, gflops: 20.61, latency_ms: 104.2, channels: 768,  mask: [0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1] },
];

function selectConfig(manualLatency, battery, velocity, temp) {
    let max_gflops = 26.0;
    let mode = "PERF";
    let status_color = "green";

    // --- 1. BATTERY LOGIC ---
    if (battery >= 80) {
        max_gflops = 26.0; mode = "PERF"; status_color = "green";
    } else if (battery >= 60) {
        max_gflops = 24.0; mode = "HIGH EFF"; status_color = "green";
    } else if (battery >= 40) {
        max_gflops = 23.0; mode = "BALANCED"; status_color = "orange";
    } else if (battery >= 20) {
        max_gflops = 21.5; mode = "ECO(Power Saving)"; status_color = "orange";
    } else if (battery > 0) {
        max_gflops = 20.7; mode = "CRITICAL(Survival)"; status_color = "red";
    } else {
        return { config: null, mode: "SYSTEM OFFLINE", color: "gray" };
    }

    // --- 2. VELOCITY LOGIC ---
    let speed_latency_limit = velocity > 0.1 ? 300 / velocity : 1000;
    let effective_latency_limit = Math.min(manualLatency, speed_latency_limit);

    // --- 3. THERMAL LOGIC ---
    if (temp >= 80) {
        max_gflops = 20.7;
        mode = "🔥 THERMAL THROTTLING";
        status_color = "red";
    } else if (temp >= 70) {
        max_gflops = Math.min(max_gflops, 22.0);
        mode += " + 🔥 Warm";
    }

    // --- FILTERING ---
    let valid_options = RDD_LOOKUP_TABLE.filter(cfg => 
        cfg.latency_ms <= effective_latency_limit && cfg.gflops <= max_gflops
    );

    // --- FALLBACK ---
    if (valid_options.length === 0) {
        valid_options = [...RDD_LOOKUP_TABLE].sort((a, b) => a.latency_ms - b.latency_ms);
        return { 
            config: valid_options[0], 
            mode: "⚠️ OVERSPEED WARNING", 
            color: "red", 
            effective_lat: effective_latency_limit 
        };
    }

    // Sort: If low battery/high temp -> Prioritize Speed, Else -> Prioritize Accuracy
    if (battery < 20 || temp >= 75) {
        valid_options.sort((a, b) => a.latency_ms - b.latency_ms);
    } else {
        valid_options.sort((a, b) => b.miou - a.miou);
    }

    return { 
        config: valid_options[0], 
        mode, 
        color: status_color, 
        effective_lat: effective_latency_limit 
    };
}

module.exports = { selectConfig };