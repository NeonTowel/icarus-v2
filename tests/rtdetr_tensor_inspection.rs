#[test]
#[ignore = "diagnostic: inspect RT-DETR tensor names"]
fn inspect_rtdetr_tensor_paths() {
    use std::path::Path;

    let weights_path = Path::new(
        "/home/developer/.cache/huggingface/hub/models--PekingU--rtdetr_r50vd_coco_o365/snapshots",
    );

    if !weights_path.exists() {
        println!("Weights not cached yet");
        return;
    }

    if let Ok(entries) = std::fs::read_dir(weights_path) {
        for entry in entries.flatten() {
            let snap_path = entry.path();
            let model_file = snap_path.join("model.safetensors");

            if model_file.exists() {
                println!("Found: {}\n", model_file.display());

                let data = std::fs::read(&model_file).expect("Failed to read");
                let safetensors =
                    safetensors::SafeTensors::deserialize(&data).expect("Failed to parse");

                let mut keys: Vec<_> = safetensors
                    .names()
                    .into_iter()
                    .map(|s| s.to_string())
                    .collect();
                keys.sort();

                println!("Total tensors: {}\n", keys.len());

                println!("=== First 100 keys ===");
                for (index, key) in keys.iter().take(100).enumerate() {
                    println!("{:3}: {}", index, key);
                }

                println!("\n\n=== Backbone keys ===");
                let backbone: Vec<_> = keys
                    .iter()
                    .filter(|key| key.as_str().contains("backbone"))
                    .collect();
                println!("Found {} backbone keys:", backbone.len());
                for key in backbone.iter().take(50) {
                    println!("  {}", key);
                }

                println!("\n\n=== Keys with 'conv' ===");
                let conv_keys: Vec<_> = keys
                    .iter()
                    .filter(|key| key.as_str().contains("conv"))
                    .take(40)
                    .collect();
                for key in conv_keys {
                    println!("  {}", key);
                }
            }
        }
    }
}
