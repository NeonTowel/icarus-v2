#[test]
#[ignore = "diagnostic: inspect actual tensor names in DETR weights"]
fn inspect_detr_tensor_paths() {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;
    use std::path::Path;

    let weights_path = Path::new("/home/developer/.cache/huggingface/hub/models--facebook--detr-resnet-50/snapshots/1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b/model.safetensors");

    if !weights_path.exists() {
        println!("Weights file not found at {}", weights_path.display());
        println!("Skipping diagnostic test");
        return;
    }

    println!("\n\n=== DETR Tensor Path Diagnostic ===\n");

    let device = Device::Cpu;

    // Try to load the VarBuilder and see what errors we get
    let _vb_result =
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device) };

    // Use safetensors directly to inspect all keys
    let data = std::fs::read(weights_path).expect("Failed to read weights file");
    let safetensors =
        safetensors::SafeTensors::deserialize(&data).expect("Failed to parse safetensors");

    let mut keys: Vec<_> = safetensors
        .names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    keys.sort();

    println!("Total tensors: {}\n", keys.len());

    println!("=== First 100 tensor keys ===");
    for (i, key) in keys.iter().take(100).enumerate() {
        println!("{:3}: {}", i, key);
    }

    // Find layer1 specifically
    println!("\n\n=== Keys containing 'layer1' ===");
    let layer1_keys: Vec<_> = keys
        .iter()
        .filter(|k| k.as_str().contains("layer1"))
        .collect();
    println!("Found {} keys containing 'layer1':", layer1_keys.len());
    for key in layer1_keys.iter().take(30) {
        println!("  {}", key);
    }

    // Find backbone keys
    println!("\n\n=== Keys containing 'backbone' ===");
    let backbone_keys: Vec<_> = keys
        .iter()
        .filter(|k| k.as_str().contains("backbone"))
        .collect();
    println!("Found {} keys containing 'backbone':", backbone_keys.len());
    for key in backbone_keys.iter().take(40) {
        println!("  {}", key);
    }
}
