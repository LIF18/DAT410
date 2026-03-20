import torch
from model import GomokuResNet

def export_to_onnx():
    device = torch.device("cpu") # Exporting the computation graph on CPU is sufficient
    print("Loading model...")
    model = GomokuResNet(num_blocks=5, num_filters=64).to(device)
    
    try:
        model.load_state_dict(torch.load("best_resnet_gomoku.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Error: 'best_resnet_gomoku.pth' not found. Please ensure the model has been trained.")
        return

    # Create a dummy input tensor (Batch_Size=1, Channels=3, Height=15, Width=15)
    dummy_input = torch.randn(1, 3, 15, 15, device=device)
    onnx_filename = "gomoku_resnet.onnx"

    print(f"Exporting to ONNX format")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filename,
        export_params=True,
        opset_version=18,          # ONNX opset version
        do_constant_folding=True,  # Optimize constant folding
        input_names=['board_state'],   # Specify input node names
        output_names=['log_probabilities'], # Specify output node names
        dynamic_axes={'board_state': {0: 'batch_size'}, 'log_probabilities': {0: 'batch_size'}}
    )
    print(f"Export successfully. Open https://netron.app/ to view the network structure diagram.")

if __name__ == "__main__":
    export_to_onnx()