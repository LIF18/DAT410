import torch
from torchinfo import summary
import torch.onnx
from step2_simple_model import SimpleSegmenter
from step3 import EncoderDecoderSegmenter

def main():
    print("Generating Professional Summaries (torchinfo)")
    model_step2 = SimpleSegmenter()
    model_step3 = EncoderDecoderSegmenter()
    
    # Simple Model Summary
    print("\n[Step 2: Simple CNN Model]")
    summary(model_step2, input_size=(1, 4, 224, 224), 
            col_names=["input_size", "output_size", "num_params", "kernel_size"], 
            row_settings=["var_names"])
            
    # Step 3: U-Net Model Summary
    print("\n[Step 3: Encoder-Decoder U-Net Model]")
    summary(model_step3, input_size=(1, 5, 224, 224), 
            col_names=["input_size", "output_size", "num_params", "kernel_size"], 
            row_settings=["var_names"])

    
    # Create dummy input tensors
    dummy_input_step2 = torch.randn(1, 4, 224, 224)
    dummy_input_step3 = torch.randn(1, 5, 224, 224)
    
    # Export models to ONNX format
    torch.onnx.export(model_step2, dummy_input_step2, "step2_model.onnx", 
                      input_names=['RGB_IR_Input'], output_names=['Segmentation_Output'])
                      
    torch.onnx.export(model_step3, dummy_input_step3, "step3_unet.onnx", 
                      input_names=['RGB_IR_Elev_Input'], output_names=['Segmentation_Output'])
                      
if __name__ == '__main__':
    main()