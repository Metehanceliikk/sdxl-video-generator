import torch
from diffusers import (
    DiffusionPipeline,
    LMSDiscreteScheduler,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)
from PIL import Image
import os
import imageio
from typing import List

from unet_architecture import UNet2DConditionModel
from vae_architecture import AutoencoderKL

def load_sdxl_components(model_id: str = "stabilityai/stable-diffusion-xl-base-1.0", vae_id: str = "madebyollin/sdxl-vae-fp16-fix"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan cihaz: {device}")
    
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    components = {}

    try:
        components["vae"] = AutoencoderKL.from_pretrained(
            vae_id, torch_dtype=torch_dtype
        ).to(device)
        
        components["tokenizer_one"] = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        components["text_encoder_one"] = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch_dtype
        ).to(device)
        
        components["tokenizer_two"] = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer_2"
        )
        components["text_encoder_two"] = CLIPTextModelWithProjection.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=torch_dtype
        ).to(device)

        components["unet"] = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch_dtype
        ).to(device)
        
        components["scheduler"] = LMSDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        
        return components
    
    except Exception as e:
        print(f"Model bileşenleri yüklenirken bir hata oluştu: {e}")
        return None

def run_diffusion_process(components, prompt: str):
    pipe = DiffusionPipeline(**components)
    pipe.set_progress_bar_config(disable=True)
    
    generated_image = pipe(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        height=1024,
        width=1024
    ).images[0]
    
    return generated_image

def create_video_from_frames(frame_paths: List[str], fps: int = 12, output_name: str = "output_video.mp4"):
    try:
        writer = imageio.get_writer(output_name, fps=fps)
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)
        writer.close()
        print(f"\nVideo başarıyla oluşturuldu: {output_name}")
    except Exception as e:
        print(f"Video oluşturulurken bir hata oluştu: {e}")

if __name__ == "__main__":
    sdxl_components = load_sdxl_components()
    
    if sdxl_components:
        prompt_text = "full shot of a majestic queen, standing on an ornate, sun-drenched marble balcony overlooking a sprawling, high-fantasy cityscape at golden hour. Her deep crimson silk gown, with intricate golden embroidery, cascades to the floor, catching the brilliant last rays of sunlight. A gentle breeze lifts the fabric, revealing a soft, inner lining. The queen's face is a perfect blend of serene contemplation and powerful authority. She wears an elegant, subtle crown adorned with tiny rubies that catch the light. The background features towering, spired castles and an endless sky painted in hues of orange and purple. Highly detailed, photorealistic, cinematic lighting, 8k, film grain, hyper-realistic textures, volumetric light, bokeh, masterpiece, best quality, ultra-detailed."
        num_frames = 24
        fps = 12
        output_folder = "/content/drive/MyDrive/sdxl_project/output_frames"

        print(f"'{prompt_text}' metninden {num_frames} adet kare oluşturuluyor...")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        frame_list = []
        for i in range(num_frames):
            print(f"--- Kare {i+1}/{num_frames} oluşturuluyor ---")
            current_prompt = f"{prompt_text}, cinematic shot, frame {i}/{num_frames}"
            
            generated_image = run_diffusion_process(
                components=sdxl_components,
                prompt=current_prompt
            )
            
            image_path = os.path.join(output_folder, f"frame_{i:04d}.png")
            generated_image.save(image_path)
            frame_list.append(image_path)
            print(f"Kare {i+1} kaydedildi: {image_path}")

        print(f"\nToplam {len(frame_list)} kare başarıyla oluşturuldu.")
        
        if frame_list:
            output_video_path = os.path.join(output_folder, "output_video.mp4")
            create_video_from_frames(frame_paths=frame_list, fps=fps, output_name=output_video_path)
    else:
        print("Model bileşenleri yüklenemediği için işlem sonlandırıldı.")