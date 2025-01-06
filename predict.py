from cog import BasePredictor, Input, Path
import os
import torch
import subprocess

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        os.environ["PYTHONPATH"] = "./"
        
        # Extract BFM files
        if os.path.exists("deep_3drecon/BFM/BFM.zip"):
            subprocess.run(["unzip", "deep_3drecon/BFM/BFM.zip", "-d", "deep_3drecon/BFM/"])
            
        # Extract motion2video_nerf
        if os.path.exists("checkpoints/motion2video_nerf.zip"):
            subprocess.run(["unzip", "checkpoints/motion2video_nerf.zip", "-d", "checkpoints/"])
            
        self.device = "cuda"

    def predict(
        self,
        audio_file: Path = Input(description="Input audio file"),
        blink_mode: str = Input(
            default="none",
            choices=["none", "period"],
            description="Whether to blink periodically"
        ),
        lle_percent: float = Input(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Higher -> drag pred.landmark closer to training video's landmark set"
        ),
        temperature: float = Input(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Audio to secc temperature"
        ),
        mouth_amp: float = Input(
            default=0.4,
            ge=0.0,
            le=1.0,
            description="Higher -> mouth will open wider"
        ),
        fp16: bool = Input(
            default=False,
            description="Whether to utilize fp16 to speed up inference"
        )
    ) -> Path:
        input_path = f"data/raw/val_wavs/{audio_file.name}"
        os.makedirs(os.path.dirname(input_path), exist_ok=True)
        os.system(f"cp {audio_file} {input_path}")

        cmd = [
            "python", "inference/genefacepp_infer.py",
            "--a2m_ckpt=checkpoints/audio2motion_vae",
            "--head_ckpt=",
            "--torso_ckpt=checkpoints/motion2video_nerf/may_torso",
            f"--drv_aud={input_path}",
            f"--blink_mode={blink_mode}",
            f"--lle_percent={lle_percent}",
            f"--temperature={temperature}",
            f"--mouth_amp={mouth_amp}",
            "--out_name=output.mp4"
        ]
        
        if fp16:
            cmd.append("--fp16")
            
        subprocess.run(cmd)
        return Path("output.mp4")