"""
Michelle Salon Rap - ACE-Step generation script
Doja Cat husky/raspy style, chill hip-hop, 50-60sec with vocals
"""

import sys
import os
import time
import shutil

OUTPUT_WAV = r"C:\NewtonHQ\00_meta\OUTPUT\michelle-landing\bonus\michelle-rap-ACE.wav"
OUTPUT_MP3 = r"C:\NewtonHQ\00_meta\OUTPUT\michelle-landing\bonus\michelle-rap-ACE.mp3"

PROMPT = (
    "chill hip-hop, lo-fi rnb hiphop fusion, "
    "female solo rap, husky raspy voice, mid-low register, "
    "breathy texture, slight grit, doja cat style melodic rap, "
    "chill swag confident playful tone, "
    "boom-bap drums laid back, soft jazz piano keys, "
    "warm bass, vinyl crackle subtle, 80 bpm relaxed, "
    "hair salon premium, manhattan koreatown, family pride, "
    "cool sophisticated, polished modern mix, vocal forward, clean"
)

LYRICS = """[verse]
Step into Kakaboka, baby take a seat
Michelle behind me, scissors flow so neat
Korean grandmaster touch, texture so unique
Layer by layer, watch me peak yeah

[chorus]
She know my hair (she know it)
Yeah she know my hair (she got it)
Family hands, premium care
Manhattan walls, that Michelle stare

[verse]
Veteran scissors, twenty something years
She don't rush, she listen, no fears
NJ to Manhattan we been holdin on
Cuts so clean even haters lookin gone

[outro]
Book it up, Kakaboka
We the ones, on top
She know my hair"""

DURATION = 55
STEPS = 27
CFG = 7.5
SEEDS = [42]

try:
    from acestep.pipeline_ace_step import ACEStepPipeline
    import acestep.pipeline_ace_step
    import torch
    import soundfile as sf
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def patched_save_wav_file(self, target_wav, idx, save_path=None, sample_rate=48000, format="wav"):
    if save_path is None:
        save_path = "./outputs/"
    if os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
        output_path_wav = os.path.join(save_path, f"output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        output_path_wav = save_path
    target_wav = target_wav.float().cpu()
    audio_np = target_wav.numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T
    sf.write(output_path_wav, audio_np, sample_rate, format=format.upper())
    print(f"WAV saved: {output_path_wav}")
    return output_path_wav

acestep.pipeline_ace_step.ACEStepPipeline.save_wav_file = patched_save_wav_file

print("=== Michelle Rap - ACE-Step Generation ===")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nInitializing pipeline...")
t0 = time.time()

model = ACEStepPipeline(
    checkpoint_dir=None,
    device_id=0 if torch.cuda.is_available() else -1,
    dtype="bfloat16" if torch.cuda.is_available() else "float32",
    torch_compile=False,
    cpu_offload=False
)

print(f"Pipeline ready in {time.time()-t0:.1f}s")
print("\nGenerating music...")
tgen = time.time()

model(
    prompt=PROMPT,
    lyrics=LYRICS,
    audio_duration=float(DURATION),
    infer_step=STEPS,
    guidance_scale=CFG,
    manual_seeds=SEEDS,
    scheduler_type="euler",
    use_erg_tag=True,
    use_erg_lyric=True,
    use_erg_diffusion=True,
    format="wav",
    batch_size=1,
    save_path=OUTPUT_WAV
)

gen_time = time.time() - tgen
print(f"\nGeneration done in {gen_time:.1f}s")

if os.path.exists(OUTPUT_WAV):
    print(f"WAV: {OUTPUT_WAV}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        import subprocess
        r = subprocess.run(
            [ffmpeg, "-y", "-i", OUTPUT_WAV, "-b:a", "192k", OUTPUT_MP3],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            print(f"MP3: {OUTPUT_MP3}")
        else:
            print(f"ffmpeg MP3 failed: {r.stderr[-200:]}")
    else:
        print("ffmpeg not found - WAV only")
else:
    print("ERROR: WAV not created!")
    sys.exit(1)

print(f"\nTotal: {time.time()-t0:.1f}s | Gen: {gen_time:.1f}s")
print("=== DONE ===")
