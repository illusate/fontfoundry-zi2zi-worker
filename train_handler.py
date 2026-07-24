#!/usr/bin/env python3
"""FontFoundry × zi2zi-JiT — RunPod Serverless LoRA 微调 Worker

请求契约:
  input: {
    "style_id": "xxx",                 # 网络卷 styles/ 下的风格目录名
    "variant": "B",                    # 底座变体 B/L
    "base_checkpoint": "/runpod-volume/...",  # 可选，默认用 WEIGHTS_ROOT/zi2zi-JiT-{variant}-16.pth
    "epochs": 200,
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "batch_size": 8,
  }
响应契约:
  output: { "ok": true, "lora_dir": "...", "message": "..." }
"""
import base64
import io
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import runpod
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/app/zi2zi-jit")

WEIGHTS_ROOT = os.environ.get("ZI2ZI_WEIGHTS", "/runpod-volume/zi2zi-jit")
SOURCE_FONT = os.environ.get("ZI2ZI_SOURCE_FONT", "/app/fonts/source.ttf")
TRAIN_SCRIPT = os.environ.get(
    "ZI2ZI_TRAIN_SCRIPT", "/app/zi2zi-jit/lora_single_gpu_finetune_jit.py")

IMG_SIZE = 256
REF_SIZE = 128


def _char_from_filename(fn: str) -> str:
    """从 永.png / U+6C38.png 等文件名还原字符"""
    stem = Path(fn).stem
    if stem.startswith("U+") and len(stem) > 2:
        try:
            return chr(int(stem[2:], 16))
        except ValueError:
            pass
    return stem


def write_inline_style_images(style_id: str, glyph_images):
    """把内联 base64 参考图写入网络卷 styles/ 目录。"""
    style_dir = os.path.join(WEIGHTS_ROOT, "styles", style_id)
    os.makedirs(style_dir, exist_ok=True)
    # 清空旧文件，确保训练只用了本次传入的图
    for fn in os.listdir(style_dir):
        fp = os.path.join(style_dir, fn)
        if os.path.isfile(fp):
            os.remove(fp)
    saved = 0
    for idx, g in enumerate(glyph_images):
        ch = g.get("char", "")
        b64 = g.get("png") or g.get("image", "")
        if not b64 or not ch:
            continue
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            codepoint = f"U+{ord(ch):04X}"
            out_path = os.path.join(style_dir, f"{idx:05d}_{codepoint}.png")
            img.save(out_path, "PNG")
            saved += 1
        except Exception as e:
            print(f"[warn] 内联图 #{idx} ({ch}) 写入失败: {e}")
    if saved == 0:
        raise ValueError("没有成功写入任何内联参考图")
    print(f"[train] 已写入 {saved} 张内联参考图到 {style_dir}")
    return style_dir, saved


def render_source_char(ch: str, font_path: str, size: int = IMG_SIZE) -> Image.Image:
    """渲染源字体内容图（白字黑底）"""
    if not os.path.isfile(font_path):
        raise FileNotFoundError(f"源字体不存在: {font_path}")
    font = ImageFont.truetype(font_path, int(size * 0.75))
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((size // 2, size // 2), ch, fill=(255, 255, 255), font=font, anchor="mm")
    return img


def _fill_grid(grid: Image.Image, refs):
    """把 refs 前 4 张图填入 2x2 网格"""
    for i in range(4):
        ref = refs[i] if i < len(refs) else Image.new("RGB", (REF_SIZE, REF_SIZE), (255, 255, 255))
        ref = ref.resize((REF_SIZE, REF_SIZE), Image.LANCZOS)
        x = (i % 2) * REF_SIZE
        y = (i // 2) * REF_SIZE
        grid.paste(ref, (x, y))


def build_train_dataset(style_id: str, style_dir: str, train_dir: str):
    """把上传的参考字图整理成 zi2zi-JiT 训练格式"""
    # 收集参考图
    ref_entries = []
    for fn in sorted(os.listdir(style_dir)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            ch = _char_from_filename(fn)
            img = Image.open(os.path.join(style_dir, fn)).convert("RGB")
            ref_entries.append((ch, img))

    if len(ref_entries) < 4:
        raise ValueError(f"风格 '{style_id}' 至少需要 4 张参考字图，当前 {len(ref_entries)}")

    # 复用同一张图作为 8 个 ref patch 的候选（训练时会随机抽一个 patch）
    ref_pool = [img.resize((REF_SIZE, REF_SIZE), Image.LANCZOS) for _, img in ref_entries]

    font_dir = os.path.join(train_dir, f"001_{style_id}")
    os.makedirs(font_dir, exist_ok=True)

    for idx, (ch, target_img) in enumerate(ref_entries):
        source_img = render_source_char(ch, SOURCE_FONT, size=IMG_SIZE)
        target_img = target_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        grid0 = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        grid1 = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        _fill_grid(grid0, ref_pool)
        _fill_grid(grid1, ref_pool[4:] + ref_pool[:4])  # 另一 grid 用不同顺序

        composite = Image.new("RGB", (IMG_SIZE * 4, IMG_SIZE), (255, 255, 255))
        composite.paste(source_img, (0, 0))
        composite.paste(target_img, (IMG_SIZE, 0))
        composite.paste(grid0, (IMG_SIZE * 2, 0))
        composite.paste(grid1, (IMG_SIZE * 3, 0))

        codepoint = f"U+{ord(ch):04X}"
        out_path = os.path.join(font_dir, f"{idx:05d}_{codepoint}.png")
        composite.save(out_path, "PNG")

    return font_dir, len(ref_entries)


def prepare_base_checkpoint(src_path: str, variant: str) -> str:
    """把 zi2zi-JiT 预训练 checkpoint 转成 lora_single_gpu_finetune_jit.py 能接受的格式。

    上游训练脚本只识别 dict 里的 'model' 键；而公开预训练权重通常只有
    'model_ema1' / 'model_ema2' / 'args'。这里把它们统一存成 'model' 键，
    并把 args 也附进去，供脚本读取 lora_r/alpha 等元数据。
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"底座权重不存在: {src_path}")

    tmp_path = f"/tmp/zi2zi-base-{variant}.pth"
    if os.path.isfile(tmp_path):
        return tmp_path

    print(f"[train] 准备底座 checkpoint: {src_path} -> {tmp_path}")
    checkpoint = torch.load(src_path, map_location="cpu", weights_only=False)
    out = {"args": checkpoint.get("args")}
    for key in ("model_ema1", "model_ema2", "model"):
        if key in checkpoint:
            out["model"] = checkpoint[key]
            break
    if "model" not in out or out["model"] is None:
        # 兜底：整个 checkpoint 本身就是 state_dict
        out["model"] = checkpoint
    torch.save(out, tmp_path)
    print(f"[train] 底座 checkpoint 已准备，state_dict keys 样例: {list(out['model'].keys())[:5]}")
    return tmp_path


def run_training(style_id: str, variant: str, base_checkpoint: str, epochs: int,
                 lora_r: int, lora_alpha: int, lora_dropout: float, batch_size: int):
    style_dir = os.path.join(WEIGHTS_ROOT, "styles", style_id)
    if not os.path.isdir(style_dir):
        raise FileNotFoundError(f"风格目录不存在: {style_dir}")

    train_dir = f"/tmp/zi2zi-train-{style_id}"
    shutil.rmtree(train_dir, ignore_errors=True)
    font_dir, n_samples = build_train_dataset(style_id, style_dir, train_dir)

    if not os.path.isfile(base_checkpoint):
        raise FileNotFoundError(f"底座权重不存在: {base_checkpoint}")

    # 为上游脚本准备格式兼容的底座权重
    base_ckpt_for_train = prepare_base_checkpoint(base_checkpoint, variant)

    output_dir = os.path.join(WEIGHTS_ROOT, "loras", style_id)
    os.makedirs(output_dir, exist_ok=True)

    bs = min(batch_size, n_samples)
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--data_path", train_dir,
        "--output_dir", output_dir,
        "--base_checkpoint", base_ckpt_for_train,
        "--model", f"JiT-{variant}/16",
        "--epochs", str(epochs),
        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--batch_size", str(bs),
        "--max_chars_per_font", str(n_samples),
        "--num_workers", "4",
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "--save_last_freq", "5",
        "--log_freq", "10",
    ]

    print(f"[train] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    print(proc.stdout)
    print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"训练脚本退出码 {proc.returncode}: {proc.stderr[-500:]}")

    last_ckpt = os.path.join(output_dir, "checkpoint-last.pth")
    if not os.path.isfile(last_ckpt):
        raise RuntimeError("训练完成但未找到 checkpoint-last.pth")

    return output_dir, n_samples


def handler(event):
    inp = event.get("input", {}) or {}
    style_id = str(inp.get("style_id", "")).strip()
    if not style_id:
        return {"ok": False, "error": "style_id 不能为空"}

    variant = str(inp.get("variant", "B")).upper()
    if variant not in ("B", "L"):
        variant = "B"

    base_checkpoint = inp.get("base_checkpoint", "")
    if not base_checkpoint:
        base_checkpoint = os.path.join(WEIGHTS_ROOT, f"zi2zi-JiT-{variant}-16.pth")

    epochs = int(inp.get("epochs", 200))
    lora_r = int(inp.get("lora_r", 32))
    lora_alpha = int(inp.get("lora_alpha", 32))
    lora_dropout = float(inp.get("lora_dropout", 0.0))
    batch_size = int(inp.get("batch_size", 8))

    try:
        glyph_images = inp.get("glyph_images")
        if glyph_images:
            write_inline_style_images(style_id, glyph_images)

        output_dir, n_samples = run_training(
            style_id, variant, base_checkpoint, epochs,
            lora_r, lora_alpha, lora_dropout, batch_size,
        )
        return {
            "ok": True,
            "lora_dir": output_dir,
            "checkpoint": os.path.join(output_dir, "checkpoint-last.pth"),
            "message": f"风格 '{style_id}' LoRA 微调完成，参考字 {n_samples} 个，输出 {output_dir}",
        }
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
