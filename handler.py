"""FontFoundry × zi2zi-JiT — RunPod Serverless Worker

把 zi2zi-JiT 扩散造字模型部署为按秒计费的云端端点，供「字库工坊」的「云端扩散生成」模式调用。

请求契约:
  input: {
    "chars": ["字", "库", ...],          # 本批要生成的字符（建议 ≤ 60 个/批）
    "style_id": "default",                # 风格标识 → 对应网络卷 styles/ 目录
    "size": 256,                          # 输出位图边长（固定 256，与 zi2zi-JiT 训练一致）
    "variant": "B",                       # 模型变体："B" 或 "L"
    "cfg": 2.6,                           # Classifier-free guidance（B 推荐 2.6，L 推荐 2.4）
    "steps": 20,                          # 采样步数（ab2 推荐 20）
    "sampling_method": "ab2"              # euler / heun / ab2
  }
响应契约:
  output: {
    "glyphs": [{"char": "字", "png": "<base64 PNG>"}],  # 白字黑底 RGB
    "failed": ["某字", ...]
  }
"""
import base64
import io
import os
import sys
import traceback
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import runpod
import torch
from PIL import Image, ImageDraw, ImageFont

# 把 zi2zi-jit 仓库加入路径（Dockerfile 已 clone 到 /app/zi2zi-jit）
sys.path.insert(0, "/app/zi2zi-jit")

from denoiser import Denoiser
from util.lora_utils import inject_lora, _is_lora_state_dict

# ---------------------------------------------------------------- 配置
WEIGHTS_ROOT = os.environ.get("ZI2ZI_WEIGHTS", "/runpod-volume/zi2zi-jit")
DEFAULT_VARIANT = os.environ.get("ZI2ZI_VARIANT", "B")
SOURCE_FONT = os.environ.get("ZI2ZI_SOURCE_FONT", "/app/fonts/source.ttf")

DEFAULT_CFG = {"B": 2.6, "L": 2.4}
DEFAULT_STEPS = {"ab2": 20, "euler": 20, "heun": 50}

# zi2zi-JiT 预训练底座参数（num_fonts/num_chars 必须与底座一致）
# 官方预训练权重使用 1000 fonts / 20000 chars（见 README 的 LoRA 示例）
BASE_ARGS = {
    "img_size": 256,
    "class_num": 1000,
    "num_fonts": 1000,
    "num_chars": 20000,
    "attn_dropout": 0.0,
    "proj_dropout": 0.0,
    "label_drop_prob": 0.1,
    "P_mean": -0.8,
    "P_std": 0.8,
    "t_eps": 5e-2,
    "noise_scale": 1.0,
    "ema_decay1": 0.9999,
    "ema_decay2": 0.99999,
    "interval_min": 0.0,
    "interval_max": 1.0,
}


@dataclass
class GenConfig:
    variant: str
    cfg: float
    steps: int
    method: str


# ---------------------------------------------------------------- 模型缓存
_model_cache = {}


def variant_args(variant: str, cfg: float, steps: int, method: str):
    """构造 Denoiser 所需的 args 对象"""
    class Args:
        pass

    args = Args()
    for k, v in BASE_ARGS.items():
        setattr(args, k, v)
    args.model = f"JiT-{variant}/16"
    args.sampling_method = method
    args.num_sampling_steps = steps
    args.cfg = cfg
    return args


def load_model(cfg: GenConfig) -> Tuple[Denoiser, torch.device]:
    """懒加载模型；按 (variant, style_id) 缓存，无 LoRA 时 style_id 不影响权重"""
    cache_key = cfg.variant
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and torch.backends.mps.is_available():
        device = torch.device("mps")

    # 路径：/runpod-volume/zi2zi-jit/JiT-B-16.pth 或 JiT-L-16.pth
    weight_name = f"zi2zi-JiT-{cfg.variant}-16.pth"
    checkpoint_path = os.path.join(WEIGHTS_ROOT, weight_name)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到模型权重: {checkpoint_path}")

    print(f"[worker] Loading checkpoint: {checkpoint_path} on {device}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = checkpoint.get("args") or variant_args(cfg.variant, cfg.cfg, cfg.steps, cfg.method)

    model = Denoiser(ckpt_args)
    state_dict = checkpoint.get("model_ema1") or checkpoint.get("model") or checkpoint

    is_lora = _is_lora_state_dict(state_dict)
    if is_lora:
        lora_r = getattr(ckpt_args, "lora_r", 8)
        lora_alpha = getattr(ckpt_args, "lora_alpha", 16)
        lora_dropout = getattr(ckpt_args, "lora_dropout", 0.0)
        lora_targets = getattr(ckpt_args, "lora_targets", "qkv,proj,w12,w3").split(",")
        inject_lora(model.net, lora_targets, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[worker] load_state_dict warning: missing={len(missing)}, unexpected={len(unexpected)}")

    # 覆盖生成超参
    model.cfg_scale = cfg.cfg
    model.steps = cfg.steps
    model.method = cfg.method
    model.cfg_interval = (BASE_ARGS["interval_min"], BASE_ARGS["interval_max"])

    model.to(device)
    model.eval()

    _model_cache[cache_key] = (model, device)
    print(f"[worker] Model {cfg.variant} loaded, params={sum(p.numel() for p in model.parameters()):,}")
    return model, device


def load_lora_for_style(model: Denoiser, style_id: str):
    """若存在该风格的 LoRA，加载到已缓存模型上（会修改缓存中的模型）。"""
    if not style_id:
        return
    lora_dir = os.path.join(WEIGHTS_ROOT, "loras", style_id)
    lora_path = os.path.join(lora_dir, "checkpoint-last.pth")
    if not os.path.isfile(lora_path):
        return

    print(f"[worker] Loading LoRA for style '{style_id}': {lora_path}")
    checkpoint = torch.load(lora_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model") or checkpoint
    if not _is_lora_state_dict(state_dict):
        print(f"[worker] Warning: {lora_path} 不是 LoRA 权重，跳过")
        return

    # 已注入 LoRA 的模型直接加载即可
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[worker] LoRA loaded: missing={len(missing)}, unexpected={len(unexpected)}")


# ---------------------------------------------------------------- 图像预处理

def render_source_char(ch: str, font_path: str, size: int = 256) -> np.ndarray:
    """把字符渲染成黑字白底 256x256 RGB 内容图（与训练时一致）"""
    if not os.path.isfile(font_path):
        raise FileNotFoundError(f"源字体不存在: {font_path}")

    # 尝试加载字体；macOS/Windows 常见中文字体兜底
    try:
        font = ImageFont.truetype(font_path, int(size * 0.75))
    except Exception as e:
        raise RuntimeError(f"无法加载源字体 {font_path}: {e}")

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((size // 2, size // 2), ch, fill=(0, 0, 0), font=font, anchor="mm")
    return np.array(img)  # (H, W, 3) uint8


def load_style_refs(style_id: str, max_refs: int = 8) -> List[Image.Image]:
    """从网络卷加载风格参考图"""
    style_dir = os.path.join(WEIGHTS_ROOT, "styles", style_id or "default")
    if not os.path.isdir(style_dir):
        return []

    refs = []
    for fn in sorted(os.listdir(style_dir)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            refs.append(Image.open(os.path.join(style_dir, fn)).convert("RGB"))
            if len(refs) >= max_refs:
                break
    return refs


def prepare_style_image(refs: List[Image.Image], size: int = 128) -> np.ndarray:
    """把参考图拼成 128x128 RGB；无参考时返回白底"""
    if refs:
        # 简单策略：取第一张并 resize；后续可改进为随机/平均
        img = refs[0].resize((size, size), Image.LANCZOS)
    else:
        img = Image.new("RGB", (size, size), (255, 255, 255))
    return np.array(img)


def normalize_image(arr: np.ndarray) -> np.ndarray:
    """uint8 [0,255] HWC -> float32 [-1,1] CHW"""
    arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
    return arr.transpose(2, 0, 1)


# ---------------------------------------------------------------- 生成

def synthesize(model: Denoiser, device: torch.device, ch: str,
               content_arr: np.ndarray, style_arr: np.ndarray) -> Image.Image:
    """生成单字，返回白字黑底 RGB PIL Image"""
    content = torch.from_numpy(normalize_image(content_arr)).unsqueeze(0).to(device)
    style = torch.from_numpy(normalize_image(style_arr)).unsqueeze(0).to(device)

    font_labels = torch.zeros(1, dtype=torch.long, device=device)
    char_labels = torch.zeros(1, dtype=torch.long, device=device)
    labels = (font_labels, char_labels, style, content)

    with torch.no_grad():
        generated = model.generate(labels)

    # [-1,1] CHW -> [0,255] HWC uint8
    gen = generated[0].detach().cpu().numpy()
    gen = np.transpose(gen, (1, 2, 0))
    gen = (gen + 1.0) / 2.0
    gen = np.clip(gen * 255.0, 0, 255).astype(np.uint8)

    # 统一为白字黑底：若平均灰度 > 127（黑字白底），则反色
    if float(gen.mean()) > 127:
        gen = 255 - gen

    return Image.fromarray(gen, "RGB")


# ---------------------------------------------------------------- RunPod handler

def handler(event):
    inp = event.get("input", {}) or {}

    # 调试模式：返回 checkpoint 的元信息，便于排查模型参数
    if inp.get("debug"):
        variant = str(inp.get("variant", DEFAULT_VARIANT)).upper()
        weight_name = f"zi2zi-JiT-{variant}-16.pth"
        checkpoint_path = os.path.join(WEIGHTS_ROOT, weight_name)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            keys = list(checkpoint.keys())
            args = checkpoint.get("args")
            args_dict = {k: str(v) for k, v in vars(args).items()} if args else None
            sd = checkpoint.get("model_ema1") or checkpoint.get("model") or checkpoint
            sd_keys = list(sd.keys())[:10] if isinstance(sd, dict) else ["not-a-dict"]
            return {"debug": True, "checkpoint_keys": keys, "checkpoint_args": args_dict, "state_dict_sample": sd_keys}
        except Exception as e:
            return {"debug": True, "error": str(e)}

    chars = list(dict.fromkeys(inp.get("chars", [])))[:200]  # 去重且限制 200 字
    style_id = inp.get("style_id", "")
    size_raw = inp.get("size")
    size = int(size_raw if size_raw is not None else 256)
    variant = str(inp.get("variant") or DEFAULT_VARIANT).upper()
    if variant not in ("B", "L"):
        variant = DEFAULT_VARIANT

    method = str(inp.get("sampling_method") or "ab2").lower()
    if method not in ("euler", "heun", "ab2"):
        method = "ab2"

    cfg_raw = inp.get("cfg")
    cfg = float(cfg_raw if cfg_raw is not None else DEFAULT_CFG.get(variant, 2.6))

    steps_raw = inp.get("steps")
    steps = int(steps_raw if steps_raw is not None else DEFAULT_STEPS.get(method, 20))

    if not chars:
        return {"error": "chars 不能为空"}

    if not os.path.isfile(SOURCE_FONT):
        return {"error": f"源字体不存在: {SOURCE_FONT}"}

    try:
        cfg_obj = GenConfig(variant=variant, cfg=cfg, steps=steps, method=method)
        model, device = load_model(cfg_obj)
        load_lora_for_style(model, style_id)

        style_refs = load_style_refs(style_id)
        style_arr = prepare_style_image(style_refs, size=128)

        glyphs, failed = [], []
        for ch in chars:
            try:
                content_arr = render_source_char(ch, SOURCE_FONT, size=size)
                img = synthesize(model, device, ch, content_arr, style_arr)

                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                glyphs.append({"char": ch, "png": base64.b64encode(buf.getvalue()).decode()})
            except Exception as e:
                print(f"[warn] '{ch}' 生成失败: {e}")
                failed.append(ch)

        return {"glyphs": glyphs, "failed": failed}
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "glyphs": [], "failed": chars}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
