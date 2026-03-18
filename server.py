#!/usr/bin/env python3
"""Unified RKLLM Web Server for ROCK 5B (RK3588 NPU)
Handles text LLMs and VLM with live stream support.
Only one model loaded at a time to prevent OOM."""

import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import json
import base64
import tempfile
import gc
import signal
from flask import Flask, request, jsonify, Response, send_file

app = Flask(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────
MODELS = {
    "qwen3-1.7b": {
        "path": os.path.expanduser("~/models/Qwen3-1.7B-w8a8-rk3588.rkllm"),
        "type": "text",
        "ram_mb": 2500,
        "label": "Qwen3 1.7B (Fast)",
    },
    "qwen3-4b": {
        "path": os.path.expanduser("~/models/Qwen3-4B-w8a8-rk3588.rkllm"),
        "type": "text",
        "ram_mb": 5500,
        "label": "Qwen3 4B (Smart)",
    },
    "qwen3-vl-2b": {
        "path": os.path.expanduser("~/models/qwen3-vl-2b-instruct_w8a8_rk3588.rkllm"),
        "vision_path": os.path.expanduser("~/models/qwen3-vl-2b_vision_448_rk3588.rknn"),
        "type": "vlm",
        "ram_mb": 3500,
        "label": "Qwen3-VL 2B (Vision)",
    },
    "sam2.1": {
        "type": "sam",
        "ram_mb": 1200,
        "label": "SAM 2.1 (Accurate Segment)",
    },
}

VLM_STREAM_BIN = os.path.expanduser("~/rkllm-web/vlm_stream")
PLATFORM = "rk3588"
MEMORY_SAFETY_MB = 1500  # Keep at least 1.5GB free

# ─── RKLLM ctypes definitions ───────────────────────────────────────────────
rkllm_lib = None

LLMCallState_NORMAL = 0
LLMCallState_WAITING = 1
LLMCallState_FINISH = 2
LLMCallState_ERROR = 3

RKLLMInputType_PROMPT = 0
RKLLMInferMode_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


# ─── Callback for text inference ─────────────────────────────────────────────
global_text = []
global_state = -1


def callback_impl(result, userdata, state):
    global global_text, global_state
    if state == LLMCallState_FINISH:
        global_state = state
    elif state == LLMCallState_ERROR:
        global_state = state
    elif state == LLMCallState_NORMAL:
        global_state = state
        global_text.append(result.contents.text.decode("utf-8"))
    return 0


callback_type = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)
callback_func = callback_type(callback_impl)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_memory_info():
    """Get memory info in MB from /proc/meminfo."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if parts[0] in ("MemTotal:", "MemAvailable:", "MemFree:", "SwapTotal:", "SwapFree:"):
                info[parts[0].rstrip(":")] = int(parts[1]) / 1024  # KB to MB
    return {
        "total_mb": int(info.get("MemTotal", 0)),
        "available_mb": int(info.get("MemAvailable", 0)),
        "swap_total_mb": int(info.get("SwapTotal", 0)),
        "swap_free_mb": int(info.get("SwapFree", 0)),
    }


# ─── Model Manager ──────────────────────────────────────────────────────────
class ModelManager:
    def __init__(self):
        self.current_model = None
        self.model_type = None
        self.rkllm_handle = None
        self.rkllm_run_fn = None
        self.rkllm_destroy_fn = None
        self.rkllm_abort_fn = None
        self.vlm_process = None
        self.vlm_lock = threading.Lock()
        self.text_lock = threading.Lock()
        self.is_busy = False
        self.infer_params = None
        # Start memory watchdog
        self._watchdog = threading.Thread(target=self._memory_watchdog, daemon=True)
        self._watchdog.start()

    def _memory_watchdog(self):
        """Periodically restart VLM process to prevent memory leaks, and
        kill model if memory drops dangerously low."""
        self._vlm_infer_count = 0
        VLM_RESTART_EVERY = 200  # Restart VLM process every N inferences to reclaim leaked memory
        while True:
            time.sleep(10)
            try:
                mem = get_memory_info()

                # If memory is critical, unload everything
                if self.current_model and mem["available_mb"] < 500:
                    print(f"[WATCHDOG] Memory critical: {mem['available_mb']}MB free. Unloading model.")
                    sys.stdout.flush()
                    self.unload()
                    continue

                # Periodic VLM restart to combat memory leaks
                if (self.model_type == "vlm" and self._vlm_infer_count >= VLM_RESTART_EVERY
                        and not self.is_busy):
                    print(f"[WATCHDOG] Restarting VLM after {self._vlm_infer_count} inferences to reclaim memory.")
                    sys.stdout.flush()
                    model_name = self.current_model
                    self.unload()
                    time.sleep(2)
                    self.load(model_name)
                    self._vlm_infer_count = 0
            except Exception:
                pass

    def _init_rkllm_lib(self):
        global rkllm_lib
        if rkllm_lib is None:
            rkllm_lib = ctypes.CDLL("/usr/lib/librkllmrt.so")

    def load_text_model(self, model_name):
        cfg = MODELS[model_name]
        mem = get_memory_info()
        needed = cfg["ram_mb"] + MEMORY_SAFETY_MB
        if mem["available_mb"] < needed:
            return False, f"Not enough memory. Need {needed}MB, have {mem['available_mb']}MB"

        self._init_rkllm_lib()

        param = RKLLMParam()
        param.model_path = cfg["path"].encode("utf-8")
        param.max_context_len = 4096
        param.max_new_tokens = 4096
        param.skip_special_token = True
        param.n_keep = -1
        param.top_k = 1
        param.top_p = 0.9
        param.temperature = 0.8
        param.repeat_penalty = 1.1
        param.frequency_penalty = 0.0
        param.presence_penalty = 0.0
        param.mirostat = 0
        param.mirostat_tau = 5.0
        param.mirostat_eta = 0.1
        param.is_async = False
        param.img_start = b""
        param.img_end = b""
        param.img_content = b""
        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1
        param.extend_param.n_batch = 1
        param.extend_param.use_cross_attn = 0
        param.extend_param.enabled_cpus_num = 4
        param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

        handle = ctypes.c_void_p()
        init_fn = rkllm_lib.rkllm_init
        init_fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), callback_type]
        init_fn.restype = ctypes.c_int

        ret = init_fn(ctypes.byref(handle), ctypes.byref(param), callback_func)
        if ret != 0:
            return False, "rkllm_init failed"

        self.rkllm_handle = handle

        self.rkllm_run_fn = rkllm_lib.rkllm_run
        self.rkllm_run_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run_fn.restype = ctypes.c_int

        self.rkllm_destroy_fn = rkllm_lib.rkllm_destroy
        self.rkllm_destroy_fn.argtypes = [ctypes.c_void_p]
        self.rkllm_destroy_fn.restype = ctypes.c_int

        self.rkllm_abort_fn = rkllm_lib.rkllm_abort
        self.rkllm_abort_fn.argtypes = [ctypes.c_void_p]
        self.rkllm_abort_fn.restype = ctypes.c_int

        self.infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.infer_params.mode = RKLLMInferMode_GENERATE
        self.infer_params.keep_history = 0

        self.current_model = model_name
        self.model_type = "text"
        return True, "OK"

    def load_vlm(self, model_name):
        cfg = MODELS[model_name]
        mem = get_memory_info()
        needed = cfg["ram_mb"] + MEMORY_SAFETY_MB
        if mem["available_mb"] < needed:
            return False, f"Not enough memory. Need {needed}MB, have {mem['available_mb']}MB"

        if not os.path.exists(VLM_STREAM_BIN):
            return False, f"VLM binary not found: {VLM_STREAM_BIN}"

        try:
            self.vlm_process = subprocess.Popen(
                [VLM_STREAM_BIN, cfg["vision_path"], cfg["path"], "256", "1024", "3"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            # Wait for ---READY--- signal (up to 120s)
            ready = False
            start = time.time()
            while time.time() - start < 120:
                line = self.vlm_process.stdout.readline().strip()
                if line == "---READY---":
                    ready = True
                    break
                if self.vlm_process.poll() is not None:
                    stderr_out = self.vlm_process.stderr.read()
                    return False, f"VLM process exited: {stderr_out[:500]}"

            if not ready:
                self._kill_vlm()
                return False, "VLM process timed out during init"

        except Exception as e:
            return False, str(e)

        self.current_model = model_name
        self.model_type = "vlm"
        return True, "OK"

    def unload(self):
        if self.model_type == "text" and self.rkllm_handle:
            try:
                self.rkllm_destroy_fn(self.rkllm_handle)
            except Exception:
                pass
            self.rkllm_handle = None
            self.rkllm_run_fn = None
            self.rkllm_destroy_fn = None
            self.infer_params = None

        if self.model_type == "vlm":
            self._kill_vlm()

        if self.model_type == "sam":
            sam.unload()

        self.current_model = None
        self.model_type = None
        gc.collect()
        time.sleep(1)

    def _kill_vlm(self):
        if self.vlm_process:
            try:
                self.vlm_process.stdin.write("EXIT\n")
                self.vlm_process.stdin.flush()
                self.vlm_process.wait(timeout=5)
            except Exception:
                try:
                    self.vlm_process.kill()
                    self.vlm_process.wait(timeout=3)
                except Exception:
                    pass
            self.vlm_process = None

    def load(self, model_name):
        if model_name not in MODELS:
            return False, f"Unknown model: {model_name}"

        cfg = MODELS[model_name]

        # SAM models are handled by the SAM service, not ModelManager
        if cfg["type"] == "sam":
            ok, msg = sam.load(model_name)
            if ok:
                self.current_model = model_name
                self.model_type = "sam"
            return ok, msg

        if not os.path.exists(cfg["path"]):
            return False, f"Model file not found: {cfg['path']}"
        if cfg["type"] == "vlm" and not os.path.exists(cfg.get("vision_path", "")):
            return False, f"Vision model not found: {cfg.get('vision_path')}"

        # Unload current model first
        if self.current_model:
            self.unload()

        if cfg["type"] == "text":
            return self.load_text_model(model_name)
        else:
            return self.load_vlm(model_name)

    def chat(self, prompt, enable_thinking=False):
        global global_text, global_state
        if self.model_type != "text" or not self.rkllm_handle:
            return None, "No text model loaded"

        if not self.text_lock.acquire(timeout=1):
            return None, "Server busy"

        try:
            self.is_busy = True
            global_text = []
            global_state = -1

            rkllm_input = RKLLMInput()
            rkllm_input.role = b"user"
            rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking)
            rkllm_input.input_type = RKLLMInputType_PROMPT
            rkllm_input.input_data.prompt_input = prompt.encode("utf-8")

            thread = threading.Thread(
                target=self.rkllm_run_fn,
                args=(self.rkllm_handle, ctypes.byref(rkllm_input), ctypes.byref(self.infer_params), None),
            )
            thread.start()

            output = ""
            while True:
                while global_text:
                    output += global_text.pop(0)
                    time.sleep(0.005)
                thread.join(timeout=0.01)
                if not thread.is_alive():
                    # Drain remaining tokens
                    while global_text:
                        output += global_text.pop(0)
                    break

            return output, None
        except Exception as e:
            return None, str(e)
        finally:
            self.is_busy = False
            self.text_lock.release()

    def chat_stream(self, prompt, enable_thinking=False):
        """Generator that yields tokens as they arrive."""
        global global_text, global_state
        if self.model_type != "text" or not self.rkllm_handle:
            yield json.dumps({"error": "No text model loaded"})
            return

        if not self.text_lock.acquire(timeout=1):
            yield json.dumps({"error": "Server busy"})
            return

        try:
            self.is_busy = True
            global_text = []
            global_state = -1

            rkllm_input = RKLLMInput()
            rkllm_input.role = b"user"
            rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking)
            rkllm_input.input_type = RKLLMInputType_PROMPT
            rkllm_input.input_data.prompt_input = prompt.encode("utf-8")

            thread = threading.Thread(
                target=self.rkllm_run_fn,
                args=(self.rkllm_handle, ctypes.byref(rkllm_input), ctypes.byref(self.infer_params), None),
            )
            thread.start()

            while True:
                while global_text:
                    token = global_text.pop(0)
                    yield f"data: {json.dumps({'token': token})}\n\n"
                thread.join(timeout=0.01)
                if not thread.is_alive():
                    while global_text:
                        token = global_text.pop(0)
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    break

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            self.is_busy = False
            self.text_lock.release()

    def vlm_infer(self, image_path, prompt, timeout_sec=30):
        if self.model_type != "vlm" or not self.vlm_process:
            return None, "No VLM model loaded"
        if self.vlm_process.poll() is not None:
            return None, "VLM process has died"

        if not self.vlm_lock.acquire(timeout=1):
            return None, "VLM busy"

        try:
            self.is_busy = True
            self.vlm_process.stdin.write(f"{image_path}\t{prompt}\n")
            self.vlm_process.stdin.flush()

            # Read until ---RESPONSE_END--- or ---ERROR---
            response = ""
            in_response = False
            in_error = False
            start = time.time()

            while time.time() - start < timeout_sec:
                line = self.vlm_process.stdout.readline()
                if not line:
                    if self.vlm_process.poll() is not None:
                        return None, "VLM process died during inference"
                    continue

                line = line.rstrip("\n")
                if line == "---RESPONSE_START---":
                    in_response = True
                    continue
                if line == "---RESPONSE_END---":
                    return response.strip(), None
                if line == "---ERROR---":
                    in_error = True
                    continue
                if line == "---END---":
                    return None, response.strip()

                if in_response or in_error:
                    response += line + "\n"

            return None, "VLM inference timed out"
        except Exception as e:
            return None, str(e)
        finally:
            self.is_busy = False
            self._vlm_infer_count = getattr(self, '_vlm_infer_count', 0) + 1
            self.vlm_lock.release()


    def abort(self):
        """Abort current LLM generation."""
        if self.model_type == "text" and self.rkllm_abort_fn and self.rkllm_handle:
            try:
                self.rkllm_abort_fn(self.rkllm_handle)
                return True
            except Exception:
                pass
        return False


mgr = ModelManager()


# ─── SAM Service ─────────────────────────────────────────────────────────────
class SAMService:
    """Handles SAM 2.1 segmentation on RK3588 NPU."""

    SAM_MODELS = {
        "sam2.1": {
            "encoder": os.path.expanduser("~/models/sam2.1/sam2.1_hiera_small_encoder.rknn"),
            "decoder": os.path.expanduser("~/models/sam2.1/sam2.1_hiera_small_decoder.onnx"),
            "backend": "rknn",
            "img_size": 1024,
            "mask_size": 256,
            "label": "SAM 2.1 Small (Accurate ~4s)",
        },
    }

    def __init__(self):
        self.current_model = None
        self.encoder_session = None
        self.decoder_session = None
        self.cfg = None
        self.image_embedding = None  # Cached after encode
        self.image_orig_size = None
        self.lock = threading.Lock()

    def list_models(self):
        result = {}
        for name, cfg in self.SAM_MODELS.items():
            available = os.path.exists(cfg["encoder"]) and os.path.exists(cfg["decoder"])
            result[name] = {"label": cfg["label"], "available": available, "loaded": self.current_model == name}
        return result

    def load(self, model_name):
        if model_name not in self.SAM_MODELS:
            return False, f"Unknown SAM model: {model_name}"
        cfg = self.SAM_MODELS[model_name]
        if not os.path.exists(cfg["encoder"]) or not os.path.exists(cfg["decoder"]):
            return False, "Model files not found"

        self.unload()

        try:
            import onnxruntime
            from rknnlite.api import RKNNLite
            self.encoder_session = RKNNLite(verbose=False)
            self.encoder_session.load_rknn(cfg["encoder"])
            self.encoder_session.init_runtime()
            self.decoder_session = onnxruntime.InferenceSession(cfg["decoder"])
            self.cfg = cfg
            self.current_model = model_name
            self.image_embedding = None
            return True, "OK"
        except Exception as e:
            self.unload()
            return False, str(e)

    def unload(self):
        if self.encoder_session:
            try:
                self.encoder_session.release()
            except Exception:
                pass
            self.encoder_session = None
        self.decoder_session = None
        self.current_model = None
        self.cfg = None
        self.image_embedding = None
        self.image_orig_size = None

    def encode(self, image_data):
        """Encode an image and cache the embedding for repeated segmentation."""
        import numpy as np
        import cv2

        if not self.encoder_session:
            return False, "No SAM model loaded"

        with self.lock:
            img_array = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                return False, "Failed to decode image"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            self.image_orig_size = (h, w)

            size = self.cfg["img_size"]

            # Resize keeping aspect ratio, pad to 1024x1024
            scale = min(size / w, size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            padded = np.zeros((size, size, 3), dtype=np.float32)
            px, py = (size - new_w) // 2, (size - new_h) // 2
            padded[py:py + new_h, px:px + new_w] = resized
            inp = padded.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
            self._preprocess_info = (scale, px, py, new_w, new_h)

            outputs = self.encoder_session.inference(inputs=[inp], data_format="nchw")
            self.image_embedding = outputs  # [high_res_0, high_res_1, embed]

            return True, "OK"

    def segment(self, points, labels):
        """Run decoder with click points on the cached embedding.
        points: [[x, y], ...] in original image coordinates
        labels: [1, 0, ...] 1=positive, 0=negative, 2=box-tl, 3=box-br
        """
        import numpy as np
        import cv2

        if self.image_embedding is None:
            return None, "No image encoded. Call encode first."

        with self.lock:
            h, w = self.image_orig_size
            scale, px, py, new_w, new_h = self._preprocess_info
            size = self.cfg["img_size"]

            coords = np.array(points, dtype=np.float32)

            # Transform to padded 1024x1024 space
            coords[:, 0] = coords[:, 0] * scale + px
            coords[:, 1] = coords[:, 1] * scale + py
            pt_coords = coords[np.newaxis].astype(np.float32)
            pt_labels = np.array([labels], dtype=np.float32)
            mask_in = np.zeros((1, 1, 256, 256), dtype=np.float32)
            has_mask = np.zeros(1, dtype=np.float32)

            high0, high1, embed = self.image_embedding
            masks, ious = self.decoder_session.run(None, {
                "image_embed": embed, "high_res_feats_0": high0, "high_res_feats_1": high1,
                "point_coords": pt_coords, "point_labels": pt_labels,
                "mask_input": mask_in, "has_mask_input": has_mask,
            })

            # Post-process: resize mask back to original image
            best_idx = 0
            mask_1024 = cv2.resize(masks[0, best_idx], (size, size), interpolation=cv2.INTER_LINEAR)
            mask_crop = mask_1024[py:py + new_h, px:px + new_w]
            mask = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_LINEAR)

            binary_mask = (mask > 0.0).astype(np.uint8) * 255

            # Encode mask as PNG
            _, png_data = cv2.imencode(".png", binary_mask)
            mask_b64 = base64.b64encode(png_data.tobytes()).decode()

            return mask_b64, None


sam = SAMService()


# ─── Flask Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/status")
def status():
    mem = get_memory_info()
    return jsonify({
        "current_model": mgr.current_model,
        "model_type": mgr.model_type,
        "model_label": MODELS[mgr.current_model]["label"] if mgr.current_model else None,
        "is_busy": mgr.is_busy,
        "memory": mem,
    })


@app.route("/api/models")
def list_models():
    result = {}
    for name, cfg in MODELS.items():
        if cfg["type"] == "sam":
            sam_info = SAMService.SAM_MODELS.get(name, {})
            exists = os.path.exists(sam_info.get("encoder", "")) and os.path.exists(sam_info.get("decoder", ""))
        else:
            exists = os.path.exists(cfg["path"])
            if cfg["type"] == "vlm":
                exists = exists and os.path.exists(cfg.get("vision_path", ""))
        result[name] = {
            "label": cfg["label"],
            "type": cfg["type"],
            "ram_mb": cfg["ram_mb"],
            "available": exists,
            "loaded": mgr.current_model == name,
        }
    return jsonify(result)


@app.route("/api/load", methods=["POST"])
def load_model():
    data = request.json or {}
    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "Missing 'model' field"}), 400

    ok, msg = mgr.load(model_name)
    if ok:
        return jsonify({"status": "ok", "model": model_name})
    return jsonify({"error": msg}), 500


@app.route("/api/unload", methods=["POST"])
def unload_model():
    mgr.unload()
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    messages = data.get("messages", [])
    enable_thinking = data.get("enable_thinking", False)
    stream = data.get("stream", False)

    if not messages:
        return jsonify({"error": "No messages"}), 400

    # Extract last user message
    prompt = messages[-1].get("content", "")
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    if stream:
        return Response(
            mgr.chat_stream(prompt, enable_thinking),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result, err = mgr.chat(prompt, enable_thinking)
    if err:
        return jsonify({"error": err}), 500

    return jsonify({
        "choices": [{
            "message": {"role": "assistant", "content": result},
            "finish_reason": "stop",
        }]
    })


@app.route("/api/vlm", methods=["POST"])
def vlm():
    """VLM inference with uploaded image."""
    prompt = request.form.get("prompt", "What is in this image?")

    # Handle base64 image from JSON
    if request.is_json:
        data = request.json
        prompt = data.get("prompt", prompt)
        image_b64 = data.get("image", "")
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400
        # Strip data URL prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        image_data = base64.b64decode(image_b64)
    else:
        # Handle file upload
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        image_data = request.files["image"].read()

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir="/tmp")
    try:
        tmp.write(image_data)
        tmp.close()

        result, err = mgr.vlm_infer(tmp.name, prompt)
        if err:
            return jsonify({"error": err}), 500

        return jsonify({"response": result})
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route("/api/vlm/stream", methods=["POST"])
def vlm_stream():
    """VLM live stream - receive a frame, return description.
    If busy, returns immediately with skip=true."""
    if mgr.is_busy:
        return jsonify({"skip": True, "reason": "Inference in progress"}), 200

    data = request.json or {}
    image_b64 = data.get("image", "")
    prompt = data.get("prompt", "Briefly describe what you see in this image in 2-3 sentences.")

    if not image_b64:
        return jsonify({"error": "No image"}), 400

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_data = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"error": "Invalid base64 image"}), 400

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir="/tmp")
    try:
        tmp.write(image_data)
        tmp.close()

        result, err = mgr.vlm_infer(tmp.name, prompt, timeout_sec=15)
        if err:
            return jsonify({"error": err}), 500

        return jsonify({"response": result, "skip": False})
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# Also support the original RKLLM API format for backward compatibility
@app.route("/rkllm_chat", methods=["POST"])
def rkllm_chat():
    return chat()


@app.route("/api/stop", methods=["POST"])
def stop_generation():
    ok = mgr.abort()
    return jsonify({"stopped": ok})


# ─── SAM Routes ──────────────────────────────────────────────────────────────

@app.route("/api/sam/models")
def sam_models():
    return jsonify(sam.list_models())


@app.route("/api/sam/load", methods=["POST"])
def sam_load():
    data = request.json or {}
    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "Missing 'model' field"}), 400
    ok, msg = sam.load(model_name)
    if ok:
        return jsonify({"status": "ok", "model": model_name})
    return jsonify({"error": msg}), 500


@app.route("/api/sam/unload", methods=["POST"])
def sam_unload():
    sam.unload()
    return jsonify({"status": "ok"})


@app.route("/api/sam/encode", methods=["POST"])
def sam_encode():
    """Encode an image for segmentation. Send base64 image."""
    data = request.json or {}
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"error": "No image"}), 400
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        image_data = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"error": "Invalid base64"}), 400

    ok, msg = sam.encode(image_data)
    if ok:
        return jsonify({"status": "ok"})
    return jsonify({"error": msg}), 500


@app.route("/api/sam/segment", methods=["POST"])
def sam_segment():
    """Segment with points on the already-encoded image.
    Body: {"points": [[x,y],...], "labels": [1,0,...]}
    Labels: 1=positive click, 0=negative click, 2=box top-left, 3=box bottom-right
    """
    data = request.json or {}
    points = data.get("points", [])
    labels = data.get("labels", [])
    if not points or not labels or len(points) != len(labels):
        return jsonify({"error": "Invalid points/labels"}), 400

    mask_b64, err = sam.segment(points, labels)
    if err:
        return jsonify({"error": err}), 500
    return jsonify({"mask": mask_b64})


@app.route("/api/sam/auto", methods=["POST"])
def sam_auto():
    """Auto-segment a full frame: encode + segment center point.
    For live stream usage — sends back a mask overlay."""
    if not sam.current_model:
        return jsonify({"error": "No SAM model loaded"}), 400

    data = request.json or {}
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"error": "No image"}), 400
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        image_data = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"error": "Invalid base64"}), 400

    # Encode the frame
    ok, msg = sam.encode(image_data)
    if not ok:
        return jsonify({"error": msg}), 500

    # Auto-segment: use a grid of points across the image to find objects
    import numpy as np
    h, w = sam.image_orig_size
    # 3x3 grid of positive points
    points = []
    labels_list = []
    for gy in [0.25, 0.5, 0.75]:
        for gx in [0.25, 0.5, 0.75]:
            points.append([int(w * gx), int(h * gy)])
            labels_list.append(1)

    # Just use center point for cleaner results
    mask_b64, err = sam.segment([[w // 2, h // 2]], [1])
    if err:
        return jsonify({"error": err}), 500
    return jsonify({"mask": mask_b64, "skip": False})


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RKLLM Web Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--preload", type=str, default=None, help="Model to preload at startup")
    args = parser.parse_args()

    # Increase file descriptor limit
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))
    except Exception:
        pass

    # Generate self-signed SSL cert for HTTPS (needed for camera access)
    cert_dir = os.path.expanduser("~/rkllm-web/ssl")
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")
    if not os.path.exists(cert_file):
        os.makedirs(cert_dir, exist_ok=True)
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_file, "-out", cert_file,
            "-days", "3650", "-nodes",
            "-subj", "/CN=rock-5b"
        ], capture_output=True)
        print("Generated self-signed SSL certificate")

    # Fix NPU frequency for stable performance
    fix_script = os.path.expanduser(f"~/rkllm-web/fix_freq_{PLATFORM}.sh")
    if os.path.exists(fix_script):
        subprocess.run(f"sudo bash {fix_script}", shell=True, capture_output=True)

    if args.preload:
        print(f"Preloading model: {args.preload}")
        ok, msg = mgr.load(args.preload)
        if ok:
            print(f"Model {args.preload} loaded successfully")
        else:
            print(f"Failed to preload: {msg}")

    print(f"\n{'='*50}")
    print(f"  RKLLM Web Server")
    print(f"  https://0.0.0.0:{args.port}")
    print(f"{'='*50}\n")

    ssl_ctx = (cert_file, key_file) if os.path.exists(cert_file) else None
    app.run(host=args.host, port=args.port, threaded=True, debug=False, ssl_context=ssl_ctx)

    # Cleanup on exit
    mgr.unload()
