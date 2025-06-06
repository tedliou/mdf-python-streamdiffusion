from tools.module import Config
import pythonosc.dispatcher
import pythonosc.osc_server
import torch
from utils.wrapper import StreamDiffusionWrapper
import numpy as np
import cv2
import NDIlib as ndi
import pythonosc
from threading import Thread
from typing import Tuple

# Model Config
cfg = Config()
if cfg.is_new_file:
    cfg.set_name("StreamDiffusion Module")
    cfg.set_version("1.0.0")
    cfg.set_inputs(["ndi", "osc"])
    cfg.set_input_params("ndi_name", "td1")
    cfg.set_input_params("osc_ip", "127.0.0.1")
    cfg.set_input_params("osc_port", 12345)
    cfg.set_input_params("osc_path", "/prompt")
    cfg.set_process(["streamdiffusion"])
    cfg.set_process_params("base_model", "stabilityai/sd-turbo")
    cfg.set_process_params("t_index_list", [24, 32, 42, 46])
    cfg.set_process_params("default_prompt", "flower, detailed, fantasy, 8k, no human, no text")
    cfg.set_process_params("width", 512)
    cfg.set_process_params("height", 512)
    cfg.set_outputs(["ndi"])
    cfg.set_output_params("ndi_name", "sd1")

# Input Config
if "ndi" in cfg.get_inputs():
    ndi_input_name = cfg.get_input_params("ndi_name")
if "osc" in cfg.get_inputs():
    osc_server_ip = cfg.get_input_params("osc_ip")
    osc_server_port = cfg.get_input_params("osc_port")
    osc_server_path = cfg.get_input_params("osc_path")

# Process Config
if "streamdiffusion" in cfg.get_process():
    sd_model = cfg.get_process_params("base_model")
    sd_t_index_list= cfg.get_process_params("t_index_list")
    sd_default_prompt = cfg.get_process_params("default_prompt")
    sd_width = cfg.get_process_params("width")
    sd_height = cfg.get_process_params("height")

# Output Config
if "ndi" in cfg.get_outputs():
    ndi_output_name = cfg.get_output_params("ndi_name")


def process_image(image_np: np.ndarray, range: Tuple[int, int] = (0, 1)) -> Tuple[torch.Tensor, np.ndarray]:
    image = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    r_min, r_max = range[0], range[1]
    image = image * (r_max - r_min) + r_min
    return image.unsqueeze(0), image_np


def np2tensor(image_np: np.ndarray) -> torch.Tensor:
    height, width, _ = image_np.shape
    imgs = []
    img, _ = process_image(image_np)
    imgs.append(img)
    imgs = torch.vstack(imgs)
    images = torch.nn.functional.interpolate(
        imgs, size=(height, width), mode="bilinear", align_corners=False
    )
    image_tensors = images.to(torch.float16)
    return image_tensors

def oscprompt(address, args):
    if address == osc_server_path:
        global osc_shared_message
        osc_shared_message = args
        print(f"Receive New Prompt: {osc_shared_message}")

def main():
    stream = StreamDiffusionWrapper(
        model_id_or_path=sd_model,
        t_index_list=sd_t_index_list,
        width=sd_width,
        height=sd_height,
        acceleration="xformers",
        mode="img2img"
    )
    stream.prepare(prompt=sd_default_prompt)
    prompt = sd_default_prompt

    ndi.initialize()
    ndi_recv_find = ndi.find_create_v2()
    ndi_recv_sources = []
    while not len(ndi_recv_sources) > 0:
        ndi.find_wait_for_sources(ndi_recv_find, 1000)
        ndi_recv_sources = ndi.find_get_current_sources(ndi_recv_find)

    ndi_recv_source = None
    for s in ndi_recv_sources:
        if s.ndi_name.endswith(f"({ndi_input_name})"):
            ndi_recv_source = s
            break

    ndi_recv_setting = ndi.RecvCreateV3()
    ndi_recv_setting.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
    ndi_recv = ndi.recv_create_v3(ndi_recv_setting)
    ndi.recv_connect(ndi_recv, ndi_recv_source)
    ndi.find_destroy(ndi_recv_find)

    ndi_send_setting = ndi.SendCreate()
    ndi_send_setting.ndi_name = ndi_output_name
    ndi_send = ndi.send_create(ndi_send_setting)
    ndi_send_frame = ndi.VideoFrameV2()
    ndi_send_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX

    global osc_shared_message
    osc_shared_message = None
    osc_dispatcher = pythonosc.dispatcher.Dispatcher()
    osc_dispatcher.map(osc_server_path, oscprompt)
    osc_server = pythonosc.osc_server.ThreadingOSCUDPServer((osc_server_ip, osc_server_port), osc_dispatcher)
    osc_server_thread = Thread(target=osc_server.serve_forever)
    osc_server_thread.start()

    try:
        while True:
            t, v, _, _ = ndi.recv_capture_v2(ndi_recv, 3000)
            if t == ndi.FRAME_TYPE_VIDEO:
                # Prompt Input
                if osc_shared_message is not None:
                    prompt = str(osc_shared_message)
                    osc_shared_message = None
                    print(f"Apply Prompt: {prompt}")

                frame = np.copy(v.data)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame_tensor = np2tensor(frame)
                frame_output = stream(image=frame_tensor, prompt=prompt)
                
                frame_output_np = np.asarray(frame_output)

                frame_output_rgba = cv2.cvtColor(frame_output_np, cv2.COLOR_RGB2RGBA)
                ndi.recv_free_video_v2(ndi_recv, v)
                ndi_send_frame.data = frame_output_rgba
                ndi_send_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
                ndi.send_send_video_v2(ndi_send, ndi_send_frame)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping the server")
    finally:
        ndi.recv_destroy(ndi_recv)
        ndi.send_destroy(ndi_send)
        ndi.destroy()
        osc_server.shutdown()
        osc_server_thread.join()
        print("Stopped")


if __name__ == "__main__":
    main()
