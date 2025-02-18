import os
import time
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from loguru import logger
from datetime import datetime
import gradio as gr
import random
import json
from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.constants import NEGATIVE_PROMPT
from hyvideo.modules.attenion import get_attention_modes
from mmgp import offload, safetensors2, profile_type 
import torch
import gc

attention_modes_supported = get_attention_modes()

args = parse_args()
args.flow_reverse = True


lock_ui_attention = False
lock_ui_transformer = False
lock_ui_compile = False


force_profile_no = int(args.profile)
verbose_level = int(args.verbose)
quantizeTransformer = args.quantize_transformer

transformer_choices_t2v=["ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_bf16.safetensors", "ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_quanto_int8.safetensors", "ckpts/hunyuan-video-t2v-720p/transformers/fast_hunyuan_video_720_quanto_int8.safetensors"]
transformer_choices_i2v=["ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_bf16.safetensors", "ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_quanto_int8.safetensors", "ckpts/hunyuan-video-t2v-720p/transformers/fast_hunyuan_video_720_quanto_int8.safetensors"]
text_encoder_choices = ["ckpts/text_encoder/llava-llama-3-8b-v1_1_fp16.safetensors", "ckpts/text_encoder/llava-llama-3-8b-v1_1_quanto_int8.safetensors"]

server_config_filename = "gradio_config.json"

if not Path(server_config_filename).is_file():
    server_config = {"attention_mode" : "sdpa",  
                     "transformer_filename": transformer_choices_t2v[1], 
                     "transformer_filename_i2v": transformer_choices_i2v[1],  ########
                     "text_encoder_filename" : text_encoder_choices[1],
                     "compile" : "",
                     "default_ui": "t2v",
                     "profile" : profile_type.LowRAM_LowVRAM }

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))
else:
    with open(server_config_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    server_config = json.loads(text)


transformer_filename_t2v = server_config["transformer_filename"]
transformer_filename_i2v = server_config.get("transformer_filename_i2v", transformer_choices_i2v[1]) ########

text_encoder_filename = server_config["text_encoder_filename"]
attention_mode = server_config["attention_mode"]
if len(args.attention)> 0:
    if args.attention in ["sdpa", "sage", "flash", "xformers"]:
        attention_mode = args.attention
        lock_ui_attention = True
    else:
        raise Exception(f"Unknown attention mode '{args.attention}'")

profile =  force_profile_no if force_profile_no >=0 else server_config["profile"]
compile = server_config.get("compile", "")
default_ui = server_config.get("default_ui", "t2v") 
use_image2video = default_ui != "t2v"
if args.t2v:
    use_image2video = False
if args.i2v:
    use_image2video = True

if use_image2video:
    lora_preselected =args.lora_weight_i2v
    lora_dir =args.lora_dir_i2v
    lora_preseleted_multiplier = [float(i) for i in args.lora_multiplier_i2v ]
else:
    lora_preselected =args.lora_weight
    lora_dir =args.lora_dir
    lora_preseleted_multiplier  = [float(i) for i in args.lora_multiplier ]

default_tea_cache = 0
if args.fast or args.fastest:
    transformer_filename_t2v = transformer_choices_t2v[2]
    attention_mode="sage"
    default_tea_cache = 0.15
    lock_ui_attention = True
    lock_ui_transformer = True

if args.fastest or args.compile:
    compile="transformer"
    lock_ui_compile = True

fast_hunyan = "fast" in transformer_filename_t2v

#transformer_filename = "ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_bf16.safetensors"
#transformer_filename = "ckpts/hunyuan-video-t2v-720p/transformers/hunyuan_video_720_quanto_int8.safetensors"
#transformer_filename = "ckpts/hunyuan-video-t2v-720p/transformers/fast_hunyuan_video_720_quanto_int8.safetensors"

#text_encoder_filename = "ckpts/text_encoder/llava-llama-3-8b-v1_1_fp16.safetensors"
#text_encoder_filename = "ckpts/text_encoder/llava-llama-3-8b-v1_1_quanto_int8.safetensors"

#attention_mode="sage"
#attention_mode="flash"
#attention_mode="sdpa"
#attention_mode="xformers"
# compile = "transformer"

def download_models(transformer_filename, text_encoder_filename):
    def computeList(filename):
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        if not "quanto" in filename:
            return [filename]        
        pos = filename.rfind(".")
        return [filename, filename[:pos] +"_map.json"]
    
    from huggingface_hub import hf_hub_download, snapshot_download    
    repoId = "DeepBeepMeep/HunyuanVideo" 
    sourceFolderList = ["text_encoder_2", "text_encoder", "hunyuan-video-t2v-720p/vae", "hunyuan-video-t2v-720p/transformers" ]
    fileList = [ [], ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"] + computeList(text_encoder_filename) , [],  computeList(transformer_filename) ]
    targetRoot = "ckpts/" 
    for sourceFolder, files in zip(sourceFolderList,fileList ):
        if len(files)==0:
            if not Path(targetRoot + sourceFolder).exists():
                snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= targetRoot)
        else:
             for onefile in files:      
                if not os.path.isfile(targetRoot + sourceFolder + "/" + onefile ):          
                    hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot, subfolder=sourceFolder)


offload.default_verboseLevel = verbose_level

download_models(transformer_filename_i2v if use_image2video else transformer_filename_t2v, text_encoder_filename) 

with open("./ckpts/hunyuan-video-t2v-720p/vae/config.json", "r", encoding="utf-8") as reader:
    text = reader.read()
vae_config= json.loads(text)
# reduce time window used by the VAE for temporal splitting (former time windows is too large for 24 GB) 
if vae_config["sample_tsize"] == 64:
    vae_config["sample_tsize"] = 32 
with open("./ckpts/hunyuan-video-t2v-720p/vae/config.json", "w", encoding="utf-8") as writer:
    writer.write(json.dumps(vae_config))



def setup_loras(pipe, lora_preselected, lora_dir, lora_preseleted_multiplier):
    # lora_weight =["ckpts/arny_lora.safetensors"] # 'ohwx person' ,; 'wick'
    # lora_multi = [1.0]
    loras =[]
    loras_names = []
    default_loras_choices = []
    default_loras_multis_str = ""

    from pathlib import Path
    if len(lora_preselected) > 0:
        lora_preselected = [  os.path.join(*Path(lora).parts)  for lora in lora_preselected]
        loras += lora_preselected
        loras_multis = (lora_preseleted_multiplier + ([1.0] * len(loras)) ) [:len(loras)]
        default_loras_choices = [ str(i) for i in range(len(loras))]
        default_loras_multis_str = "_".join([str(el) for el in loras_multis])


    if lora_dir != None:
        if not os.path.isdir(lora_dir):
            raise Exception("--lora-dir should be a path to a directory that contains Loras")
        
        import glob
        dir_loras =  glob.glob( os.path.join(lora_dir , "*.sft") ) + glob.glob( os.path.join(lora_dir , "*.safetensors") ) 
        dir_loras.sort()
        loras += [element for element in dir_loras if element not in loras ]

    if len(loras) > 0:
        loras_names = [ Path(lora).stem for lora in loras  ]
        offload.load_loras_into_model(pipe.transformer, loras,  activate_all_loras=False) #lora_multiplier,
    return loras, loras_names, default_loras_choices, default_loras_multis_str


def load_models(i2v,lora_preselected, lora_dir, lora_preseleted_multiplier ):
    download_models(transformer_filename_i2v if i2v else transformer_filename_t2v, text_encoder_filename) 
    if profile == 5:
        pinToMemory = False
        partialPinning = False
    else:    
        pinToMemory =  True
        import psutil
        physical_memory= psutil.virtual_memory().total    
        partialPinning = physical_memory <= 2**30 * 32 

    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(transformer_filename_i2v if i2v else transformer_filename_t2v, text_encoder_filename, attention_mode = attention_mode, pinToMemory = pinToMemory, partialPinning = partialPinning, args=args,  device="cpu") 
    pipe = hunyuan_video_sampler.pipeline
    pipe.transformer.any_compilation = len(compile)>0

    kwargs = { "extraModelsToQuantize": None}
    if profile == 2 or profile == 4:
        kwargs["budgets"] = { "transformer" : 100, "*" : 3000 }

    loras, loras_names, default_loras_choices, default_loras_multis_str = setup_loras(pipe, lora_preselected, lora_dir, lora_preseleted_multiplier)
    offloadobj = offload.profile(pipe, profile_no= profile, compile = compile, quantizeTransformer = quantizeTransformer, **kwargs)  


    return hunyuan_video_sampler, offloadobj, loras, loras_names, default_loras_choices, default_loras_multis_str

hunyuan_video_sampler, offloadobj,  loras, loras_names, default_loras_choices, default_loras_multis_str = load_models(use_image2video,lora_preselected, lora_dir, lora_preseleted_multiplier )
gen_in_progress = False

def get_default_steps_flow(fast_hunyan):
    return 6 if fast_hunyan else 30, 17.0 if fast_hunyan else 7.0 

def generate_header(fast_hunyan, compile, attention_mode):
    header = "<H2 ALIGN=CENTER><SPAN> ----------------- "
    header += "Fast HunyuanVideo model" if fast_hunyan else "HunyuanVideo model" 
    header += " (attention mode: " + attention_mode
    if attention_mode not in attention_modes_supported:
        header += " -NOT INSTALLED-"

    if compile:
        header += ", pytorch compilation ON"
    header += ") -----------------</SPAN></H2>"

    return header

def apply_changes(  state,
                    transformer_t2v_choice,
                    transformer_i2v_choice,
                    text_encoder_choice,
                    attention_choice,
                    compile_choice,
                    profile_choice,
                    default_ui_choice ="t2v",
):

    if gen_in_progress:
        yield "<DIV ALIGN=CENTER>Unable to change config when a generation is in progress</DIV>"
        return
    global offloadobj, hunyuan_video_sampler, loras, loras_names, default_loras_choices, default_loras_multis_str
    server_config = {"attention_mode" : attention_choice,  
                     "transformer_filename": transformer_choices_t2v[transformer_t2v_choice], 
                     "transformer_filename_i2v": transformer_choices_i2v[transformer_i2v_choice],  ##########
                     "text_encoder_filename" : text_encoder_choices[text_encoder_choice],
                     "compile" : compile_choice,
                     "profile" : profile_choice,
                     "default_ui" : default_ui_choice,
                       }

    if Path(server_config_filename).is_file():
        with open(server_config_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        old_server_config = json.loads(text)
        if lock_ui_transformer:
            server_config["transformer_filename"] = old_server_config["transformer_filename"]
            server_config["transformer_filename_i2v"] = old_server_config["transformer_filename_i2v"]
        if lock_ui_attention:
            server_config["attention_mode"] = old_server_config["attention_mode"]
        if lock_ui_compile:
            server_config["compile"] = old_server_config["compile"]

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))

    changes = []
    for k, v in server_config.items():
        v_old = old_server_config[k]
        if v != v_old:
            changes.append(k)

    state["config_changes"] = changes
    state["config_new"] = server_config
    state["config_old"] = old_server_config

    global attention_mode, profile, compile, transformer_filename_t2v, transformer_filename_i2v, text_encoder_filename
    attention_mode = server_config["attention_mode"]
    profile = server_config["profile"]
    compile = server_config["compile"]
    transformer_filename_t2v = server_config["transformer_filename"]
    transformer_filename_i2v = server_config["transformer_filename_i2v"]
    text_encoder_filename = server_config["text_encoder_filename"]

    if  all(change in ["attention_mode"] for change in changes ):
        if "attention_mode" in changes:
            # quick attention mode update to avoid reinitializing everything
            transformer = hunyuan_video_sampler.pipeline.transformer
            transformer.attention_mode = attention_mode
            for module in transformer.double_blocks:
                module.attention_mode = attention_mode
            for module in transformer.single_blocks:
                module.attention_mode = attention_mode

    else:
        hunyuan_video_sampler = None
        offloadobj.release()
        offloadobj = None
        yield "<DIV ALIGN=CENTER>Please wait while the new configuration is being applied</DIV>"

        hunyuan_video_sampler, offloadobj,  loras, loras_names, default_loras_choices, default_loras_multis_str = load_models(use_image2video,lora_preselected, lora_dir, lora_preseleted_multiplier )


    yield "<DIV ALIGN=CENTER>The new configuration has been succesfully applied</DIV>"

    # return "<DIV ALIGN=CENTER>New Config file created. Please restart the Gradio Server</DIV>"

def update_defaults(state, num_inference_steps,flow_shift):
    if "config_changes" not in state:
        return get_default_steps_flow(False)
    changes = state["config_changes"] 
    server_config = state["config_new"] 
    old_server_config = state["config_old"] 

    new_fast_hunyuan = "fast" in server_config["transformer_filename"]
    old_fast_hunyuan = "fast" in old_server_config["transformer_filename"]

    if  "transformer_filename" in changes:
        if new_fast_hunyuan != old_fast_hunyuan:
            num_inference_steps, flow_shift = get_default_steps_flow(new_fast_hunyuan)

    header = generate_header(new_fast_hunyuan, server_config["compile"], server_config["attention_mode"] )
    return num_inference_steps, flow_shift, header 


from moviepy.editor import ImageSequenceClip
import numpy as np

def save_video(final_frames, output_path, fps=24):
    assert final_frames.ndim == 4 and final_frames.shape[3] == 3, f"invalid shape: {final_frames} (need t h w c)"
    if final_frames.dtype != np.uint8:
        final_frames = (final_frames * 255).astype(np.uint8)
    ImageSequenceClip(list(final_frames), fps=fps).write_videofile(output_path, verbose= False, logger = None)

def build_callback(state, pipe, progress, status, num_inference_steps):
    def callback(step_idx, t, latents):
        step_idx += 1         
        if state.get("abort", False):
            # pipe._interrupt = True
            status_msg = status + " - Aborting"    
        elif step_idx  == num_inference_steps:
            status_msg = status + " - VAE Decoding"    
        else:
            status_msg = status + " - Denoising"   

        progress( (step_idx , num_inference_steps) , status_msg  ,  num_inference_steps)
            
    return callback

def abort_generation(state):
    if "in_progress" in state:
        state["abort"] = True
        hunyuan_video_sampler.pipeline._interrupt= True
        return gr.Button(interactive=  False)
    else:
        return gr.Button(interactive=  True)

def refresh_gallery(state):
    file_list = state.get("file_list", None)      
    return file_list
        
def finalize_gallery(state):
    choice = 0
    if "in_progress" in state:
        del state["in_progress"]
        choice = state.get("selected",0)
    
    time.sleep(0.2)
    gen_in_progress = False
    return gr.Gallery(selected_index=choice), gr.Button(interactive=  True)

def select_video(state , event_data: gr.EventData):
    data=  event_data._data
    if data!=None:
        state["selected"] = data.get("index",0)
    return 


def generate_video(
    prompt,
    negative_prompt,
    resolution,
    video_length,
    seed,
    num_inference_steps,
    guidance_scale,
    flow_shift,
    embedded_guidance_scale,
    repeat_generation,
    tea_cache,
    loras_choices,
    loras_mult_choices,
    image_to_continue,
    video_to_continue,
    max_frames,
    state,
    progress=gr.Progress() #track_tqdm= True

):
    
    from PIL import Image
    import numpy as np
    import tempfile


    if hunyuan_video_sampler == None:
        raise gr.Error("Unable to generate a Video while a new configuration is being applied.")
    if not attention_mode in attention_modes_supported:
        raise gr.Error(f"You have selected attention mode '{attention_mode}'. However it is not installed on your system. You should either install it or switch to the default 'sdpa' attention.")


    global gen_in_progress
    gen_in_progress = True
    temp_filename = None
    if use_image2video:
        if image_to_continue is not None:
            PIL_image = Image.fromarray(np.uint8(image_to_continue)).convert('RGB')
            with tempfile.NamedTemporaryFile("w+b", delete = False, suffix=".png") as fp: 
                PIL_image.save(fp, format="png")
                fp.close()

            input_image_or_video_path = fp.name
            temp_filename = input_image_or_video_path
            # pipeline.num_input_frames = 1 
            # pipeline.max_frames = 1 

        elif video_to_continue != None and len(video_to_continue) >0 :
            input_image_or_video_path = video_to_continue
            # pipeline.num_input_frames = max_frames
            # pipeline.max_frames = max_frames
        else:
            return
    else:
        input_image_or_video_path = None


    if len(loras) > 0:
        def is_float(element: any) -> bool:
            if element is None: 
                return False
            try:
                float(element)
                return True
            except ValueError:
                return False
        list_mult_choices_nums = []
        if len(loras_mult_choices) > 0:
            list_mult_choices_str = loras_mult_choices.split(" ")
            for i, mult in enumerate(list_mult_choices_str):
                mult = mult.strip()
                if not is_float(mult):                
                    raise gr.Error(f"Lora Multiplier no {i+1} ({mult}) is invalid")
                list_mult_choices_nums.append(float(mult))
        if len(list_mult_choices_nums ) < len(loras_choices):
            list_mult_choices_nums  += [1.0] * ( len(loras_choices) - len(list_mult_choices_nums ) )

        offload.activate_loras(hunyuan_video_sampler.pipeline.transformer, loras_choices, list_mult_choices_nums)

    seed = None if seed == -1 else seed
    width, height = resolution.split("x")
    width, height = int(width), int(height)

    if "abort" in state:
        del state["abort"]
    state["in_progress"] = True
    state["selected"] = 0
 


   # TeaCache
    trans = hunyuan_video_sampler.pipeline.transformer
    trans.enable_teacache = tea_cache > 0
 
    import random
    if seed == None or seed <0:
        seed = random.randint(0, 999999999)

    file_list = []
    state["file_list"] = file_list    
    from einops import rearrange
    save_path = os.path.join(os.getcwd(), "gradio_outputs")
    os.makedirs(save_path, exist_ok=True)
    prompts = prompt.replace("\r", "").split("\n")
    video_no = 0
    total_video =  repeat_generation * len(prompts)
    abort = False
    start_time = time.time()
    for prompt in prompts:
        for _ in range(repeat_generation):
            if abort:
                break

            if trans.enable_teacache:
                trans.num_steps = num_inference_steps
                trans.cnt = 0
                trans.rel_l1_thresh = tea_cache #0.15 # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
                trans.accumulated_rel_l1_distance = 0
                trans.previous_modulated_input = None
                trans.previous_residual = None

            video_no += 1
            status = f"Video {video_no}/{total_video}"
            progress(0, desc=status + " - Encoding Prompt" )   
            
            callback = build_callback(state, hunyuan_video_sampler.pipeline, progress, status, num_inference_steps)

            if use_image2video:
                # input_image_or_video_path
                raise Exception("image 2 video not yet supported") #################
            else:
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    outputs = hunyuan_video_sampler.predict(
                        prompt=prompt,
                        height=height,
                        width=width, 
                        video_length=(video_length // 4)* 4 + 1 ,
                        seed=seed,
                        negative_prompt=negative_prompt,
                        infer_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        num_videos_per_prompt=1,
                        flow_shift=flow_shift,
                        batch_size=1,
                        embedded_guidance_scale=embedded_guidance_scale,
                        callback = callback,
                        callback_steps = 1,

                    )
                except:
                    gen_in_progress = False
                    if temp_filename!= None and  os.path.isfile(temp_filename):
                        os.remove(temp_filename)
                    offload.last_offload_obj.unload_all()
                    gc.collect()
                    torch.cuda.empty_cache()                        
                    raise gr.Error("The generation of the video has encountered an error: it is likely that you have unsufficient VRAM and you should therefore reduce the video resolution or its number of frames.")


            samples = outputs['samples']
            if samples != None:
                samples = samples.to("cpu")
            outputs['samples'] = None
            offload.last_offload_obj.unload_all()
            gc.collect()
            torch.cuda.empty_cache()

            if samples == None:
                end_time = time.time()
                abort = True
                yield f"Video generation was aborted. Total Generation Time: {end_time-start_time:.1f}s"
            else:
                idx = 0
                # just in case one day we will have enough VRAM for batch generation ...
                for i,sample in enumerate(samples):
                    # sample = samples[0]
                    video = rearrange(sample.cpu().numpy(), "c t h w -> t h w c")

                    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
                    file_name = f"{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','').strip()}.mp4".replace(':',' ').replace('\\',' ')
                    idx = 0 
                    basis_video_path = os.path.join(os.getcwd(), "gradio_outputs", file_name)        
                    video_path = basis_video_path
                    while True:
                        if  not Path(video_path).is_file():
                            idx = 0
                            break
                        idx += 1
                        video_path = basis_video_path[:-4] + f"_{idx}" + ".mp4"


                    save_video(video, video_path )
                    print(f"New video saved to Path: "+video_path)
                    file_list.append(video_path)
                    if video_no < total_video:
                        yield  status
                    else:
                        end_time = time.time()
                        yield f"Total Generation Time: {end_time-start_time:.1f}s"
            seed += 1
  
    if temp_filename!= None and  os.path.isfile(temp_filename):
        os.remove(temp_filename)
    gen_in_progress = False

def create_demo():
    
    default_inference_steps, default_flow_shift = get_default_steps_flow(fast_hunyan)
    
    with gr.Blocks() as demo:
        state = gr.State({})
       
        if use_image2video:
            gr.Markdown("<div align=center><H1>HunyuanVideo<SUP>GP</SUP> - AI Image To Video Generator</H1></div>")
        else:
            gr.Markdown("<div align=center><H1>HunyuanVideo<SUP>GP</SUP> - AI Text To Video Generator</H1></div>")

        gr.Markdown("<H2><B>GPU Poor version by DeepBeepMeep</B> (<A HREF='https://github.com/deepbeepmeep/HunyuanVideoGP'>Updates</A> / <A HREF='https://github.com/Tencent/HunyuanVideo'>Original by Tencent</A>).</H>")

        if use_image2video:
            pass
        else:
            gr.Markdown("The resolution and the duration of the video will depend on the amount of VRAM your GPU has, for instance if you have 24 GB of VRAM (RTX 3090 / RTX 4090):")
            gr.Markdown("- max 192 frames for 848 x 480 ")
            gr.Markdown("- max 97 frames for 1280 x 720")
        gr.Markdown("In order to find the sweet spot you will need try different resolution / duration and reduce these if the app is hanging : in the very worst case one generation step should not take more than 2 minutes. If it is the case you may be running out of RAM / VRAM.")
        gr.Markdown("Please note that if your turn on compilation, the first generation step of the first video generation will be slow due to the compilation. Therefore all your tests should be done with compilation turned off.")


        # css = """<STYLE>
        #         h2 { width: 100%;  text-align: center; border-bottom: 1px solid #000; line-height: 0.1em; margin: 10px 0 20px;  } 
        #         h2 span {background:#fff;  padding:0 10px; }</STYLE>"""
        # gr.HTML(css)

        header = gr.Markdown(generate_header(fast_hunyan, compile, attention_mode)  )            

        with gr.Accordion("Video Engine Configuration - click here to change it", open = False):
            gr.Markdown("For the changes to be effective you will need to restart the gradio_server. Some choices below may be locked if the app has been launched by specifying a config preset.")

            with gr.Column():
                index = transformer_choices_t2v.index(transformer_filename_t2v)
                index = 0 if index ==0 else index
                transformer_t2v_choice = gr.Dropdown(
                    choices=[
                        ("Hunyuan Text to Video 16 bits - the default engine in its original glory, offers a slightly better image quality but slower and requires more RAM", 0),
                        ("Hunyuan Text to Video quantized to 8 bits (recommended) - the default engine but quantized", 1),
                        ("Fast Hunyuan Text to Video quantized to 8 bits - requires less than 10 steps but worse quality", 2), 
                    ],
                    value= index,
                    label="Transformer model for Text to Video",
                    interactive= not lock_ui_transformer
                 )

                index = transformer_choices_i2v.index(transformer_filename_i2v)
                index = 0 if index ==0 else index
                transformer_i2v_choice = gr.Dropdown(
                    choices=[
                        ("Hunyuan Image to Video 16 bits - the default engine in its original glory, offers a slightly better image quality but slower and requires more RAM", 0),
                        ("Hunyuan Image to Video quantized to 8 bits (recommended) - the default engine but quantized", 1),
                        # ("Fast Hunyuan Video quantized to 8 bits - requires less than 10 steps but worse quality", 2), 
                    ],
                    value= index,
                    label="Transformer model for Image to Video",
                    interactive= not lock_ui_transformer,
                    visible = False, ###############
                 )

                index = text_encoder_choices.index(text_encoder_filename)
                index = 0 if index ==0 else index

                text_encoder_choice = gr.Dropdown(
                    choices=[
                        ("Llava Llama 1.1 16 bits - unquantized text encoder, better quality uses more RAM", 0),
                        ("Llava Llama 1.1 quantized to 8 bits - quantized text encoder, slightly worse quality but uses less RAM", 1),
                    ],
                    value= index,
                    label="Text Encoder model"
                 )
                def check(mode): 
                    if not mode in attention_modes_supported:
                        return " (NOT INSTALLED)"
                    else:
                        return ""
                attention_choice = gr.Dropdown(
                    choices=[
                        ("Scale Dot Product Attention: default", "sdpa"),
                        ("Flash" + check("flash")+ ": good quality - requires additional install (usually complex to set up on Windows without WSL)", "flash"),
                        ("Xformers" + check("xformers")+ ": good quality - requires additional install (usually complex, may consume less VRAM to set up on Windows without WSL)", "xformers"),
                        ("Sage" + check("sage")+ ": 30% faster but worse quality - requires additional install (usually complex to set up on Windows without WSL)", "sage"),
                    ],
                    value= attention_mode,
                    label="Attention Type",
                    interactive= not lock_ui_attention
                 )
                gr.Markdown("Beware: when restarting the server or changing a resolution or video duration, the first step of generation for a duration / resolution may last a few minutes due to recompilation")
                compile_choice = gr.Dropdown(
                    choices=[
                        ("ON: works only on Linux / WSL", "transformer"),
                        ("OFF: no other choice if you have Windows without using WSL", "" ),
                    ],
                    value= compile,
                    label="Compile Transformer (up to 50% faster and 30% more frames but requires Linux / WSL and Flash or Sage attention)",
                    interactive= not lock_ui_compile
                 )                
                profile_choice = gr.Dropdown(
                    choices=[
                ("HighRAM_HighVRAM, profile 1: at least 48 GB of RAM and 24 GB of VRAM, the fastest for short videos a RTX 3090 / RTX 4090", 1),
                ("HighRAM_LowVRAM, profile 2 (Recommended): at least 48 GB of RAM and 12 GB of VRAM, the most versatile profile with high RAM, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos", 2),
                ("LowRAM_HighVRAM, profile 3: at least 32 GB of RAM and 24 GB of VRAM, adapted for RTX 3090 / RTX 4090 with limited RAM for good speed short video",3),
                ("LowRAM_LowVRAM, profile 4 (Default): at least 32 GB of RAM and 12 GB of VRAM, if you have little VRAM or want to generate longer videos",4),
                ("VerylowRAM_LowVRAM, profile 5: (Fail safe): at least 16 GB of RAM and 10 GB of VRAM, if you don't have much it won't be fast but maybe it will work",5)
                    ],
                    value= profile,
                    label="Profile (for power users only, not needed to change it)"
                 )

                default_ui_choice = gr.Dropdown(
                    choices=[
                        ("Text to Video", "t2v"),
                        ("Image to Video", "i2v"),
                    ],
                    value= default_ui,
                    label="Default mode when launching the App if not '--t2v' ot '--i2v' switch is specified when launching the server ",
                    visible= False ############
                 )                

                msg = gr.Markdown()            
                apply_btn  = gr.Button("Apply Changes")


        with gr.Row():
            with gr.Column():
                video_to_continue = gr.Video(label= "Video to continue", visible= use_image2video and False) #######  
                image_to_continue = gr.Image(label= "Image as a starting point for a new video", visible=use_image2video)

                prompt = gr.Textbox(label="Prompts (multiple prompts separated by carriage returns will generate multiple videos)", value="A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect.", lines=3)
                negative_prompt = gr.Textbox(label="Negative Prompt", value=NEGATIVE_PROMPT, lines=3)
                with gr.Row():
                    resolution = gr.Dropdown(
                        choices=[
                            # 720p
                            ("1280x720 (16:9, 720p)", "1280x720"),
                            ("720x1280 (9:16, 720p)", "720x1280"), 
                            ("1104x832 (4:3, 720p)", "1104x832"),
                            ("832x1104 (3:4, 720p)", "832x1104"),
                            ("960x960 (1:1, 720p)", "960x960"),
                            # 540p
                            ("960x544 (16:9, 540p)", "960x544"),
                            ("848x480 (16:9, 540p)", "848x480"),
                            ("544x960 (9:16, 540p)", "544x960"),
                            ("832x624 (4:3, 540p)", "832x624"), 
                            ("624x832 (3:4, 540p)", "624x832"),
                            ("720x720 (1:1, 540p)", "720x720"),
                        ],
                        value="848x480",
                        label="Resolution"
                    )

                video_length = gr.Slider(5, 193, value=97, step=4, label="Number of frames (24 = 1s)")

                    # video_length = gr.Dropdown(
                    #     label="Video Length",
                    #     choices=[
                    #         ("1.5s(41f)", 41),
                    #         ("2s(65f)", 65),
                    #         ("4s(97f)", 97),
                    #         ("5s(129f)", 129),
                    #     ],
                    #     value=97,
                    # )
                num_inference_steps = gr.Slider(1, 100, value=  default_inference_steps, step=1, label="Number of Inference Steps")
                seed = gr.Number(value=-1, label="Seed (-1 for random)")
                max_frames = gr.Slider(1, 100, value=9, step=1, label="Number of input frames to use for Video2World prediction", visible=use_image2video and False) #########
    

                loras_choices = gr.Dropdown(
                    choices=[
                        (lora_name, str(i) ) for i, lora_name in enumerate(loras_names)
                    ],
                    value= default_loras_choices,
                    multiselect= True,
                    visible= len(loras)>0,
                    label="Activated Loras"
                )
                loras_mult_choices = gr.Textbox(label="Loras Multipliers (1.0 by default) separated by space characters", value=default_loras_multis_str, visible= len(loras)>0 )

                show_advanced = gr.Checkbox(label="Show Advanced Options", value=False)
                with gr.Row(visible=False) as advanced_row:
                    with gr.Column():
                        guidance_scale = gr.Slider(1.0, 20.0, value=1.0, step=0.5, label="Guidance Scale")
                        flow_shift = gr.Slider(0.0, 25.0, value= default_flow_shift, step=0.1, label="Flow Shift") 
                        embedded_guidance_scale = gr.Slider(1.0, 20.0, value=6.0, step=0.5, label="Embedded Guidance Scale")
                        repeat_generation = gr.Slider(1, 25.0, value=1.0, step=1, label="Number of Generated Video per prompt") 
                        tea_cache_setting = gr.Dropdown(
                            choices=[
                                ("Disabled", 0),
                                ("Fast (x1.6 speed up)", 0.1), 
                                ("Faster (x2.1 speed up)", 0.15), 
                            ],
                            value=default_tea_cache,
                            label="Tea Cache acceleration (the faster the acceleration the higher the degradation of the quality of the video)"
                        )

                show_advanced.change(fn=lambda x: gr.Row(visible=x), inputs=[show_advanced], outputs=[advanced_row])
            
            with gr.Column():
                gen_status = gr.Text(label="Status", interactive= False) 
                output = gr.Gallery(
                        label="Generated videos", show_label=False, elem_id="gallery"
                    , columns=[3], rows=[1], object_fit="contain", height="auto", selected_index=0, interactive= False)
                generate_btn = gr.Button("Generate")
                abort_btn = gr.Button("Abort")

        gen_status.change(refresh_gallery, inputs = [state], outputs = output )

        abort_btn.click(abort_generation,state,abort_btn )
        output.select(select_video, state, None )

        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                negative_prompt,
                resolution,
                video_length,
                seed,
                num_inference_steps,
                guidance_scale,
                flow_shift,
                embedded_guidance_scale,
                repeat_generation,
                tea_cache_setting,
                loras_choices,
                loras_mult_choices,
                image_to_continue,
                video_to_continue,
                max_frames,
                state
            ],
            outputs= [gen_status] #,state 

        ).then( 
            finalize_gallery,
            [state], 
            [output , abort_btn]
        )

        apply_btn.click(
                fn=apply_changes,
                inputs=[
                    state,
                    transformer_t2v_choice,
                    transformer_i2v_choice,
                    text_encoder_choice,
                    attention_choice,
                    compile_choice,                            
                    profile_choice,
                    default_ui_choice,
                ],
                outputs= msg
            ).then( 
            update_defaults, 
            [state, num_inference_steps,  flow_shift], 
            [num_inference_steps,  flow_shift, header]
                )

    return demo

if __name__ == "__main__":
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_port = int(args.server_port)

    if server_port == 0:
        server_port = int(os.getenv("SERVER_PORT", "7860"))

    server_name = args.server_name
    if len(server_name) == 0:
        server_name = os.getenv("SERVER_NAME", "0.0.0.0")

        
    demo = create_demo()
    if args.open_browser:
        import webbrowser 
        if server_name.startswith("http"):
            url = server_name 
        else:
            url = "http://" + server_name 
        webbrowser.open(url + ":" + str(server_port), new = 0, autoraise = True)

    demo.launch(server_name=server_name, server_port=server_port)

 
