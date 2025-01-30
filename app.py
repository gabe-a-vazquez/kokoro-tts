import spaces
from kokoro import KModel, KPipeline
import gradio as gr
import os
import random
import torch

IS_DUPLICATE = not os.getenv('SPACE_ID', '').startswith('hexgrad/')
N_MAX_CHARS = None if IS_DUPLICATE else 5000
S_MAX_CHARS = '∞' if IS_DUPLICATE else str(N_MAX_CHARS)

CUDA_AVAILABLE = torch.cuda.is_available()

models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

@spaces.GPU(duration=10)
def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)

def return_audio_ps(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    text = text if N_MAX_CHARS is None else text.strip()[:N_MAX_CHARS]
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU. To avoid this error, change Hardware to CPU.')
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        return (24000, audio.numpy()), ps
    return None, ''

# Arena API
def predict(text, voice='af_heart', speed=1):
    return return_audio_ps(text, voice, speed, use_gpu=False)[0]

def return_ps(text, voice='af_heart'):
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ''

def yield_audio(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE):
    text = text if N_MAX_CHARS is None else text.strip()[:N_MAX_CHARS]
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Switching to CPU')
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(e)
        yield 24000, audio.numpy()

random_texts = {}
for lang in ['en']:
    with open(f'{lang}.txt', 'r') as r:
        random_texts[lang] = [line.strip() for line in r]

def get_random_text(voice):
    lang = dict(a='en', b='en')[voice[0]]
    return random.choice(random_texts[lang])

CHOICES = {
'🇺🇸 🚺 Heart ❤️': 'af_heart',
'🇺🇸 🚺 Bella 🔥': 'af_bella',
'🇺🇸 🚺 Nicole 🎧': 'af_nicole',
'🇺🇸 🚺 Aoede': 'af_aoede',
'🇺🇸 🚺 Kore': 'af_kore',
'🇺🇸 🚺 Sarah': 'af_sarah',
'🇺🇸 🚺 Nova': 'af_nova',
'🇺🇸 🚺 Sky': 'af_sky',
'🇺🇸 🚺 Alloy': 'af_alloy',
'🇺🇸 🚺 Jessica': 'af_jessica',
'🇺🇸 🚺 River': 'af_river',
'🇺🇸 🚹 Michael': 'am_michael',
'🇺🇸 🚹 Fenrir': 'am_fenrir',
'🇺🇸 🚹 Puck': 'am_puck',
'🇺🇸 🚹 Echo': 'am_echo',
'🇺🇸 🚹 Eric': 'am_eric',
'🇺🇸 🚹 Liam': 'am_liam',
'🇺🇸 🚹 Onyx': 'am_onyx',
'🇺🇸 🚹 Santa': 'am_santa',
'🇺🇸 🚹 Adam': 'am_adam',
'🇬🇧 🚺 Emma': 'bf_emma',
'🇬🇧 🚺 Isabella': 'bf_isabella',
'🇬🇧 🚺 Alice': 'bf_alice',
'🇬🇧 🚺 Lily': 'bf_lily',
'🇬🇧 🚹 George': 'bm_george',
'🇬🇧 🚹 Fable': 'bm_fable',
'🇬🇧 🚹 Lewis': 'bm_lewis',
'🇬🇧 🚹 Daniel': 'bm_daniel',
}
for v in CHOICES.values():
    pipelines[v[0]].load_voice(v)

TOKEN_NOTE = '''
💡 You can customize pronunciation like this: `[Kokoro](/kˈOkəɹO/)`

⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`

⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
'''

with gr.Blocks() as generate_tab:
    out_audio = gr.Audio(label='Output Audio', interactive=False, streaming=False, autoplay=True)
    generate_btn = gr.Button('Generate', variant='primary')
    with gr.Accordion('Output Tokens', open=False):
        out_ps = gr.Textbox(interactive=False, show_label=False, info='Tokens used to generate the audio, up to 510 context length.')
        tokenize_btn = gr.Button('Tokenize', variant='secondary')
        gr.Markdown(TOKEN_NOTE)
        predict_btn = gr.Button('Predict', variant='secondary', visible=False)

STREAM_NOTE = ['⚠️ There is an unknown Gradio bug that might yield no audio the first time you click `Stream`.']
if N_MAX_CHARS is not None:
    STREAM_NOTE.append(f'✂️ Each stream is capped at {N_MAX_CHARS} characters.')
    STREAM_NOTE.append('🚀 Want more characters? You can [use Kokoro directly](https://huggingface.co/hexgrad/Kokoro-82M#usage) or duplicate this space:')
STREAM_NOTE = '\n\n'.join(STREAM_NOTE)

with gr.Blocks() as stream_tab:
    out_stream = gr.Audio(label='Output Audio Stream', interactive=False, streaming=True, autoplay=True)
    with gr.Row():
        stream_btn = gr.Button('Stream', variant='primary')
        stop_btn = gr.Button('Stop', variant='stop')
    with gr.Accordion('Note', open=True):
        gr.Markdown(STREAM_NOTE)
        gr.DuplicateButton()

with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown('[***Kokoro*** **is an open-weight TTS model with 82 million parameters.**](https://hf.co/hexgrad/Kokoro-82M)', container=True)
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='Input Text', info=f'Up to ~500 characters per Generate, or {S_MAX_CHARS} characters per Stream')
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af_heart', label='Voice', info='Quality and availability vary by language')
                use_gpu = gr.Dropdown(
                    [('ZeroGPU 🚀', True), ('CPU 🐌', False)],
                    value=CUDA_AVAILABLE,
                    label='Hardware',
                    info='GPU is usually faster, but has a usage quota',
                    interactive=CUDA_AVAILABLE
                )
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='Speed')
            random_btn = gr.Button('Random Text', variant='secondary')
            random_btn.click(get_random_text, inputs=[voice], outputs=[text])
        with gr.Column():
            gr.TabbedInterface([generate_tab, stream_tab], ['Generate', 'Stream'])
    generate_btn.click(return_audio_ps, inputs=[text, voice, speed, use_gpu], outputs=[out_audio, out_ps])
    tokenize_btn.click(return_ps, inputs=[text, voice], outputs=[out_ps])
    stream_event = stream_btn.click(yield_audio, inputs=[text, voice, speed, use_gpu], outputs=[out_stream])
    stop_btn.click(fn=None, cancels=stream_event)
    predict_btn.click(predict, inputs=[text, voice, speed], outputs=[out_audio])

if IS_DUPLICATE:
    app.queue(api_open=True).launch(show_api=True, ssr_mode=True)
else:
    app.queue(api_open=False).load(api_name=False).launch(show_api=False, ssr_mode=True)
