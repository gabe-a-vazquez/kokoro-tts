from datetime import datetime
from hashlib import sha256
from huggingface_hub import snapshot_download
from katsu import Katsu
from models import build_model
import gradio as gr
import librosa
import numpy as np
import os
import phonemizer
import pypdf
import random
import re
import spaces
import subprocess
import torch
import yaml

CUDA_AVAILABLE = torch.cuda.is_available()

snapshot = snapshot_download(repo_id='hexgrad/kokoro', allow_patterns=['*.pt', '*.pth', '*.yml'], use_auth_token=os.environ['TOKEN'])
config = yaml.safe_load(open(os.path.join(snapshot, 'config.yml')))

models = {device: build_model(config['model_params'], device) for device in ['cpu'] + (['cuda'] if CUDA_AVAILABLE else [])}
for key, state_dict in torch.load(os.path.join(snapshot, 'net.pth'), map_location='cpu', weights_only=True)['net'].items():
    for device in models:
        assert key in models[device], key
        try:
            models[device][key].load_state_dict(state_dict)
        except:
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            models[device][key].load_state_dict(state_dict, strict=False)

PARAM_COUNT = sum(p.numel() for value in models['cpu'].values() for p in value.parameters())
assert PARAM_COUNT < 82_000_000, PARAM_COUNT
with open(os.path.join(snapshot, 'net.pth'), 'rb') as rb:
    model_hash = sha256(rb.read()).hexdigest()
    print('model_hash', model_hash)
    # SHA256 hash matches https://huggingface.co/hexgrad/Kokoro-82M/blob/main/kokoro-v0_19.pth
    assert model_hash == '3b0c392f87508da38fad3a2f9d94c359f1b657ebd2ef79f9d56d69503e470b0a'

random_texts = {}
for lang in ['en', 'fr', 'ja', 'ko', 'zh']:
    with open(f'{lang}.txt', 'r') as r:
        random_texts[lang] = [line.strip() for line in r]

def get_random_text(voice):
    lang = dict(a='en', b='en', f='fr', j='ja', k='ko', z='zh')[voice[0]]
    return random.choice(random_texts[lang])

sents = set()
for txt in {'harvard_sentences', 'llama3_command-r_sentences_1st_person', 'llama3_command-r_sentences_excla', 'llama3_command-r_questions'}:
    txt += '.txt'
    subprocess.run(['wget', f'https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena/resolve/main/{txt}'])
    with open(txt, 'r') as r:
        sents.update(r.read().strip().splitlines())
print('len(sents)', len(sents))

def parens_to_angles(s):
    return s.replace('(', '«').replace(')', '»')

def split_num(num):
    num = num.group()
    if '.' in num:
        return num
    elif ':' in num:
        h, m = [int(n) for n in num.split(':')]
        if m == 0:
            return f"{h} o'clock"
        elif m < 10:
            return f'{h} oh {m}'
        return f'{h} {m}'
    year = int(num[:4])
    if year < 1100 or year % 1000 < 10:
        return num
    left, right = num[:2], int(num[2:4])
    s = 's' if num.endswith('s') else ''
    if 100 <= year % 1000 <= 999:
        if right == 0:
            return f'{left} hundred{s}'
        elif right < 10:
            return f'{left} oh {right}{s}'
    return f'{left} {right}{s}'

def flip_money(m):
    m = m.group()
    bill = 'dollar' if m[0] == '$' else 'pound'
    if m[-1].isalpha():
        return f'{m[1:]} {bill}s'
    elif '.' not in m:
        s = '' if m[1:] == '1' else 's'
        return f'{m[1:]} {bill}{s}'
    b, c = m[1:].split('.')
    s = '' if b == '1' else 's'
    c = int(c.ljust(2, '0'))
    coins = f"cent{'' if c == 1 else 's'}" if m[0] == '$' else ('penny' if c == 1 else 'pence')
    return f'{b} {bill}{s} and {c} {coins}'

def point_num(num):
    a, b = num.group().split('.')
    return ' point '.join([a, ' '.join(b)])

def normalize_text(text, lang):
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace('«', chr(8220)).replace('»', chr(8221))
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    text = parens_to_angles(text)
    for a, b in zip('、。！，：；？', ',.!,:;?'):
        text = text.replace(a, b+' ')
    text = re.sub(r'[^\S \n]', ' ', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'(?<=\n) +(?=\n)', '', text)
    if lang == 'j':
        return text.strip()
    text = re.sub(r'\bD[Rr]\.(?= [A-Z])', 'Doctor', text)
    text = re.sub(r'\b(?:Mr\.|MR\.(?= [A-Z]))', 'Mister', text)
    text = re.sub(r'\b(?:Ms\.|MS\.(?= [A-Z]))', 'Miss', text)
    text = re.sub(r'\b(?:Mrs\.|MRS\.(?= [A-Z]))', 'Mrs', text)
    text = re.sub(r'\betc\.(?! [A-Z])', 'etc', text)
    text = re.sub(r'(?i)\b(y)eah?\b', r"\1e'a", text)
    text = re.sub(r'\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)', split_num, text)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = re.sub(r'(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b', flip_money, text)
    text = re.sub(r'\d*\.\d+', point_num, text)
    text = re.sub(r'(?<=\d)-(?=\d)', ' to ', text) # TODO: could be minus
    text = re.sub(r'(?<=\d)S', ' S', text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", 's', text)
    text = re.sub(r'(?:[A-Za-z]\.){2,} [a-z]', lambda m: m.group().replace('.', '-'), text)
    text = re.sub(r'(?i)(?<=[A-Z])\.(?=[A-Z])', '-', text)
    return text.strip()

phonemizers = dict(
    a=phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True),
    b=phonemizer.backend.EspeakBackend(language='en-gb', preserve_punctuation=True, with_stress=True),
    j=Katsu(),
)

# Starred voices are more stable
CHOICES = {
'🇺🇸 🚺 American Female ⭐': 'af',
'🇺🇸 🚺 Bella ⭐': 'af_bella',
'🇺🇸 🚺 Nicole ⭐': 'af_nicole',
'🇺🇸 🚺 Sarah ⭐': 'af_sarah',
'🇺🇸 🚺 American Female 1': 'af_1',
'🇺🇸 🚺 Alloy': 'af_alloy',
'🇺🇸 🚺 Jessica': 'af_jessica',
'🇺🇸 🚺 Nova': 'af_nova',
'🇺🇸 🚺 River': 'af_river',
'🇺🇸 🚺 Sky': 'af_sky',
'🇺🇸 🚹 Michael ⭐': 'am_michael',
'🇺🇸 🚹 Adam': 'am_adam',
'🇺🇸 🚹 Echo': 'am_echo',
'🇺🇸 🚹 Eric': 'am_eric',
'🇺🇸 🚹 Liam': 'am_liam',
'🇺🇸 🚹 Onyx': 'am_onyx',
'🇬🇧 🚺 British Female 0': 'bf_0',
'🇬🇧 🚺 British Female 1': 'bf_1',
'🇬🇧 🚺 British Female 2': 'bf_2',
'🇬🇧 🚺 British Female 3': 'bf_3',
'🇬🇧 🚺 Alice': 'bf_alice',
'🇬🇧 🚺 Lily': 'bf_lily',
'🇬🇧 🚹 British Male 0': 'bm_0',
'🇬🇧 🚹 British Male 1': 'bm_1',
'🇬🇧 🚹 Daniel': 'bm_daniel',
'🇬🇧 🚹 Fable': 'bm_fable',
'🇬🇧 🚹 George': 'bm_george',
'🇬🇧 🚹 Lewis': 'bm_lewis',
'🇯🇵 🚺 Japanese Female ⭐': 'jf_0',
'🇯🇵 🚺 Japanese Female 1': 'jf_1',
'🇯🇵 🚺 Japanese Female 2': 'jf_2',
'🇯🇵 🚺 Japanese Female 3': 'jf_3',
}
VOICES = {device: {k: torch.load(os.path.join(snapshot, 'voicepacks', f'{k}.pt'), weights_only=True).to(device) for k in CHOICES.values()} for device in models}

def resolve_voices(voice, warn=True):
    if not isinstance(voice, str) or voice == list(CHOICES.keys())[0]:
        return ['af']
    voices = voice.lower().replace(' ', '+').replace(',', '+').split('+')
    if warn:
        unks = {v for v in voices if v and v not in VOICES['cpu']}
        if unks:
            gr.Warning(f"Unknown voice{'s' if len(unks) > 1 else ''}: {','.join(unks)}")
    voices = [v for v in voices if v in VOICES['cpu']]
    return voices if voices else ['af']

def get_vocab():
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»“” '
    _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    dicts = {}
    for i in range(len((symbols))):
        dicts[symbols[i]] = i
    return dicts

VOCAB = get_vocab()

def tokenize(ps):
    return [i for i in map(VOCAB.get, ps) if i is not None]

def phonemize(text, voice, norm=True):
    lang = resolve_voices(voice)[0][0]
    if norm:
        text = normalize_text(text, lang)
    ps = phonemizers[lang].phonemize([text])
    ps = ps[0] if ps else ''
    # TODO: Custom phonemization rules?
    ps = parens_to_angles(ps)
    # https://en.wiktionary.org/wiki/kokoro#English
    if lang in 'ab':
        ps = ps.replace('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ').replace('kəkˈɔːɹəʊ', 'kˈəʊkəɹəʊ')
        ps = ps.replace('ʲ', 'j').replace('r', 'ɹ').replace('x', 'k').replace('ɬ', 'l')
        ps = re.sub(r'(?<=[a-zɹː])(?=hˈʌndɹɪd)', ' ', ps)
        ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»“” ]|$)', 'z', ps)
        if lang == 'a':
            ps = re.sub(r'(?<=nˈaɪn)ti(?!ː)', 'di', ps)
    ps = ''.join(filter(lambda p: p in VOCAB, ps))
    if lang == 'j' and any(p in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for p in ps):
        gr.Warning('Japanese tokenizer does not handle English letters')
    return ps.strip()

harvard_sentences = set()
with open('harvard_sentences.txt', 'r') as r:
    for line in r:
        harvard_sentences.add(phonemize(line, 'af'))
        harvard_sentences.add(phonemize(line, 'bf_0'))

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

SAMPLE_RATE = 24000

@torch.no_grad()
def forward(tokens, voices, speed, sk, device='cpu'):
    assert sk in {os.environ['SK'], os.environ['ARENA'], os.environ['TEMP']}, sk
    ref_s = torch.mean(torch.stack([VOICES[device][v][len(tokens)] for v in voices]), dim=0)
    tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    bert_dur = models[device].bert(tokens, attention_mask=(~text_mask).int())
    d_en = models[device].bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = models[device].predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = models[device].predictor.lstm(d)
    duration = models[device].predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long()
    pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item())
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + pred_dur[0,i].item()] = 1
        c_frame += pred_dur[0,i].item()
    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
    F0_pred, N_pred = models[device].predictor.F0Ntrain(en, s)
    t_en = models[device].text_encoder(tokens, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
    return models[device].decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu().numpy()

@spaces.GPU(duration=10)
def forward_gpu(tokens, voices, speed, sk):
    return forward(tokens, voices, speed, sk, device='cuda')

def clamp_speed(speed):
    if not isinstance(speed, float) and not isinstance(speed, int):
        return 1
    elif speed < 0.5:
        return 0.5
    elif speed > 2:
        return 2
    return speed

def clamp_trim(trim):
    if not isinstance(trim, float) and not isinstance(trim, int):
        return 0.5
    elif trim < 0:
        return 0
    elif trim > 1:
        return 0.5
    return trim

def trim_if_needed(out, trim):
    if not trim:
        return out
    a, b = librosa.effects.trim(out, top_db=30)[1]
    a = int(a*trim)
    b = int(len(out)-(len(out)-b)*trim)
    return out[a:b]

# Must be backwards compatible with https://huggingface.co/spaces/Pendrokar/TTS-Spaces-Arena
def generate(text, voice='af', ps=None, speed=1, trim=0.5, use_gpu='auto', sk=None):
    if not text.strip():
        return (None, '')
    ps = ps or phonemize(text, voice)
    if sk not in {os.environ['SK'], os.environ['ARENA'], os.environ['TEMP']}:
        assert text in sents or ps.strip('"') in harvard_sentences, ('❌', datetime.now(), text, voice, use_gpu, sk)
        sk = os.environ['ARENA']
    voices = resolve_voices(voice, warn=ps)
    speed = clamp_speed(speed)
    trim = clamp_trim(trim)
    use_gpu = use_gpu if use_gpu in ('auto', False, True) else 'auto'
    tokens = tokenize(ps)
    if not tokens:
        return (None, '')
    elif len(tokens) > 510:
        tokens = tokens[:510]
    ps = ''.join(next(k for k, v in VOCAB.items() if i == v) for i in tokens)
    use_gpu = len(ps) > 99 if use_gpu == 'auto' else use_gpu
    debug = '🔥' if sk == os.environ['SK'] else '🏆'
    try:
        if use_gpu:
            out = forward_gpu(tokens, voices, speed, sk)
        else:
            out = forward(tokens, voices, speed, sk)
    except gr.exceptions.Error as e:
        if use_gpu:
            gr.Warning(str(e))
            gr.Info('Switching to CPU')
            out = forward(tokens, voices, speed, sk)
        else:
            raise gr.Error(e)
            print(debug, datetime.now(), voices, repr(text), len(ps), use_gpu, repr(e))
            return (None, '')
    out = trim_if_needed(out, trim)
    print(debug, datetime.now(), voices, repr(text), len(ps), use_gpu, len(out))
    return ((SAMPLE_RATE, out), ps)

def toggle_autoplay(autoplay):
    return gr.Audio(interactive=False, label='Output Audio', autoplay=autoplay)

ML_LANGUAGES = {
'🇺🇸 en-US': 'a',
'🇬🇧 en-GB': 'b',
'🇫🇷 fr-FR': 'f',
'🇯🇵 ja-JP': 'j',
'🇰🇷 ko-KR': 'k',
'🇨🇳 zh-CN': 'z',
}

from gradio_client import Client
client = Client('hexgrad/kokoro-src', hf_token=os.environ['SRC'])
import json
ML_CHOICES = json.loads(client.predict(api_name='/list_voices'))
DEFAULT_VOICE = list(ML_CHOICES['a'].values())[0]
def change_language(value):
    choices = list(ML_CHOICES[value].items())
    return gr.Dropdown(choices, value=choices[0][1], label='Voice', info='⭐ voices are stable, 🧪 are unstable')

def multilingual(text, voice, speed, trim, sk):
    if not text.strip():
        return None
    assert sk == os.environ['SK'], ('❌', datetime.now(), text, voice, sk)
    try:
        audio, out_ps = client.predict(text=text, voice=voice, speed=speed, trim=trim, use_gpu=True, sk=sk, api_name='/generate')
        if len(out_ps) == 510:
            gr.Warning('Input may have been truncated')
    except Exception as e:
        print('📡', datetime.now(), text, voice, repr(e))
        gr.Warning('v0.23 temporarily unavailable')
        gr.Info('Switching to v0.19')
        audio = generate(text, voice=voice, speed=speed, trim=trim, sk=sk)[0]
    return audio

with gr.Blocks() as ml_tts:
    with gr.Row():
        lang = gr.Radio(choices=ML_LANGUAGES.items(), value='a', label='Language', show_label=False)
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='Input Text', info='Generate speech for one segment of text, up to ~500 characters')
            voice = gr.Dropdown(list(ML_CHOICES['a'].items()), value=DEFAULT_VOICE, label='Voice', info='⭐ voices are stable, 🧪 are unstable')
            lang.change(fn=change_language, inputs=[lang], outputs=[voice])
            with gr.Row():
                random_btn = gr.Button('Random Text', variant='secondary')
                generate_btn = gr.Button('Generate', variant='primary')
            random_btn.click(get_random_text, inputs=[lang], outputs=[text])
        with gr.Column():
            audio = gr.Audio(interactive=False, label='Output Audio', autoplay=True)
            with gr.Accordion('Audio Settings', open=False):
                autoplay = gr.Checkbox(value=True, label='Autoplay')
                autoplay.change(toggle_autoplay, inputs=[autoplay], outputs=[audio])
                speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='⚡️ Speed', info='Adjust the speaking speed')
                trim = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label='✂️ Trim', info='How much to cut from both ends')
    with gr.Row():
        gr.Markdown('''
❗ **This space is experiencing heavy lag, possibly due to high traffic.**

🎄 Kokoro v0.19, Bella, & Sarah have been open sourced at [hf.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

🎉 New! Kokoro v0.23 now supports 5 languages. 🎉

🧪 Note that v0.23 is experimental/WIP and may produce shaky speech. v0.19 is the last stable version.

⚠️ v0.23 does not yet support custom pronunciation, Long Form, or Voice Mixer. You can still use these features in v0.19.

📡 Telemetry: For debugging purposes, the text you enter anywhere in this space may be printed to temporary logs, which are periodically wiped.

🇨🇳🇯🇵🇰🇷 Tokenizers for Chinese, Japanese, and Korean do not correctly handle English letters yet. Remove or convert them to CJK first.
''', container=True)
    with gr.Row():
        sk = gr.Textbox(visible=False)
    text.change(lambda: os.environ['SK'], outputs=[sk])
    text.submit(multilingual, inputs=[text, voice, speed, trim, sk], outputs=[audio])
    generate_btn.click(multilingual, inputs=[text, voice, speed, trim, sk], outputs=[audio])

USE_GPU_CHOICES = [('Auto 🔀', 'auto'), ('CPU 💬', False), ('ZeroGPU 📄', True)]
USE_GPU_INFOS = {
    'auto': 'Use CPU or GPU, whichever is faster',
    False: 'CPU is ~faster <100 tokens',
    True: 'ZeroGPU is ~faster >100 tokens',
}
def change_use_gpu(value):
    return gr.Dropdown(USE_GPU_CHOICES, value=value, label='Hardware', info=USE_GPU_INFOS[value], interactive=CUDA_AVAILABLE)

with gr.Blocks() as basic_tts:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='Input Text', info='Generate speech for one segment of text using Kokoro, a TTS model with 82 million parameters')
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af', allow_custom_value=True, label='Voice', info='Starred voices are more stable')
                use_gpu = gr.Dropdown(
                    USE_GPU_CHOICES,
                    value='auto' if CUDA_AVAILABLE else False,
                    label='Hardware',
                    info=USE_GPU_INFOS['auto' if CUDA_AVAILABLE else False],
                    interactive=CUDA_AVAILABLE
                )
                use_gpu.change(fn=change_use_gpu, inputs=[use_gpu], outputs=[use_gpu])
            with gr.Row():
                random_btn = gr.Button('Random Text', variant='secondary')
                generate_btn = gr.Button('Generate', variant='primary')
            random_btn.click(get_random_text, inputs=[voice], outputs=[text])
            with gr.Accordion('Input Tokens', open=False):
                in_ps = gr.Textbox(show_label=False, info='Override the input text with custom phonemes. Leave this blank to automatically tokenize the input text instead.')
                with gr.Row():
                    clear_btn = gr.ClearButton(in_ps)
                    phonemize_btn = gr.Button('Tokenize Input Text', variant='primary')
            phonemize_btn.click(phonemize, inputs=[text, voice], outputs=[in_ps])
        with gr.Column():
            audio = gr.Audio(interactive=False, label='Output Audio', autoplay=True)
            with gr.Accordion('Audio Settings', open=False):
                autoplay = gr.Checkbox(value=True, label='Autoplay')
                autoplay.change(toggle_autoplay, inputs=[autoplay], outputs=[audio])
                speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='⚡️ Speed', info='Adjust the speaking speed')
                trim = gr.Slider(minimum=0, maximum=1, value=0.5, step=0.1, label='✂️ Trim', info='How much to cut from both ends of each segment')
            with gr.Accordion('Output Tokens', open=True):
                out_ps = gr.Textbox(interactive=False, show_label=False, info='Tokens used to generate the audio, up to 510 allowed. Same as input tokens if supplied, excluding unknowns.')
    with gr.Accordion('Voice Mixer', open=False):
        gr.Markdown('Create a custom voice by mixing and matching other voices. Click an orange button to add one part to your mix, or click a gray button to start over. You can also enter a voice mix as text.')
        for i in range(8):
            with gr.Row():
                for j in range(4):
                    with gr.Column():
                        btn = gr.Button(list(CHOICES.values())[i*4+j], variant='primary' if i*4+j < 10 else 'secondary')
                        btn.click(lambda v, b: f'{v}+{b}' if v.startswith(b[:2]) else b, inputs=[voice, btn], outputs=[voice])
                        voice.change(lambda v, b: gr.Button(b, variant='primary' if v.startswith(b[:2]) else 'secondary'), inputs=[voice, btn], outputs=[btn])
    with gr.Row():
        sk = gr.Textbox(visible=False)
    text.change(lambda: os.environ['SK'], outputs=[sk])
    text.submit(generate, inputs=[text, voice, in_ps, speed, trim, use_gpu, sk], outputs=[audio, out_ps])
    generate_btn.click(generate, inputs=[text, voice, in_ps, speed, trim, use_gpu, sk], outputs=[audio, out_ps])

@torch.no_grad()
def lf_forward(token_lists, voices, speed, sk, device='cpu'):
    assert sk == os.environ['SK'], sk
    voicepack = torch.mean(torch.stack([VOICES[device][v] for v in voices]), dim=0)
    outs = []
    for tokens in token_lists:
        ref_s = voicepack[len(tokens)]
        s = ref_s[:, 128:]
        tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        bert_dur = models[device].bert(tokens, attention_mask=(~text_mask).int())
        d_en = models[device].bert_encoder(bert_dur).transpose(-1, -2)
        d = models[device].predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = models[device].predictor.lstm(d)
        duration = models[device].predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long()
        pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item())
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + pred_dur[0,i].item()] = 1
            c_frame += pred_dur[0,i].item()
        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        F0_pred, N_pred = models[device].predictor.F0Ntrain(en, s)
        t_en = models[device].text_encoder(tokens, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        outs.append(models[device].decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu().numpy())
    return outs

@spaces.GPU
def lf_forward_gpu(token_lists, voices, speed, sk):
    return lf_forward(token_lists, voices, speed, sk, device='cuda')

def resplit_strings(arr):
    # Handle edge cases
    if not arr:
        return '', ''
    if len(arr) == 1:
        return arr[0], ''
    # Try each possible split point
    min_diff = float('inf')
    best_split = 0
    # Calculate lengths when joined with spaces
    lengths = [len(s) for s in arr]
    spaces = len(arr) - 1  # Total spaces needed
    # Try each split point
    left_len = 0
    right_len = sum(lengths) + spaces
    for i in range(1, len(arr)):
        # Add current word and space to left side
        left_len += lengths[i-1] + (1 if i > 1 else 0)
        # Remove current word and space from right side
        right_len -= lengths[i-1] + 1
        diff = abs(left_len - right_len)
        if diff < min_diff:
            min_diff = diff
            best_split = i
    # Join the strings with the best split point
    return ' '.join(arr[:best_split]), ' '.join(arr[best_split:])

def recursive_split(text, voice):
    if not text:
        return []
    tokens = phonemize(text, voice, norm=False)
    if len(tokens) < 511:
        return [(text, tokens, len(tokens))] if tokens else []
    if ' ' not in text:
        return []
    for punctuation in ['!.?…', ':;', ',—']:
        splits = re.split(f'(?:(?<=[{punctuation}])|(?<=[{punctuation}]["\'»])|(?<=[{punctuation}]["\'»]["\'»])) ', text)
        if len(splits) > 1:
            break
        else:
            splits = None
    splits = splits or text.split(' ')
    a, b = resplit_strings(splits)
    return recursive_split(a, voice) + recursive_split(b, voice)

def segment_and_tokenize(text, voice, skip_square_brackets=True, newline_split=2):
    lang = resolve_voices(voice)[0][0]
    if skip_square_brackets:
        text = re.sub(r'\[.*?\]', '', text)
    texts = [t.strip() for t in re.split('\n{'+str(newline_split)+',}', normalize_text(text, lang))] if newline_split > 0 else [normalize_text(text, lang)]
    segments = [row for t in texts for row in recursive_split(t, voice)]
    return [(i, *row) for i, row in enumerate(segments)]

def lf_generate(segments, voice, speed=1, trim=0, pad_between=0, use_gpu=True, sk=None):
    if sk != os.environ['SK']:
        return
    token_lists = list(map(tokenize, segments['Tokens']))
    voices = resolve_voices(voice)
    speed = clamp_speed(speed)
    trim = clamp_trim(trim)
    pad_between = int(pad_between)
    use_gpu = True
    batch_sizes = [89, 55, 34, 21, 13, 8, 5, 3, 2, 1, 1]
    i = 0
    while i < len(token_lists):
        bs = batch_sizes.pop() if batch_sizes else 100
        tokens = token_lists[i:i+bs]
        print('📖', datetime.now(), len(tokens), voices, use_gpu, ''.join(segments['Text'][i:i+bs]).replace('\n', ' '))
        try:
            if use_gpu:
                outs = lf_forward_gpu(tokens, voices, speed, sk)
            else:
                outs = lf_forward(tokens, voices, speed, sk)
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Switching to CPU')
                outs = lf_forward(tokens, voices, speed, sk)
                use_gpu = False
            elif outs:
                gr.Warning(repr(e))
                i = len(token_lists)
            else:
                raise gr.Error(e)
        for out in outs:
            if i > 0 and pad_between > 0:
                yield (SAMPLE_RATE, np.zeros(pad_between))
            out = trim_if_needed(out, trim)
            yield (SAMPLE_RATE, out)
        i += bs

def did_change_segments(segments):
    x = len(segments) if segments['Length'].any() else 0
    return [
        gr.Button('Tokenize', variant='secondary' if x else 'primary'),
        gr.Button(f'Generate x{x}', variant='primary' if x else 'secondary', interactive=x > 0),
    ]

def extract_text(file):
    if file.endswith('.pdf'):
        with open(file, 'rb') as rb:
            pdf_reader = pypdf.PdfReader(rb)
            return '\n'.join([page.extract_text() for page in pdf_reader.pages])
    elif file.endswith('.txt'):
        with open(file, 'r') as r:
            return '\n'.join([line for line in r])
    return None

with gr.Blocks() as lf_tts:
    with gr.Row():
        with gr.Column():
            file_input = gr.File(file_types=['.pdf', '.txt'], label='pdf or txt')
            text = gr.Textbox(label='Input Text', info='Generate speech in batches of 100 text segments and automatically join them together')
            file_input.upload(fn=extract_text, inputs=[file_input], outputs=[text])
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af', allow_custom_value=True, label='Voice', info='Starred voices are more stable')
                use_gpu = gr.Dropdown(
                    [('ZeroGPU 🚀', True), ('CPU 🐌', False)],
                    value=CUDA_AVAILABLE,
                    label='Hardware',
                    info='GPU is >10x faster but has a usage quota',
                    interactive=CUDA_AVAILABLE
                )
            with gr.Accordion('Text Settings', open=False):
                skip_square_brackets = gr.Checkbox(True, label='Skip [Square Brackets]', info='Recommended for academic papers, Wikipedia articles, or texts with citations')
                newline_split = gr.Number(2, label='Newline Split', info='Split the input text on this many newlines. Affects how the text is segmented.', precision=0, minimum=0)
        with gr.Column():
            audio_stream = gr.Audio(label='Output Audio Stream', interactive=False, streaming=True, autoplay=True)
            with gr.Accordion('Audio Settings', open=True):
                speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='⚡️ Speed', info='Adjust the speaking speed')
                trim = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label='✂️ Trim', info='How much to cut from both ends')
                pad_between = gr.Slider(minimum=0, maximum=24000, value=0, step=1000, label='🔇 Pad Between', info='How many silent samples to insert between segments')
            with gr.Row():
                segment_btn = gr.Button('Tokenize', variant='primary')
                generate_btn = gr.Button('Generate x0', variant='secondary', interactive=False)
                stop_btn = gr.Button('Stop', variant='stop')
    with gr.Row():
        segments = gr.Dataframe(headers=['#', 'Text', 'Tokens', 'Length'], row_count=(1, 'dynamic'), col_count=(4, 'fixed'), label='Segments', interactive=False, wrap=True)
        segments.change(fn=did_change_segments, inputs=[segments], outputs=[segment_btn, generate_btn])
    with gr.Row():
        sk = gr.Textbox(visible=False)
    segments.change(lambda: os.environ['SK'], outputs=[sk])
    segment_btn.click(segment_and_tokenize, inputs=[text, voice, skip_square_brackets, newline_split], outputs=[segments])
    generate_event = generate_btn.click(lf_generate, inputs=[segments, voice, speed, trim, pad_between, use_gpu, sk], outputs=[audio_stream])
    stop_btn.click(fn=None, cancels=generate_event)

with gr.Blocks() as about:
    gr.Markdown('''
Kokoro is a frontier TTS model for its size. It has [82 million](https://hf.co/spaces/hexgrad/Kokoro-TTS/blob/main/app.py#L34) parameters, uses a lean [StyleTTS 2](https://github.com/yl4579/StyleTTS2) architecture, and was trained on high-quality data. The weights are currently private, but a free public demo is hosted here, at `https://hf.co/spaces/hexgrad/Kokoro-TTS`. The Community tab is open for feature requests, bug reports, etc. For other inquiries, contact `@rzvzn` on Discord.

### FAQ
**Will this be open sourced?**<br/>
v0.19 has been open sourced at [hf.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) along with the voicepacks Bella, Sarah, and `af`. There currently isn't a release date scheduled for the other voices.

**What is the difference between stable and unstable voices?**<br/>
Unstable voices are more likely to stumble or produce unnatural artifacts, especially on short or strange texts. Stable voices are more likely to deliver natural speech on a wider range of inputs. The first two audio clips in this [blog post](https://hf.co/blog/hexgrad/kokoro-short-burst-upgrade) are examples of unstable and stable speech. Note that even unstable voices can sound fine on medium to long texts.

**How can CPU be faster than ZeroGPU?**<br/>
The CPU is a dedicated resource for this Space, while the ZeroGPU pool is shared and dynamically allocated across all of HF. The ZeroGPU queue/allocator system inevitably adds latency to each request.<br/>
For Basic TTS under ~100 tokens or characters, only a few seconds of audio need to be generated, so the actual compute is not that heavy. In these short bursts, the dedicated CPU can often compute the result faster than the total time it takes to: enter the ZeroGPU queue, wait to get allocated, and have a GPU compute and deliver the result.<br/>
ZeroGPU catches up beyond 100 tokens and especially closer to the ~500 token context window. Long Form mode processes batches of 100 segments at a time, so the GPU should outspeed the CPU by 1-2 orders of magnitude.

### Compute
Kokoro v0.19 was trained on A100 80GB vRAM instances for approximately 500 total GPU hours. The average cost for each GPU hour was around $0.80, so the total cost was around $400.

### Gradio API
The API has been restricted due to high request volume impacting CPU latency.

### Licenses
Inference code: MIT<br/>
[eSpeak NG](https://github.com/espeak-ng/espeak-ng): GPL-3.0<br/>
Random English texts: Unknown from [Quotable Data](https://github.com/quotable-io/data/blob/master/data/quotes.json)<br/>
Other random texts: CC0 public domain from [Common Voice](https://github.com/common-voice/common-voice)
''')
'''
This Space can be used via API. The following code block can be copied and run in one Google Colab cell.
```
# 1️⃣ Install the Gradio Python client
!pip install -q gradio_client
# 2️⃣ Initialize the client
from gradio_client import Client
client = Client('hexgrad/Kokoro-TTS')
# 3️⃣ Call the generate endpoint, which returns a pair: an audio path and a string of output phonemes
audio_path, out_ps = client.predict(
    text="How could I know? It's an unanswerable question. Like asking an unborn child if they'll lead a good life. They haven't even been born.",
    voice='af',
    api_name='/generate'
)
# 4️⃣ Display the audio and print the output phonemes
from IPython.display import display, Audio
display(Audio(audio_path, autoplay=True))
print(out_ps)
```
This Space and the underlying Kokoro model are both under development and subject to change. Reliability is not guaranteed. Hugging Face and Gradio might enforce their own rate limits.
'''
with gr.Blocks() as changelog:
    gr.Markdown('''
**25 Dec 2024**<br/>
🎄 Kokoro v0.19, Bella, & Sarah have been open sourced at [hf.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)

**11 Dec 2024**<br/>
🚀 Multilingual v0.23<br/>
🗣️ 85 total voices

**8 Dec 2024**<br/>
🚀 Multilingual v0.22<br/>
🌐 5 languages: English, Chinese, Japanese, Korean, French<br/>
🗣️ 68 total voices<br/>
📁 Added data card and telemetry notice

**30 Nov 2024**<br/>
✂️ Better trimming with `librosa.effects.trim`<br/>
🏆 https://hf.co/spaces/Pendrokar/TTS-Spaces-Arena

**28 Nov 2024**<br/>
🥈 CPU fallback<br/>
🌊 Long Form streaming and stop button<br/>
✋ Restricted API due to high request volume impacting CPU latency

**25 Nov 2024**<br/>
🎨 Voice Mixer added

**24 Nov 2024**<br/>
🛑 Model training halted, v0.19 is the current stable version

**23 Nov 2024**<br/>
🔀 Hardware switching between CPU and GPU<br/>
🗣️ Restored old voices, back up to 32 total

**22 Nov 2024**<br/>
🚀 Model v0.19<br/>
🧪 Validation losses: 0.261 mel, 0.627 dur, 1.897 f0<br/>
📄 https://hf.co/blog/hexgrad/kokoro-short-burst-upgrade

**15 Nov 2024**<br/>
🚀 Model v0.16<br/>
🧪 Validation losses: 0.263 mel, 0.646 dur, 1.934 f0

**12 Nov 2024**<br/>
🚀 Model v0.14<br/>
🧪 Validation losses: 0.262 mel, 0.642 dur, 1.889 f0
''')

with gr.Blocks() as data_card:
    gr.Markdown('''
This data card was last updated on **8 Dec 2024**.

Kokoro was trained exclusively on **permissive/non-copyrighted audio data** and IPA phoneme labels. Examples of permissive/non-copyrighted audio include:
* Public domain audio
* Audio licensed under Apache, MIT, etc
* Synthetic audio<sup>[1]</sup> generated by closed<sup>[2]</sup> TTS models from large providers
* CC BY audio (see below for attribution table)

[1] [https://copyright.gov/ai/ai_policy_guidance.pdf](https://copyright.gov/ai/ai_policy_guidance.pdf)<br/>
[2] No synthetic audio from open TTS models or "custom voice clones"

### Creative Commons Attribution
The following CC BY audio was part of the dataset used to train Kokoro.

| Audio Data | Duration Used | License | Added to Training Set After |
| ---------- | ------------- | ------- | --------------------------- |
| [Koniwa](https://github.com/koniwa/koniwa) `tnc` | <1h | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/deed.ja) | v0.19 / 22 Nov 2024 |
| [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353) | <11h | [CC BY 4.0](https://datashare.ed.ac.uk/bitstream/handle/10283/2353/license_text) | v0.19 / 22 Nov 2024 |

### Notable Datasets Not Used
These datasets were **NOT** used to train Kokoro. They may be of interest to academics:
* Emilia, `cc-by-nc-4.0`: `https://huggingface.co/datasets/amphion/Emilia-Dataset`
* Expresso, `cc-by-nc-4.0`: `https://huggingface.co/datasets/ylacombe/expresso`
* JVS, NC clause: `https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus`
''')

with gr.Blocks() as app:
    gr.TabbedInterface(
        [ml_tts, basic_tts, lf_tts, about, data_card, changelog],
        ['🔥 Latest v0.23', '🗣️ TTS v0.19', '📖 Long Form v0.19', 'ℹ️ About', '📁 Data', '📝 Changelog'],
    )

if __name__ == '__main__':
    app.queue(api_open=True).launch(show_api=False, srr_mode=True)
