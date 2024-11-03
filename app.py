from huggingface_hub import snapshot_download
from katsu import Katsu
from models import build_model
import gradio as gr
import noisereduce as nr
import numpy as np
import os
import phonemizer
import random
import spaces
import torch
import yaml

random_texts = {}
for lang in ['en', 'ja']:
    with open(f'{lang}.txt', 'r') as r:
        random_texts[lang] = [line.strip() for line in r]

def get_random_text(voice):
    if voice[0] == 'j':
        lang = 'ja'
    else:
        lang = 'en'
    return random.choice(random_texts[lang])

def parens_to_angles(s):
    return s.replace('(', '«').replace(')', '»')

def normalize(text):
    # TODO: Custom text normalization rules?
    text = text.replace('Dr.', 'Doctor')
    text = text.replace('Mr.', 'Mister')
    text = text.replace('Ms.', 'Miss')
    text = text.replace('Mrs.', 'Mrs')
    return parens_to_angles(text)

phonemizers = dict(
    a=phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True),
    b=phonemizer.backend.EspeakBackend(language='en-gb', preserve_punctuation=True, with_stress=True),
    j=Katsu()
)

def phonemize(text, voice):
    lang = voice[0]
    text = normalize(text)
    ps = phonemizers[lang].phonemize([text])
    ps = ps[0] if ps else ''
    # TODO: Custom phonemization rules?
    ps = parens_to_angles(ps)
    # https://en.wiktionary.org/wiki/kokoro#English
    ps = ps.replace('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ').replace('kəkˈɔːɹəʊ', 'kˈəʊkəɹəʊ')
    ps = ''.join(filter(lambda p: p in VOCAB, ps))
    return ps.strip()

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

snapshot = snapshot_download(repo_id='hexgrad/kokoro', allow_patterns=['*.pt', '*.pth', '*.yml'], use_auth_token=os.environ['TOKEN'])
config = yaml.safe_load(open(os.path.join(snapshot, 'config.yml')))
model = build_model(config['model_params'])
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]
for key, state_dict in torch.load(os.path.join(snapshot, 'net.pth'), map_location='cpu', weights_only=True)['net'].items():
    assert key in model, key
    try:
        model[key].load_state_dict(state_dict)
    except:
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model[key].load_state_dict(state_dict, strict=False)

CHOICES = {
    '🇺🇸 🚺 American Female 0': 'af0',
    '🇺🇸 🚺 Bella': 'af1',
    '🇺🇸 🚺 Nicole': 'af2',
    '🇺🇸 🚹 Michael': 'am0',
    '🇺🇸 🚹 Adam': 'am1',
    '🇬🇧 🚺 British Female 0': 'bf0',
    '🇬🇧 🚺 British Female 1': 'bf1',
    '🇬🇧 🚺 British Female 2': 'bf2',
    '🇬🇧 🚹 British Male 0': 'bm0',
    '🇬🇧 🚹 British Male 1': 'bm1',
    '🇬🇧 🚹 British Male 2': 'bm2',
    '🇬🇧 🚹 British Male 3': 'bm3',
    '🇯🇵 🚺 Japanese Female 0': 'jf0',
}
VOICES = {k: torch.load(os.path.join(snapshot, 'voices', f'{k}.pt'), weights_only=True).to(device) for k in CHOICES.values()}

np_log_99 = np.log(99)
def s_curve(p):
    if p <= 0:
        return 0
    elif p >= 1:
        return 1
    s = 1 / (1 + np.exp((1-p*2)*np_log_99))
    s = (s-0.01) * 50/49
    return s

SAMPLE_RATE = 24000

@spaces.GPU(duration=10)
@torch.no_grad()
def forward(tokens, speed):
    tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    ref_s = VOICES[voice]
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration.squeeze()).clamp(min=1)
    pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
        c_frame += int(pred_dur[i].data)
    en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(tokens, input_lengths, text_mask)
    asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
    out = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128])
    return out.squeeze().cpu().numpy()

def generate(text, voice, ps=None, speed=1.0, reduce_noise=0.5, opening_cut=5000, closing_cut=0, ease_in=3000, ease_out=0):
    ps = ps or phonemize(text, voice)
    tokens = [i for i in map(VOCAB.get, ps) if i is not None]
    if not tokens:
        return (None, '')
    elif len(tokens) > 510:
        tokens = tokens[:510]
    ps = ''.join(next(k for k, v in VOCAB.items() if i == v) for i in tokens)
    out = forward(tokens, speed)
    if reduce_noise > 0:
        out = nr.reduce_noise(y=out, sr=SAMPLE_RATE, prop_decrease=reduce_noise, n_fft=512)
    opening_cut = max(0, int(opening_cut / speed))
    if opening_cut > 0:
        out[:opening_cut] = 0
    closing_cut = max(0, int(closing_cut / speed))
    if closing_cut > 0:
        out = out[-closing_cut:] = 0
    ease_in = min(int(ease_in / speed), len(out)//2 - opening_cut)
    for i in range(ease_in):
        out[i+opening_cut] *= s_curve(i / ease_in)
    ease_out = min(int(ease_out / speed), len(out)//2 - closing_cut)
    for i in range(ease_out):
        out[-i-1-closing_cut] *= s_curve(i / ease_out)
    return ((SAMPLE_RATE, out), ps)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='Input Text')
            voice = gr.Dropdown(list(CHOICES.items()), label='Voice')
            with gr.Row():
                random_btn = gr.Button('Random Text', variant='secondary')
                generate_btn = gr.Button('Generate', variant='primary')
            random_btn.click(get_random_text, inputs=[voice], outputs=[text])
            with gr.Accordion('Input Phonemes', open=False):
                in_ps = gr.Textbox(show_label=False, info='Override the input text with custom pronunciation. Leave this blank to use the input text instead.')
                with gr.Row():
                    clear_btn = gr.ClearButton(in_ps)
                    phonemize_btn = gr.Button('Phonemize Input Text', variant='primary')
            phonemize_btn.click(phonemize, inputs=[text, voice], outputs=[in_ps])
        with gr.Column():
            audio = gr.Audio(interactive=False, label='Output Audio')
            with gr.Accordion('Tokens', open=True):
                out_ps = gr.Textbox(interactive=False, show_label=False, info='Tokens used to generate the audio. Same as input phonemes if supplied, excluding unknown characters and truncated to 510 tokens.')
    with gr.Accordion('Advanced Settings', open=False):
        with gr.Row():
            reduce_noise = gr.Slider(minimum=0, maximum=1, value=0.5, label='Reduce Noise', info='👻 Fix it in post: non-stationary noise reduction via spectral gating.')
        with gr.Row():
            speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label='Speed', info='⚡️ Adjust the speed of the audio. The trim settings below are also auto-scaled by speed.')
        with gr.Row():
            with gr.Column():
                opening_cut = gr.Slider(minimum=0, maximum=24000, value=5000, step=1000, label='Opening Cut', info='✂️ Zero out this many samples at the start.')
            with gr.Column():
                closing_cut = gr.Slider(minimum=0, maximum=24000, value=0, step=1000, label='Closing Cut', info='✂️ Zero out this many samples at the end.')
        with gr.Row():
            with gr.Column():
                ease_in = gr.Slider(minimum=0, maximum=24000, value=3000, step=1000, label='Ease In', info='🚀 Ease in for this many samples, after opening cut.')
            with gr.Column():
                ease_out = gr.Slider(minimum=0, maximum=24000, value=0, step=1000, label='Ease Out', info='📐 Ease out for this many samples, before closing cut.')
    generate_btn.click(forward, inputs=[text, voice, in_ps, speed, reduce_noise, opening_cut, closing_cut, ease_in, ease_out], outputs=[audio, out_ps])

if __name__ == '__main__':
    demo.launch()
