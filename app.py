from huggingface_hub import snapshot_download
from katsu import Katsu
from models import build_model
import gradio as gr
import noisereduce as nr
import numpy as np
import os
import phonemizer
import pypdf
import random
import re
import spaces
import torch
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'

snapshot = snapshot_download(repo_id='hexgrad/kokoro', allow_patterns=['*.pt', '*.pth', '*.yml'], use_auth_token=os.environ['TOKEN'])
config = yaml.safe_load(open(os.path.join(snapshot, 'config.yml')))
model = build_model(config['model_params'])
for key, value in model.items():
    for module in value.children():
        if isinstance(module, torch.nn.RNNBase):
            module.flatten_parameters()

_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]
for key, state_dict in torch.load(os.path.join(snapshot, 'net.pth'), map_location='cpu', weights_only=True)['net'].items():
    assert key in model, key
    try:
        model[key].load_state_dict(state_dict)
    except:
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model[key].load_state_dict(state_dict, strict=False)

PARAM_COUNT = sum(p.numel() for value in model.values() for p in value.parameters())
print('PARAM_COUNT', PARAM_COUNT)
assert PARAM_COUNT < 82_000_000, PARAM_COUNT

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
    text = re.sub(r'\bD[Rr]\.(?= [A-Z])', 'Doctor', text)
    text = re.sub(r'\b(?:Mr\.|MR\.(?= [A-Z]))', 'Mister', text)
    text = re.sub(r'\b(?:Ms\.|MS\.(?= [A-Z]))', 'Miss', text)
    text = re.sub(r'\b(?:Mrs\.|MRS\.(?= [A-Z]))', 'Mrs', text)
    text = re.sub(r'\betc\.(?! [A-Z])', 'etc', text)
    text = re.sub(r'\b([Yy])eah\b', r"\1e'a", text)
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    text = re.sub(r'[^\S \n]', ' ', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'(?<=\n) +(?=\n)', '', text)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    return parens_to_angles(text).strip()

phonemizers = dict(
    a=phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True),
    b=phonemizer.backend.EspeakBackend(language='en-gb', preserve_punctuation=True, with_stress=True),
    j=Katsu(),
)

def phonemize(text, voice, norm=True):
    lang = voice[0]
    if norm:
        text = normalize(text)
    ps = phonemizers[lang].phonemize([text])
    ps = ps[0] if ps else ''
    # TODO: Custom phonemization rules?
    ps = parens_to_angles(ps)
    # https://en.wiktionary.org/wiki/kokoro#English
    ps = ps.replace('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ').replace('kəkˈɔːɹəʊ', 'kˈəʊkəɹəʊ')
    ps = ''.join(filter(lambda p: p in VOCAB, ps))
    if lang == 'j' and any(p in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for p in ps):
        gr.Warning('Japanese tokenizer does not handle English letters.')
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

def tokenize(ps):
    return [i for i in map(VOCAB.get, ps) if i is not None]

CHOICES = {
    '🇺🇸 🚺 American Female 0': 'af_0',
    '🇺🇸 🚺 Bella': 'af_bella',
    '🇺🇸 🚺 Nicole': 'af_nicole',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 British Female 0': 'bf_0',
    '🇬🇧 🚺 British Female 1': 'bf_1',
    '🇬🇧 🚺 British Female 2': 'bf_2',
    '🇬🇧 🚺 British Female 3': 'bf_3',
    '🇬🇧 🚹 British Male 0': 'bm_0',
    '🇬🇧 🚹 British Male 1': 'bm_1',
    '🇬🇧 🚹 British Male 2': 'bm_2',
    '🇯🇵 🚺 Japanese Female 0': 'jf_0',
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

@spaces.GPU(duration=5)
@torch.no_grad()
def forward(tokens, voice, speed):
    ref_s = VOICES[voice]
    tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
    input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
    text_mask = length_to_mask(input_lengths).to(device)
    bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long()
    pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item())
    c_frame = 0
    for i in range(pred_aln_trg.size(0)):
        pred_aln_trg[i, c_frame:c_frame + pred_dur[0,i].item()] = 1
        c_frame += pred_dur[0,i].item()
    en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
    F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(tokens, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
    return model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu().numpy()

def generate(text, voice, ps=None, speed=1.0, reduce_noise=0.5, opening_cut=4000, closing_cut=2000, ease_in=3000, ease_out=1000, pad_before=5000, pad_after=5000):
    ps = ps or phonemize(text, voice)
    tokens = tokenize(ps)
    if not tokens:
        return (None, '')
    elif len(tokens) > 510:
        tokens = tokens[:510]
    ps = ''.join(next(k for k, v in VOCAB.items() if i == v) for i in tokens)
    try:
        out = forward(tokens, voice, speed)
    except gr.exceptions.Error as e:
        raise gr.Error(e)
        return (None, '')
    if reduce_noise > 0:
        out = nr.reduce_noise(y=out, sr=SAMPLE_RATE, prop_decrease=reduce_noise, n_fft=512)
    opening_cut = int(opening_cut / speed)
    if opening_cut > 0:
        out = out[opening_cut:]
    closing_cut = int(closing_cut / speed)
    if closing_cut > 0:
        out = out[:-closing_cut]
    ease_in = min(int(ease_in / speed), len(out)//2)
    for i in range(ease_in):
        out[i] *= s_curve(i / ease_in)
    ease_out = min(int(ease_out / speed), len(out)//2)
    for i in range(ease_out):
        out[-i-1] *= s_curve(i / ease_out)
    pad_before = int(pad_before / speed)
    if pad_before > 0:
        out = np.concatenate([np.zeros(pad_before), out])
    pad_after = int(pad_after / speed)
    if pad_after > 0:
        out = np.concatenate([out, np.zeros(pad_after)])
    return ((SAMPLE_RATE, out), ps)

with gr.Blocks() as basic_tts:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='Input Text')
            voice = gr.Dropdown(list(CHOICES.items()), label='Voice')
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
            audio = gr.Audio(interactive=False, label='Output Audio')
            with gr.Accordion('Output Tokens', open=True):
                out_ps = gr.Textbox(interactive=False, show_label=False, info='Tokens used to generate the audio, up to 510 allowed. Same as input tokens if supplied, excluding unknowns.')
    with gr.Accordion('Audio Settings', open=False):
        with gr.Row():
            reduce_noise = gr.Slider(minimum=0, maximum=1, value=0.5, label='Reduce Noise', info='👻 Fix it in post: non-stationary noise reduction via spectral gating.')
        with gr.Row():
            speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label='Speed', info='⚡️ Adjust the speed of the audio. The settings below are auto-scaled by speed.')
        with gr.Row():
            with gr.Column():
                opening_cut = gr.Slider(minimum=0, maximum=24000, value=4000, step=1000, label='Opening Cut', info='✂️ Cut this many samples from the start.')
            with gr.Column():
                closing_cut = gr.Slider(minimum=0, maximum=24000, value=2000, step=1000, label='Closing Cut', info='✂️ Cut this many samples from the end.')
        with gr.Row():
            with gr.Column():
                ease_in = gr.Slider(minimum=0, maximum=24000, value=3000, step=1000, label='Ease In', info='🚀 Ease in for this many samples, after opening cut.')
            with gr.Column():
                ease_out = gr.Slider(minimum=0, maximum=24000, value=1000, step=1000, label='Ease Out', info='📐 Ease out for this many samples, before closing cut.')
        with gr.Row():
            with gr.Column():
                pad_before = gr.Slider(minimum=0, maximum=24000, value=5000, step=1000, label='Pad Before', info='🔇 How many samples of silence to insert before the start.')
            with gr.Column():
                pad_after = gr.Slider(minimum=0, maximum=24000, value=5000, step=1000, label='Pad After', info='🔇 How many samples of silence to append after the end.')
    generate_btn.click(generate, inputs=[text, voice, in_ps, speed, reduce_noise, opening_cut, closing_cut, ease_in, ease_out, pad_before, pad_after], outputs=[audio, out_ps])

@spaces.GPU
@torch.no_grad()
def lf_forward(token_lists, voice, speed):
    ref_s = VOICES[voice]
    s = ref_s[:, 128:]
    outs = []
    for tokens in token_lists:
        tokens = torch.LongTensor([[0, *tokens, 0]]).to(device)
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long()
        pred_aln_trg = torch.zeros(input_lengths, pred_dur.sum().item())
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + pred_dur[0,i].item()] = 1
            c_frame += pred_dur[0,i].item()
        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        outs.append(model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu().numpy())
    return outs

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
    if skip_square_brackets:
        text = re.sub(r'\[.*?\]', '', text)
    texts = [t.strip() for t in re.split('\n{'+str(newline_split)+',}', normalize(text))] if newline_split > 0 else [normalize(text)]
    segments = [row for t in texts for row in recursive_split(t, voice)]
    return [(i, *row) for i, row in enumerate(segments)]

def lf_generate(segments, voice, speed=1.0, reduce_noise=0.5, opening_cut=4000, closing_cut=2000, ease_in=3000, ease_out=1000, pad_before=5000, pad_after=5000, pad_between=10000):
    token_lists = list(map(tokenize, segments['Tokens']))
    wavs = []
    opening_cut = int(opening_cut / speed)
    closing_cut = int(closing_cut / speed)
    pad_between = int(pad_between / speed)
    batch_size = 100
    for i in range(0, len(token_lists), batch_size):
        try:
            outs = lf_forward(token_lists[i:i+batch_size], voice, speed)
        except gr.exceptions.Error as e:
            if wavs:
                gr.Warning(str(e))
            else:
                raise gr.Error(e)
            break
        for out in outs:
            if reduce_noise > 0:
                out = nr.reduce_noise(y=out, sr=SAMPLE_RATE, prop_decrease=reduce_noise, n_fft=512)
            if opening_cut > 0:
                out = out[opening_cut:]
            if closing_cut > 0:
                out = out[:-closing_cut]
            ease_in = min(int(ease_in / speed), len(out)//2)
            for i in range(ease_in):
                out[i] *= s_curve(i / ease_in)
            ease_out = min(int(ease_out / speed), len(out)//2)
            for i in range(ease_out):
                out[-i-1] *= s_curve(i / ease_out)
            if wavs and pad_between > 0:
                wavs.append(np.zeros(pad_between))
            wavs.append(out)
    pad_before = int(pad_before / speed)
    if pad_before > 0:
        wavs.insert(0, np.zeros(pad_before))
    pad_after = int(pad_after / speed)
    if pad_after > 0:
        wavs.append(np.zeros(pad_after))
    return (SAMPLE_RATE, np.concatenate(wavs)) if wavs else None

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
            file_input = gr.File(file_types=['.pdf', '.txt'], label='Input File: pdf or txt')
            text = gr.Textbox(label='Input Text')
            file_input.upload(fn=extract_text, inputs=[file_input], outputs=[text])
            voice = gr.Dropdown(list(CHOICES.items()), label='Voice')
            with gr.Accordion('Text Settings', open=False):
                skip_square_brackets = gr.Checkbox(True, label='Skip [Square Brackets]', info='Recommended for academic papers, Wikipedia articles, or texts with citations.')
                newline_split = gr.Number(2, label='Newline Split', info='Split the input text on this many newlines. Affects how the text is segmented.', precision=0, minimum=0)
            with gr.Row():
                segment_btn = gr.Button('Tokenize', variant='primary')
                generate_btn = gr.Button('Generate x0', variant='secondary', interactive=False)
        with gr.Column():
            audio = gr.Audio(interactive=False, label='Output Audio')
            with gr.Accordion('Audio Settings', open=False):
                with gr.Row():
                    reduce_noise = gr.Slider(minimum=0, maximum=1, value=0.5, label='Reduce Noise', info='👻 Fix it in post: non-stationary noise reduction via spectral gating.')
                with gr.Row():
                    speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label='Speed', info='⚡️ Adjust the speed of the audio. The settings below are auto-scaled by speed.')
                with gr.Row():
                    with gr.Column():
                        opening_cut = gr.Slider(minimum=0, maximum=24000, value=4000, step=1000, label='Opening Cut', info='✂️ Cut this many samples from the start.')
                    with gr.Column():
                        closing_cut = gr.Slider(minimum=0, maximum=24000, value=2000, step=1000, label='Closing Cut', info='✂️ Cut this many samples from the end.')
                with gr.Row():
                    with gr.Column():
                        ease_in = gr.Slider(minimum=0, maximum=24000, value=3000, step=1000, label='Ease In', info='🚀 Ease in for this many samples, after opening cut.')
                    with gr.Column():
                        ease_out = gr.Slider(minimum=0, maximum=24000, value=1000, step=1000, label='Ease Out', info='📐 Ease out for this many samples, before closing cut.')
                with gr.Row():
                    with gr.Column():
                        pad_before = gr.Slider(minimum=0, maximum=24000, value=5000, step=1000, label='Pad Before', info='🔇 How many samples of silence to insert before the start.')
                    with gr.Column():
                        pad_after = gr.Slider(minimum=0, maximum=24000, value=5000, step=1000, label='Pad After', info='🔇 How many samples of silence to append after the end.')
                with gr.Row():
                    pad_between = gr.Slider(minimum=0, maximum=24000, value=10000, step=1000, label='Pad Between', info='🔇 How many samples of silence to insert between segments.')
    with gr.Row():
        segments = gr.Dataframe(headers=['#', 'Text', 'Tokens', 'Length'], row_count=(1, 'dynamic'), col_count=(4, 'fixed'), label='Segments', interactive=False, wrap=True)
        segments.change(fn=did_change_segments, inputs=[segments], outputs=[segment_btn, generate_btn])
    segment_btn.click(segment_and_tokenize, inputs=[text, voice, skip_square_brackets, newline_split], outputs=[segments])
    generate_btn.click(lf_generate, inputs=[segments, voice, speed, reduce_noise, opening_cut, closing_cut, ease_in, ease_out, pad_before, pad_after, pad_between], outputs=[audio])

with gr.Blocks() as app:
    gr.TabbedInterface(
        [basic_tts, lf_tts],
        ['Basic TTS', 'Long-Form'],
    )

if __name__ == '__main__':
    app.queue(api_open=True).launch()
