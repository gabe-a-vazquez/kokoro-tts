# https://github.com/polm/cutlet/blob/master/cutlet/cutlet.py
from dataclasses import dataclass
from fugashi import Tagger
from num2kana import Convert
import mojimoji
import re
import unicodedata

HEPBURN = {
chr(12449):'a',   #ァ
chr(12450):'a',   #ア
chr(12451):'i',   #ィ
chr(12452):'i',   #イ
chr(12453):'ɯ',   #ゥ
chr(12454):'ɯ',   #ウ
chr(12455):'e',   #ェ
chr(12456):'e',   #エ
chr(12457):'o',   #ォ
chr(12458):'o',   #オ
chr(12459):'ka',  #カ
chr(12460):'ɡa',  #ガ
chr(12461):'ki',  #キ
chr(12462):'ɡi',  #ギ
chr(12463):'kɯ',  #ク
chr(12464):'ɡɯ',  #グ
chr(12465):'ke',  #ケ
chr(12466):'ɡe',  #ゲ
chr(12467):'ko',  #コ
chr(12468):'ɡo',  #ゴ
chr(12469):'sa',  #サ
chr(12470):'za',  #ザ
chr(12471):'ɕi',  #シ
chr(12472):'dʑi', #ジ
chr(12473):'sɨ',  #ス
chr(12474):'zɨ',  #ズ
chr(12475):'se',  #セ
chr(12476):'ze',  #ゼ
chr(12477):'so',  #ソ
chr(12478):'zo',  #ゾ
chr(12479):'ta',  #タ
chr(12480):'da',  #ダ
chr(12481):'tɕi', #チ
chr(12482):'dʑi', #ヂ
# chr(12483)        #ッ
chr(12484):'tsɨ', #ツ
chr(12485):'zɨ',  #ヅ
chr(12486):'te',  #テ
chr(12487):'de',  #デ
chr(12488):'to',  #ト
chr(12489):'do',  #ド
chr(12490):'na',  #ナ
chr(12491):'ɲi',  #ニ
chr(12492):'nɯ',  #ヌ
chr(12493):'ne',  #ネ
chr(12494):'no',  #ノ
chr(12495):'ha',  #ハ
chr(12496):'ba',  #バ
chr(12497):'pa',  #パ
chr(12498):'çi',  #ヒ
chr(12499):'bi',  #ビ
chr(12500):'pi',  #ピ
chr(12501):'ɸɯ',  #フ
chr(12502):'bɯ',  #ブ
chr(12503):'pɯ',  #プ
chr(12504):'he',  #ヘ
chr(12505):'be',  #ベ
chr(12506):'pe',  #ペ
chr(12507):'ho',  #ホ
chr(12508):'bo',  #ボ
chr(12509):'po',  #ポ
chr(12510):'ma',  #マ
chr(12511):'mi',  #ミ
chr(12512):'mɯ',  #ム
chr(12513):'me',  #メ
chr(12514):'mo',  #モ
chr(12515):'ja',  #ャ
chr(12516):'ja',  #ヤ
chr(12517):'jɯ',  #ュ
chr(12518):'jɯ',  #ユ
chr(12519):'jo',  #ョ
chr(12520):'jo',  #ヨ
chr(12521):'ra',  #ラ
chr(12522):'ri',  #リ
chr(12523):'rɯ',  #ル
chr(12524):'re',  #レ
chr(12525):'ro',  #ロ
chr(12526):'wa',  #ヮ
chr(12527):'wa',  #ワ
chr(12528):'i',   #ヰ
chr(12529):'e',   #ヱ
chr(12530):'o',   #ヲ
# chr(12531)        #ン
chr(12532):'vɯ',  #ヴ
chr(12533):'ka',  #ヵ
chr(12534):'ke',  #ヶ
}
assert len(HEPBURN) == 84 and all(i in {12483, 12531} or chr(i) in HEPBURN for i in range(12449, 12535))

for k, v in list(HEPBURN.items()):
    HEPBURN[chr(ord(k)-96)] = v
assert len(HEPBURN) == 84*2

HEPBURN.update({
chr(12535):'va',  #ヷ
chr(12536):'vi',  #ヸ
chr(12537):'ve',  #ヹ
chr(12538):'vo',  #ヺ
})
assert len(HEPBURN) == 84*2+4 and all(chr(i) in HEPBURN for i in range(12535, 12539))

HEPBURN.update({
chr(12784):'kɯ',  #ㇰ
chr(12785):'ɕi',  #ㇱ
chr(12786):'sɨ',  #ㇲ
chr(12787):'to',  #ㇳ
chr(12788):'nɯ',  #ㇴ
chr(12789):'ha',  #ㇵ
chr(12790):'çi',  #ㇶ
chr(12791):'ɸɯ',  #ㇷ
chr(12792):'he',  #ㇸ
chr(12793):'ho',  #ㇹ
chr(12794):'mɯ',  #ㇺ
chr(12795):'ra',  #ㇻ
chr(12796):'ri',  #ㇼ
chr(12797):'rɯ',  #ㇽ
chr(12798):'re',  #ㇾ
chr(12799):'ro',  #ㇿ
})
assert len(HEPBURN) == 84*2+4+16 and all(chr(i) in HEPBURN for i in range(12784, 12800))

HEPBURN.update({
chr(12452)+chr(12455):'je',  #イェ
chr(12454)+chr(12451):'wi',  #ウィ
chr(12454)+chr(12455):'we',  #ウェ
chr(12454)+chr(12457):'wo',  #ウォ
chr(12461)+chr(12455):'kʲe', #キェ
chr(12461)+chr(12515):'kʲa', #キャ
chr(12461)+chr(12517):'kʲɨ', #キュ
chr(12461)+chr(12519):'kʲo', #キョ
chr(12462)+chr(12515):'ɡʲa', #ギャ
chr(12462)+chr(12517):'ɡʲɨ', #ギュ
chr(12462)+chr(12519):'ɡʲo', #ギョ
chr(12463)+chr(12449):'kʷa', #クァ
chr(12463)+chr(12451):'kʷi', #クィ
chr(12463)+chr(12455):'kʷe', #クェ
chr(12463)+chr(12457):'kʷo', #クォ
chr(12464)+chr(12449):'ɡʷa', #グァ
chr(12464)+chr(12451):'ɡʷi', #グィ
chr(12464)+chr(12455):'ɡʷe', #グェ
chr(12464)+chr(12457):'ɡʷo', #グォ
chr(12471)+chr(12455):'ɕe',  #シェ
chr(12471)+chr(12515):'ɕa',  #シャ
chr(12471)+chr(12517):'ɕɨ',  #シュ
chr(12471)+chr(12519):'ɕo',  #ショ
chr(12472)+chr(12455):'dʑe', #ジェ
chr(12472)+chr(12515):'dʑa', #ジャ
chr(12472)+chr(12517):'dʑɨ', #ジュ
chr(12472)+chr(12519):'dʑo', #ジョ
chr(12481)+chr(12455):'tɕe', #チェ
chr(12481)+chr(12515):'tɕa', #チャ
chr(12481)+chr(12517):'tɕɨ', #チュ
chr(12481)+chr(12519):'tɕo', #チョ
chr(12482)+chr(12515):'dʑa', #ヂャ
chr(12482)+chr(12517):'dʑɨ', #ヂュ
chr(12482)+chr(12519):'dʑo', #ヂョ
chr(12484)+chr(12449):'tsa', #ツァ
chr(12484)+chr(12451):'tsi', #ツィ
chr(12484)+chr(12455):'tse', #ツェ
chr(12484)+chr(12457):'tso', #ツォ
chr(12486)+chr(12451):'ti',  #ティ
chr(12486)+chr(12517):'tʲɨ', #テュ
chr(12487)+chr(12451):'di',  #ディ
chr(12487)+chr(12517):'dʲɨ', #デュ
chr(12488)+chr(12453):'tɯ',  #トゥ
chr(12489)+chr(12453):'dɯ',  #ドゥ
chr(12491)+chr(12455):'ɲe',  #ニェ
chr(12491)+chr(12515):'ɲa',  #ニャ
chr(12491)+chr(12517):'ɲɨ',  #ニュ
chr(12491)+chr(12519):'ɲo',  #ニョ
chr(12498)+chr(12455):'çe',  #ヒェ
chr(12498)+chr(12515):'ça',  #ヒャ
chr(12498)+chr(12517):'çɨ',  #ヒュ
chr(12498)+chr(12519):'ço',  #ヒョ
chr(12499)+chr(12515):'bʲa', #ビャ
chr(12499)+chr(12517):'bʲɨ', #ビュ
chr(12499)+chr(12519):'bʲo', #ビョ
chr(12500)+chr(12515):'pʲa', #ピャ
chr(12500)+chr(12517):'pʲɨ', #ピュ
chr(12500)+chr(12519):'pʲo', #ピョ
chr(12501)+chr(12449):'ɸa',  #ファ
chr(12501)+chr(12451):'ɸi',  #フィ
chr(12501)+chr(12455):'ɸe',  #フェ
chr(12501)+chr(12457):'ɸo',  #フォ
chr(12501)+chr(12517):'ɸʲɨ', #フュ
chr(12501)+chr(12519):'ɸʲo', #フョ
chr(12511)+chr(12515):'mʲa', #ミャ
chr(12511)+chr(12517):'mʲɨ', #ミュ
chr(12511)+chr(12519):'mʲo', #ミョ
chr(12522)+chr(12515):'rʲa', #リャ
chr(12522)+chr(12517):'rʲɨ', #リュ
chr(12522)+chr(12519):'rʲo', #リョ
chr(12532)+chr(12449):'va',  #ヴァ
chr(12532)+chr(12451):'vi',  #ヴィ
chr(12532)+chr(12455):'ve',  #ヴェ
chr(12532)+chr(12457):'vo',  #ヴォ
chr(12532)+chr(12517):'vʲɨ', #ヴュ
chr(12532)+chr(12519):'vʲo', #ヴョ
})
assert len(HEPBURN) == 84*2+4+16+76

for k, v in list(HEPBURN.items()):
    if len(k) != 2:
        continue
    a, b = k
    assert a in HEPBURN and b in HEPBURN, (a, b)
    a = chr(ord(a)-96)
    b = chr(ord(b)-96)
    assert a in HEPBURN and b in HEPBURN, (a, b)
    HEPBURN[a+b] = v
assert len(HEPBURN) == 84*2+4+16+76*2

HEPBURN.update({
# symbols
# 'ー': '-', # 長音符, only used when repeated
'。': '.',
'、': ',',
'？': '?',
'！': '!',
'「': '"',
'」': '"',
'『': '"',
'』': '"',
'：': ':',
'；': ';',
'（': '(',
'）': ')',
'《': '(',
'》': ')',
'【': '[',
'】': ']',
'・': ' ',#'/',
'，': ',',
'～': '—',
'〜': '—',
'—': '—',
'«': '«',
'»': '»',

# other
'゚': '', # combining handakuten by itself, just discard
'゙': '', # combining dakuten by itself
})

def add_dakuten(kk):
    """Given a kana (single-character string), add a dakuten."""
    try:
        # ii = 'かきくけこさしすせそたちつてとはひふへほ'.index(kk)
        ii = 'カキクケコサシスセソタチツテトハヒフヘホ'.index(kk)
        return 'ガギグゲゴザジズゼゾダヂヅデドバビブベボ'[ii]
        # return 'がぎぐげござじずぜぞだぢづでどばびぶべぼ'[ii]
    except ValueError:
        # this is normal if the input is nonsense
        return None

SUTEGANA = 'ャュョァィゥェォ' #'ゃゅょぁぃぅぇぉ'
PUNCT = '\'".!?(),;:-'
ODORI = '々〃ゝゞヽゞ'

@dataclass
class Token:
    surface: str
    space: bool # if a space should follow
    def __str__(self):
        sp = " " if self.space else ""
        return f"{self.surface}{sp}"

class Katsu:
    def __init__(self):
        """Create a Katsu object, which holds configuration as well as
        tokenizer state.

        Typical usage:

        ```python
        katsu = Katsu()
        roma = katsu.romaji("カツカレーを食べた")
        # "Cutlet curry wo tabeta"
        ```
        """
        self.tagger = Tagger()
        self.table = dict(HEPBURN) # make a copy so we can modify it
        self.exceptions = {}

    def romaji(self, text):
        """Build a complete string from input text."""
        if not text:
            return ''
        text = self._normalize_text(text)
        words = self.tagger(text)
        tokens = self._romaji_tokens(words)
        out = ''.join([str(tok) for tok in tokens])
        return re.sub(r'\s+', ' ', out.strip())

    def phonemize(self, texts):
        # espeak-ng API
        return [self.romaji(text) for text in texts]

    def _normalize_text(self, text):
        """Given text, normalize variations in Japanese.

        This specifically removes variations that are meaningless for romaji
        conversion using the following steps:

        - Unicode NFKC normalization
        - Full-width Latin to half-width
        - Half-width katakana to full-width
        """
        # perform unicode normalization
        text = re.sub(r'[〜～](?=\d)', 'から', text) # wave dash range
        text = unicodedata.normalize('NFKC', text)
        # convert all full-width alphanum to half-width, since it can go out as-is
        text = mojimoji.zen_to_han(text, kana=False)
        # replace half-width katakana with full-width
        text = mojimoji.han_to_zen(text, digit=False, ascii=False)
        return ''.join([(' '+Convert(t)) if t.isdigit() else t for t in re.findall(r'\d+|\D+', text)])

    def _romaji_tokens(self, words):
        """Build a list of tokens from input nodes."""
        out = []
        for wi, word in enumerate(words):
            po = out[-1] if out else None
            pw = words[wi - 1] if wi > 0 else None
            nw = words[wi + 1] if wi < len(words) - 1 else None
            roma = self._romaji_word(word)
            tok = Token(roma, False)
            # handle punctuation with atypical spacing
            surface = word.surface#['orig']
            if surface in '「『' or roma in '([':
                if po:
                    po.space = True
            elif surface in '」』' or roma in ']).,?!:':
                if po:
                    po.space = False
                tok.space = True
            elif roma == ' ':
                tok.space = False
            else:
                tok.space = True
            out.append(tok)
        # remove any leftover sokuon
        for tok in out:
            tok.surface = tok.surface.replace(chr(12483), '')
        return out

    def _romaji_word(self, word):
        """Return the romaji for a single word (node)."""
        surface = word.surface#['orig']
        if surface in self.exceptions:
            return self.exceptions[surface]
        assert not surface.isdigit(), surface
        if surface.isascii():
            return surface
        kana = word.feature.pron or word.feature.kana or surface
        if word.is_unk:
            if word.char_type == 7: # katakana
                pass
            elif word.char_type == 3: # symbol
                return ''.join(map(lambda c: self.table.get(c, c), surface))
            else:
                return '' # TODO: silently fail
        out = ''
        for ki, char in enumerate(kana):
            nk = kana[ki + 1] if ki < len(kana) - 1 else None
            pk = kana[ki - 1] if ki > 0 else None
            out += self._get_single_mapping(pk, char, nk)
        return out

    def _get_single_mapping(self, pk, kk, nk):
        """Given a single kana and its neighbors, return the mapped romaji."""
        # handle odoriji
        # NOTE: This is very rarely useful at present because odoriji are not
        # left in readings for dictionary words, and we can't follow kana
        # across word boundaries.
        if kk in ODORI:
            if kk in 'ゝヽ':
                if pk: return pk
                else: return '' # invalid but be nice
            if kk in 'ゞヾ': # repeat with voicing
                if not pk: return ''
                vv = add_dakuten(pk)
                if vv: return self.table[vv]
                else: return ''
            # remaining are 々 for kanji and 〃 for symbols, but we can't
            # infer their span reliably (or handle rendaku)
            return ''
        # handle digraphs
        if pk and (pk + kk) in self.table:
            return self.table[pk + kk]
        if nk and (kk + nk) in self.table:
            return ''
        if nk and nk in SUTEGANA:
            if kk == 'ッ': return '' # never valid, just ignore
            return self.table[kk][:-1] + self.table[nk]
        if kk in SUTEGANA:
            return ''
        if kk == 'ー': # 長音符
            return 'ː'
        if ord(kk) in {12387, 12483}: # っ or ッ
            tnk = self.table.get(nk)
            if tnk and tnk[0] in 'bdɸɡhçijkmnɲoprstɯvwz':
                return tnk[0]
            return kk
        if ord(kk) in {12435, 12531}: # ん or ン
            # https://en.wikipedia.org/wiki/N_(kana)
            # m before m,p,b
            # ŋ before k,g
            # ɲ before ɲ,tɕ,dʑ
            # n before n,t,d,r,z
            # ɴ otherwise
            tnk = self.table.get(nk)
            if tnk:
                if tnk[0] in 'mpb':
                    return 'm'
                elif tnk[0] in 'kɡ':
                    return 'ŋ'
                elif any(tnk.startswith(p) for p in ('ɲ','tɕ','dʑ')):
                    return 'ɲ'
                elif tnk[0] in 'ntdrz':
                    return 'n'
            return 'ɴ'
        return self.table.get(kk, '')
