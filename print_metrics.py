import json
import numpy as np
import sys

metric_path = sys.argv[1]
#metric_path = "./metrics/Other_M-CLIP/LABSE-Vit-L-14-XM3600.json"

model_type = sys.argv[2]

# empty str for most, flores for langs in flores code
# XTD10
dataset_name = sys.argv[3]

with open(metric_path, 'r') as f:
    metrics = json.load(f)

langs = list(metrics.keys())

flores_to_iso2 = {
    "arb_Arab": "ar",  # Arabic
    "ben_Beng": "bn",  # Bengali
    "ces_Latn": "cs",  # Czech
    "deu_Latn": "de",  # German
    "ell_Grek": "el",  # Greek
    "eng_Latn": "en",  # English
    "fra_Latn": "fr",  # French
    "guj_Gujr": "gu",  # Gujarati
    "heb_Hebr": "he",  # Hebrew
    "hin_Deva": "hi",  # Hindi
    "ind_Latn": "id",  # Indonesian
    "ita_Latn": "it",  # Italian
    "jpn_Jpan": "ja",  # Japanese
    "kan_Knda": "kn",  # Kannada
    "kor_Hang": "ko",  # Korean
    "mal_Mlym": "ml",  # Malayalam
    "mar_Deva": "mr",  # Marathi
    "nld_Latn": "nl",  # Dutch
    "npi_Deva": "ne",  # Nepali
    "pan_Guru": "pa",  # Punjabi
    "pes_Arab": "fa",  # Persian
    "pol_Latn": "pl",  # Polish
    "por_Latn": "pt",  # Portuguese
    "ron_Latn": "ro",  # Romanian
    "rus_Cyrl": "ru",  # Russian
    "spa_Latn": "es",  # Spanish
    "tam_Taml": "ta",  # Tamil
    "tel_Telu": "te",  # Telugu
    "tur_Latn": "tr",  # Turkish
    "ukr_Cyrl": "uk",  # Ukrainian
    "urd_Arab": "ur",  # Urdu
    "vie_Latn": "vi", # vietnamese 
    "zho_Hans": "zh s",  # Chinese (Simplified)
    "zho_Hant": "zh t"   # Chinese (Traditional)
}

sentCLIP_lang = [
    "ar", "bg", "ca", "cs", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "fr-ca",
    "gl", "gu", "he", "hi", "hr", "hu", "hy", "id", "it", "ja", "ka", "ko", "ku",
    "lt", "lv", "mk", "mn", "mr", "ms", "my", "nb", "nl", "pl", "pt", "pt-br", "ro",
    "ru", "sk", "sl", "sq", "sr", "sv", "th", "tr", "uk", "ur", "vi", "zh-cn", "zh-tw",
    # added by me
    "zh",
]
print("SentenceTransformer Multilingual CLIP langs", len(sentCLIP_lang[:-1]))

LABSE_langs = [
    ("af", "AFRIKAANS"), ("ht", "HAITIAN_CREOLE"), ("pt", "PORTUGUESE"),
    ("am", "AMHARIC"), ("hu", "HUNGARIAN"), ("ro", "ROMANIAN"),
    ("ar", "ARABIC"), ("hy", "ARMENIAN"), ("ru", "RUSSIAN"),
    ("as", "ASSAMESE"), ("id", "INDONESIAN"), ("rw", "KINYARWANDA"),
    ("az", "AZERBAIJANI"), ("ig", "IGBO"), ("si", "SINHALESE"),
    ("be", "BELARUSIAN"), ("is", "ICELANDIC"), ("sk", "SLOVAK"),
    ("bg", "BULGARIAN"), ("it", "ITALIAN"), ("sl", "SLOVENIAN"),
    ("bn", "BENGALI"), ("ja", "JAPANESE"), ("sm", "SAMOAN"),
    ("bo", "TIBETAN"), ("jv", "JAVANESE"), ("sn", "SHONA"),
    ("bs", "BOSNIAN"), ("ka", "GEORGIAN"), ("so", "SOMALI"),
    ("ca", "CATALAN"), ("kk", "KAZAKH"), ("sq", "ALBANIAN"),
    ("ceb", "CEBUANO"), ("km", "KHMER"), ("sr", "SERBIAN"),
    ("co", "CORSICAN"), ("kn", "KANNADA"), ("st", "SESOTHO"),
    ("cs", "CZECH"), ("ko", "KOREAN"), ("su", "SUNDANESE"),
    ("cy", "WELSH"), ("ku", "KURDISH"), ("sv", "SWEDISH"),
    ("da", "DANISH"), ("ky", "KYRGYZ"), ("sw", "SWAHILI"),
    ("de", "GERMAN"), ("la", "LATIN"), ("ta", "TAMIL"),
    ("el", "GREEK"), ("lb", "LUXEMBOURGISH"), ("te", "TELUGU"),
    ("en", "ENGLISH"), ("lo", "LAOTHIAN"), ("tg", "TAJIK"),
    ("eo", "ESPERANTO"), ("lt", "LITHUANIAN"), ("th", "THAI"),
    ("es", "SPANISH"), ("lv", "LATVIAN"), ("tk", "TURKMEN"),
    ("et", "ESTONIAN"), ("mg", "MALAGASY"), ("tl", "TAGALOG"),
    ("eu", "BASQUE"), ("mi", "MAORI"), ("tr", "TURKISH"),
    ("fa", "PERSIAN"), ("mk", "MACEDONIAN"), ("tt", "TATAR"),
    ("fi", "FINNISH"), ("ml", "MALAYALAM"), ("ug", "UIGHUR"),
    ("fr", "FRENCH"), ("mn", "MONGOLIAN"), ("uk", "UKRAINIAN"),
    ("fy", "FRISIAN"), ("mr", "MARATHI"), ("ur", "URDU"),
    ("ga", "IRISH"), ("ms", "MALAY"), ("uz", "UZBEK"),
    ("gd", "SCOTS_GAELIC"), ("mt", "MALTESE"), ("vi", "VIETNAMESE"),
    ("gl", "GALICIAN"), ("my", "BURMESE"), ("wo", "WOLOF"),
    ("gu", "GUJARATI"), ("ne", "NEPALI"), ("xh", "XHOSA"),
    ("ha", "HAUSA"), ("nl", "DUTCH"), ("yi", "YIDDISH"),
    ("haw", "HAWAIIAN"), ("no", "NORWEGIAN"), ("yo", "YORUBA"),
    ("he", "HEBREW"), ("ny", "NYANJA"), ("zh", "CHINESE"),
    ("hi", "HINDI"), ("or", "ORIYA"), ("zu", "ZULU"),
    ("hmn", "HMONG"), ("pa", "PUNJABI"),
    ("hr", "CROATIAN"), ("pl", "POLISH")
]
LABSE_langs = [x.lower() for x,_ in LABSE_langs]
print("LaBSE langs", len(LABSE_langs))
jina_v3_langs = [
    ("ar", "Arabic"), ("bn", "Bengali"), ("zh", "Chinese"), ("da", "Danish"),
    ("nl", "Dutch"), ("en", "English"), ("fi", "Finnish"), ("fr", "French"),
    ("ka", "Georgian"), ("de", "German"), ("el", "Greek"), ("hi", "Hindi"),
    ("id", "Indonesian"), ("it", "Italian"), ("ja", "Japanese"), ("ko", "Korean"),
    ("lv", "Latvian"), ("no", "Norwegian"), ("pl", "Polish"), ("pt", "Portuguese"),
    ("ro", "Romanian"), ("ru", "Russian"), ("sk", "Slovak"), ("es", "Spanish"),
    ("sv", "Swedish"), ("th", "Thai"), ("tr", "Turkish"), ("uk", "Ukrainian"),
    ("ur", "Urdu"), ("vi", "Vietnamese")
]
jina_v3_langs = [x.lower() for x,_ in jina_v3_langs]
print("Jina v3 langs", len(jina_v3_langs))

multilingualCLIP_langs = [
    "af", "am", "ar", "az", "bg", "bn", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa", "fa-AF", "fi", 
    "fr", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "ka", "kk", "kn", "ko", "lt", 
    "lv", "mk", "ml", "mn", "ms", "mt", "nl", "no", "pl", "ps", "pt", "ro", "ru", "si", "sk", "sl", "so", "sq", 
    "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "uz", "vi", "zh", "zh-tw"
]
print("multilingualCLIP (Carlson et al.) langs", len(multilingualCLIP_langs))



def crossmodal3600_print(all_metrics, supported_langs=None, dataset_name=""):
    
    i2t_r_10 = []
    t2i_r_10 = []

    langs = all_metrics.keys()
    
    if dataset_name == "XTD10":
        langs = [(l.split('_')[-1], l) for l in langs]
    elif dataset_name == "flores":
        langs = [(flores_to_iso2[l], l) for l in langs]
    else:
        langs = [(l, l) for l in langs]

    #langs = sorted(langs, key=lambda x: x[0])
    if supported_langs is not None:
        unsupported_langs = [l for l in langs if l[0].lower().split(' ')[0] not in supported_langs] 
        langs = [l for l in langs if l[0].lower().split(' ')[0] in supported_langs] 
        
        print("Unsupported langs", unsupported_langs)

    for l_code, l_col in langs:
        metrics = all_metrics[l_col]
        i2t_r_10.append(metrics["image_to_text"]["recall@10"])
        t2i_r_10.append(metrics["text_to_image"]["recall@10"])
            
    avg_i2t = np.mean(i2t_r_10).item()
    avg_t2i = np.mean(t2i_r_10).item()

    
    print("Lang order")
    print("**".join(['Avg']+[l_code for l_code,_ in langs]))

    print("T2I")
    t2i = [avg_t2i] + t2i_r_10
    t2i = [f"{x:.1f}" for x in t2i]
    print("**".join(t2i))
    
    print("I2T")
    i2t = [avg_i2t] + i2t_r_10
    i2t = [f"{x:.1f}" for x in i2t]
    print("**".join(i2t))

supp_lang = None
if model_type == "labse":
    supp_lang = LABSE_langs
elif model_type == "jina":
    supp_lang = jina_v3_langs
elif model_type == "m-clip":
    supp_lang = multilingualCLIP_langs
elif model_type == "sentCLIP":
    supp_lang = sentCLIP_lang
else:
    print("No valid supported lang for this model type")

print("Supported Langs")
crossmodal3600_print(metrics, supp_lang, dataset_name=dataset_name)
print("="*100)
print("All langs")
crossmodal3600_print(metrics, dataset_name=dataset_name)
