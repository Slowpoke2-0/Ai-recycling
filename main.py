import os
import discord
import aiohttp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps

TOKEN = "MTM0NjkwMDgwMTcyOTA2OTA4Nw.GPmmhW.g_2ifaFmIy-u-1tghsQ1sz9Bi3l50zN2bgYfr0"
SAVE_DIR = "images"

# --- KONFIGURACJA AI ---
MODEL_PATH = "keras_Model.h5"
model = load_model(MODEL_PATH, compile=False)
LABELS_PATH = "labels.txt"

# przygotowanie modelu
model = load_model(MODEL_PATH, compile=False)
class_names = open(LABELS_PATH, "r").readlines()

os.makedirs(SAVE_DIR, exist_ok=True)

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

def classify_image(image_path):
    """Ładuje obraz i zwraca wynik predykcji"""
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

@client.event
async def on_ready():
    print(f"Zalogowano jako {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    for attachment in message.attachments:
        if attachment.content_type and attachment.content_type.startswith("image/"):
            filename = os.path.join(SAVE_DIR, attachment.filename)

            # pobranie i zapisanie obrazu
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status == 200:
                        with open(filename, "wb") as f:
                            f.write(await resp.read())

                        # klasyfikacja obrazu
                        class_name, confidence = classify_image(filename)

                        await message.reply(
                            f"✅ Obraz zapisany i sklasyfikowany!\n"
                            f"🔎 Klasa: **{class_name}**\n"
                            f"📊 Pewność: {confidence:.2f}"
                        )

client.run(TOKEN)