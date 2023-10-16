import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pythainlp import word_tokenize
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
from pythainlp.corpus import thai_stopwords
import re

custom_dict_thai_word = {
    "อัลตร้า",
    "11PM",
    "720p",
    "คริสตัส",
    "เน็ตฟลิก",
    "p5",
    "ps5",
    "4k",
    "แอนดรอย",
    "ทุกรุ่น",
    "เอเบิลเม็น",
    "ช้ากว่า",
    "ไทเทเนี่ยม",
    "เอไอ",
    "จุงเบย",
    "แหล่ะ",
    "ps4",
    "iphone",
    "A17",
    "ไทเท",
    "s23",
    "แอป",
    "i15",
    "i14",
    "i13",
    "i12",
    "i11",
    "เเมร่ง",
    "ยูทูป",
    "5G",
    "ดรอป",
    "เเละ",
    "พับจี",
    "เเพง",
    "เเยก",
    "เเทบ",
    "ชาจ",
    "ทวิต",
    "ไอจี",
    "แล้ว",
    "เคส",
    "ตาลุกวาว",
    "IP15",
    "IP14",
    "ip15",
    "ip14",
    "เเรง",
    "บอก",
    "ตู้ม",
    "อุส่าห์",
    "ก่อน",
    "ภาพยนต์",
    "แลค",
    "วอท",
    "แม่ง",
    "เกนชิน",
    "แปป",
    "เครม",
    "ปัจใจ",
    "ชิฟ",
    "บลา",
    "อุ่นภูมิ",
    "ไลน์นิ่ง",
    "สแปค",
    "คอเกม",
    "แอนดอย",
    "อินฟลู",
    "โซเชียล",
    "จาก",
    "เกมเมอร์",
    "สน้บสนุน",
    "อีสปอร์ต",
    "ไวเลส",
    "ม้าก",
    "เปลี้ยน",
    "แอนดอย",
    "เกมมิง",
    "มาก",
    "คอนทร่า",
    "เมม",
    "โอ่ว",
    "ออนดรอย",
    "โอเอส",
    "ก่อน",
    "แอนดอรย์",
    "พอซ",
    "เกมมิ่ง",
    "โอเคร้",
    "ไอแลนด์",
    "ลี่นปื้ด",
    "15",
    "14",
    "แม้ก",
    "เหน",
    "โครต",
    "ราคา",
    "คมระเคือง",
    "ภาพ",
    "ไม่ได้",
    "ไม่เคย",
    "ไม่ใช่",
    "ไม่ถูก",
    "ไม่มีปัญหา",
    "ไม่ร้อน",
}

thai_word_set = set(thai_words())
custom_dict_thai_word.update(thai_word_set)
trie = dict_trie(dict_source=custom_dict_thai_word)

STOP_WORD = list(thai_stopwords()) + [" ", "\n", "ๆ"]
STOP_WORD.remove("ไม่")
FORMAT = r"[\u0E00-\u0E7Fa-zA-Z'0-9]+"


def tokenize(sentence):
    return word_tokenize(
        sentence, engine="newmm", keep_whitespace=False, custom_dict=trie
    )


def cleaning_stop_word(tk_list):
    return [word.replace("ๆ", "") for word in tk_list if word not in STOP_WORD]


def cleaning_symbols_emoji(tk_list):
    return [re.findall(FORMAT, text)[0] for text in tk_list if re.findall(FORMAT, text)]


def big_cleaning(sentence):
    tokens = cleaning_symbols_emoji(cleaning_stop_word(tokenize(sentence)))
    return tokens


df_0 = pd.read_csv("./data/class_0.csv")
df_1 = pd.read_csv("./data/class_1.csv")
frame = [df_0, df_1]
data = pd.concat(frame, ignore_index=True)

x = data["text"].apply(big_cleaning).astype(str)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x)


def predict_class(model, dataframeToPredict, threshold=0.6):
    dataframeToPredict = dataframeToPredict["message"].apply(big_cleaning).astype(str)
    dataframeToPredict = tokenizer.texts_to_sequences(dataframeToPredict)
    padded_sequence = pad_sequences(dataframeToPredict, maxlen=x.shape[1])

    lstm_pred = model.predict(padded_sequence)

    lstm_label = np.argmax(lstm_pred, axis=1)

    lstm_label[lstm_pred.max(axis=1) < threshold] = -1

    check = np.where(
        lstm_label == 0,
        "Positive 😁",
        np.where(lstm_label == 1, "Negative 😡", "Neutral"),
    )
    return check
