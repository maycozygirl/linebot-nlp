
from flask import Flask,request,abort
import requests
from app.Config import *
from keras.models import load_model
from pythainlp import word_tokenize
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
from pythainlp.corpus import thai_stopwords
import pandas as pd
import json
from  keras.preprocessing.text import  Tokenizer
from keras.utils import  pad_sequences
import numpy as np




app=Flask(__name__)

@app.route('/webhook',methods=['POST','GET'])


def webhook():

    words = ['กาแฟ', 'น้ำหวาน', 'เค้ก', 'เครป', 'ไอศครีม', 'เครป', 'ผลไม้', 'น้ำผลไม้', 'โรตี', 'ขนมไทย', 'น้ำแข็งไส', 'ขนมปัง', 'แซนวิช', 'อเมริกาโน่', 'แซนวิช', 'น้ำสมุนไพร', 'เตยแก้ว', 'บอก', 'โอริโอ้']
    custom_words_list = set(thai_words()).union(set(words))
    trie = dict_trie(dict_source=custom_words_list)
    STOP_WORD = list(thai_stopwords())
    STOP_WORD.extend(["อยาก", "ค่ะ", "คะ", "บอก", "ที่ไหน", "อร่อย", "แนะนำ", "ตรงไหน", "ร้าน", "บ้าง", "ซื้อ", "กิน", "ดื่ม", "เด็ด", "ดัง", "ขาย"])


    def text_process(text):
        final ="".join(u for u in text if u not in('?'))
        final = word_tokenize(final, engine="newmm",custom_dict=trie)
        final = "".join(word for word in final if word not in STOP_WORD)
        return final
    
    data_df = pd.read_csv('dataset3.csv', names=['num', 'input_text', 'labels'])
    data_df['token'] = data_df['input_text'].apply(text_process)

    sentences = data_df['token'].values

    tokens = [word_tokenize(text_process(sentence))for sentence in sentences]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)
    word_index = tokenizer.word_index
    
    
    if request.method=='POST':
        payload =request.json
        Reply_token=payload['events'][0]['replyToken']
        message=payload['events'][0]['message']['text']

        message = str(message)
        model  = load_model('my_model_LSTM.h5')
        token = text_process(message)
        sequence = tokenizer.texts_to_sequences([token])
        x_real = pad_sequences(sequence, maxlen=12, padding='post')
        msg = np.argmax(model.predict(x_real), axis=1)[0]
        food_categories = {
            0: "น้ำหวาน",
            1: "เครป",
            2: "ขนมไทย",
            3: "เค้ก",
            4: "ไอศกรีม",
            5: "น้ำแข็งไส",
            6: "โรตี",
            7: "กาแฟ",
            8: "ขนมปัง",
            9: "ผลไม้"
        }
        Reply_text = food_categories.get(msg, "Unknown Category") 

        print(Reply_text,flush=True)
        ReplyMessage(Reply_token,message,Channel_access_token)
        return request.json,200
    elif request.method=='GET':
        return "this is method GET!!!",200
    else:
        abort(400)


def ReplyMessage(Reply_token,TextMessage,Line_Acees_Token):
    LINE_API='https://api.line.me/v2/bot/message/reply/'
    
    Authorization='Bearer {}'.format(Line_Acees_Token)
    print(Authorization)
    headers={
        'Content-Type':'application/json; char=UTF-8',
        'Authorization':Authorization
    }

    data={
        "replyToken":Reply_token,
        "messages":[{
            "type":"text",
            "text":TextMessage
        }
        ]
    }
    data=json.dumps(data) # ทำเป็น json
    r=requests.post(LINE_API,headers=headers,data=data)
    return 200