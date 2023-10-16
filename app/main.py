
from flask import Flask,request,abort
import requests
from app.Config import *
from keras.models import load_model
from pythainlp import word_tokenize
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
from pythainlp.corpus import thai_stopwords
import numpy as np
import re
import json




app=Flask(__name__)

@app.route('/webhook',methods=['POST','GET'])


def webhook():
    words = ['กาแฟ', 'น้ำหวาน', 'เค้ก', 'เครป', 'ไอศครีม','เครป','ผลไม้','น้ำผลไม้','โรตี','ขนมไทย','น้ำแข็งไส','ขนมปัง','แซนวิช','อเมริกาโน่','แซนวิช','น้ำสมุนไพร','เตยแก้ว','บอก','โอริโอ้']
    custom_words_list = set(thai_words())
    custom_words_list.update(words)
    trie = dict_trie(dict_source=custom_words_list)
    STOP_WORD = list(thai_stopwords()) ;
    STOP_WORD.append ("อยาก")
    STOP_WORD.append ("ค่ะ")
    STOP_WORD.append ("คะ")
    STOP_WORD.append ("บอก")
    STOP_WORD.append ("ที่ไหน")
    STOP_WORD.append ("อร่อย")
    STOP_WORD.append ("แนะนำ")
    STOP_WORD.append ("ตรงไหน")
    STOP_WORD.append ("ร้าน")
    STOP_WORD.append ("บ้าง")
    STOP_WORD.append ("ซื้อ")
    STOP_WORD.append ("กิน")
    STOP_WORD.append ("ดื่ม")
    STOP_WORD.append ("เด็ด")
    STOP_WORD.append ("ดัง")
    

    def text_process(text):
        final ="".join(u for u in text if u not in('?'))
        final = word_tokenize(final, engine="newmm",custom_dict=trie)
        final = "".join(word for word in final if word not in STOP_WORD)
        return final
    
    if request.method=='POST':
        payload =request.json
        Reply_token=payload['events'][0]['replyToken']
        message=payload['events'][0]['message']['text']

        message = str(message)
        model  = load_model('my_model_LSTM.h5')

        token = text_process(message)
    
        msg = np.argmax(model.predict(token),axis=1)

        if 0 in msg:
            Reply_text = "น้ำหวาน"
        elif 1 in msg:
            Reply_text = "เครป"
        elif 2 in msg:
            Reply_text = "ขนมไทย"
        elif 3 in msg:
            Reply_text = "เค้ก"
        elif 4 in msg:
            Reply_text = "ไอศกรีม"
        elif 5 in msg:
            Reply_text = "น้ำแข็งไส"
        elif 6 in msg:
            Reply_text = "โรตี"
        elif 7 in msg:
            Reply_text = "กาแฟ"
        elif 8 in msg:
            Reply_text = "ขนมปัง"
        elif 9 in msg:
            Reply_text = "ผลไม้"
            
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