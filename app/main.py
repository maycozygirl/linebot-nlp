
from flask import Flask,request,abort
import requests
from app.Config import *
from keras.models import load_model
from pythainlp import word_tokenize
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
from pythainlp.corpus import thai_stopwords
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
    FORMAT = r"[\u0E00-\u0E7Fa-zA-Z'0-9]+"
    def tokenize(sentence):
        return word_tokenize(sentence, engine="newmm",custom_dict=trie)

    def cleaning_stop_word(tk_list):
        return [word for word in tk_list if word not in STOP_WORD]

    def cleaning_symbols_emoji(tk_list):
        return [re.findall(FORMAT, text)[0] for text in tk_list if re.findall(FORMAT, text)]

    def big_cleaning(sentence):
        return  cleaning_symbols_emoji( cleaning_stop_word( tokenize(sentence) ) )
    def text_process(text):
        final ="".join(u for u in text if u not in('?'))
        final = word_tokenize(final, engine="newmm",custom_dict=trie)
        final = "".join(word for word in final)
        final = "".join(word for word in final if word not in STOP_WORD)
        return final
    
    if request.method=='POST':
        payload =request.json
        Reply_token=payload['events'][0]['replyToken']
        message=payload['events'][0]['message']['text']

        message = str(message)
        model  = load_model('my_model_LSTM.h5')

        text_process(message)
        
        if "ขายอะไร" in message:
            Reply_text="- กาแฟ และเครื่องดื่มค่ะ เช่น\n- คาปูชิโน่\n- ลาเต้\n- อเมริกาโน่\n - ชาไทย"
        elif "สวัสดี" in message:
            Reply_text="สวัสดีค่ะ"
        else:
            Reply_text="ขออภัยค่ะ ฉันไม่เข้าใจคำถาม กรุณาถามคำถามใหม่ค่ะ"
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