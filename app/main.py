
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
    STOP_WORD.extend(["อยาก", "ค่ะ", "คะ", "บอก", "ที่ไหน", "อร่อย", "แนะนำ", "ตรงไหน", "ร้าน", "บ้าง", "ซื้อ", "กิน", "ดื่ม", "เด็ด", "ดัง", " "])


    def text_process(text):
        final ="".join(u for u in text if u not in('?'))
        final = word_tokenize(final, engine="newmm",custom_dict=trie)
        final = "".join(word for word in final if word not in STOP_WORD)
        return final
    
    data_df = pd.read_csv('./data/dataset.csv', names=['num', 'input_text', 'labels'])
    data_df['token'] = data_df['input_text'].apply(text_process)

    sentences = data_df['token'].values

    tokens = [word_tokenize(text_process(sentence))for sentence in sentences]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)

    
    if request.method=='POST':
        payload =request.json
        Reply_token=payload['events'][0]['replyToken']
        message=payload['events'][0]['message']['text']

        message = str(message)
        model  = load_model('./model/my_model_LSTM.h5')

        token = word_tokenize(text_process(message))

        sequence = tokenizer.texts_to_sequences([token])
        x_real = pad_sequences(sequence, maxlen=12, padding='post')
        predict = model.predict(x_real)
        categories = np.argmax(predict)
        food_categories = {
            0: "เราแนะนำร้านเตยแก้วที่บาร์ใหม่เพราะมีเมนูให้เลือกหลากหลาย มีเตยแก้ว น้ำสมุนไพรทั้งเมนูนมปั่น น้ำผลไม้ รสชาติอร่อย สะอาด นักศึกษาเลือกซื้อกันเยอะร้าน\nดอกจิกบาร์วิทย์ มีชานม ชาโซดา มีเมนูนมกล้วย รสชาติอร่อย ราคาเป็นมิตร \nร้านชอบปลาชุม เป็นร้านที่บรรยากาศสวยงามเพราะมีตู้ปลาขนาดใหญ่ร้านตกแต่งแนวอควาเรียม มีเมนูให้เลือกหลากหลาย มีทั้งกาแฟและน้ำปั่น \nร้านน้ำเต้าหู้เราแนะนำร้านโต้วตรงเศรษฐศาสตร์ มีทั้งแบบร้อนเย็น และปาท่องโก๋กรอบ",
            1: "เราแนะนำร้านเครปที่บาร์วิทย์เพราะราคาถูก อร่อยมีให้เลือกท็อปปิ้งได้หลากหลายแม่ค้าทำออร์เดอร์รวดเร็วคุณๆม่ต้องรอคิวนาน\nร้านเครปที่บาร์วิศวะเป็นร้านขึ้นชื่อของมหาวิทยาลัย เครปหน้าตาน่ากินรสชาติอร่อยด้วย",
            2: "เราแนะนำร้านสามพี่น้องกล้วยทอดที่บาร์ใหม่ มีขนมให้เลือกหลากหลาย ราคาถูกมาก รสชาติกลมกล่อมไม่หวานเกินไป\nเรายังแนะนำร้าน Sweet-หวาน ร้านนี้มีขนมให้เลือกมากมาย อยู่ในศูนย์อาหาร vet-ku  นอกจากจะมีขนมหวานแล้วยังมีขนมกรุบกรอบหรือผลไม้แช่อิ่มให้ท่านเลือกซื้อด้วย ",
            3: "เราแนะนำร้าน me eat เป็นร้านคาเฟ่เล็กๆที่มีทั้งขนมหวานและเครื่องดื่ม เค้กที่นี่น่าทานมากรสชาติละมุนละไม ราคาเป็นกันเอง \nนอกจากนี้ยังมีร้าน S&P ให้ท่านเลือกซื้อเค้กหลากหลายชนิด มีขนมปังอบใหม่น่ากิน",
            4: "เราแนะนำร้าน me eat มีไอศกรีมแบล้กซีรีย์ให้ท่านเลือกซื้อหลากหลายรสชาติ\nอีกร้านที่แนะนำคือร้าน N&B ที่บาร์ใหม่ ร้านนี้มีท็อปปิ้งให้เลือกกินกับไอสกรีมด้วย",
            5: "เราแนะนำร้านน้ำแข็งใสคุณหมิงที่บาร์ซอย คณะปฐพีวิทยา ราคาถูกมากแถมรสชาติอร่อย เลือกท็อปปิ้งได้ตามใจชอบ \nอีกร้านที่แนะนำเราแนะนำร้าน izekimo อยู่ใกล้ตึก SC เป็นร้านที่ใส่ท็อปปิ้งผลไม้หลากหลายชนิด กินแล้วสดชื่นหวานอร่อย \nหากท่านชอบปังเย็น เราแนะนำร้านปลุกปั่น หวานเย็นเหมาะแก่การทานคลายเครียด",
            6: "เราแนะนำโรตีบังบาร์วิศวะ โรตีมีให้เลือกหลายชนิด เช่นโรตีกลม โรตีเหลี่ยม หรือมะตะบะ ราคาถูก ทำรวดเร็ว รสชาติอร่อย เป็นร้านขึ้นชื่อของบาร์วิศวะ",
            7: "เราแนะนำร้าน Hottobun เป็นคาเฟ่ขึ้นชื่อของมหาวิทยาลัยเกษตรศาสตร์ มีขนมให้เลือกทานคู่กับกาแฟด้วย \n ร้าน The boon cafe ร้านคาเฟ่ที่สวยน่านั่ง คณะเศรษฐศาสตร์ กาแฟอร่อยรสชาติละมุน \nอีกร้านที่แนะนำคือร้าน Komma cafe ตรง ku avenue ร้านบรรยากาศร่มรื่น มีเมนูให้เลือกหลากหลาย",
            8: "เราแนะนำร้าน eat me cafe คาเฟ่เล็กๆบรรยากาศน่านั่ง มีขนมปังอบใหม่ ขนมเค้ก ไอศกรีมและเครื่องดื่มทานคู่กันได้\nร้าน Maru waffle บาร์บัส เหมาะสำหรับคนชอบทานวาฟเฟิล\nร้านโทระฉะ บาร์บัส เหมาะสำหรับคนชอบทานวาฟเฟิล\nร้านกวางเจา บาร์ใหม่กว่า มีปาท่องโก๋เสิร์ฟคู่กับน้ำเต้าหูเหมาะเป็นมื้อเช้า\nร้านdou ตรงคณะเศรษฐศาสตร์ มีปาท่องโก๋กรอบกินคู่กับเซ็ทของหวาน \nร้านSweet-หวาน ศูนย์อาหาร vetku มีซาลาเปาไส้หวาน, ไส้ครีม และอีกหลากหลายไส้ให้เลือกสรรค์",
            9: "เราแนะนำเจ้แหมวผลไม้บาร์ใหม่ ผลไม้สดใหม่น่าทาน ราคาถูก แม่ค้าบริการดี \nอีกร้านที่แนะนำเราแนะนำร้านน้ำผึ้งผลไม้จากสวน เหมาะสำหรับซื้อผลไม้เก็บไปทานไว้ที่บ้าน ผลไม้เกรด A สดใหม่ "
        }

        if np.max(predict) >= 0.4:
            Reply_text = food_categories.get(categories)
        else:
            Reply_text = "ขออภัยค่ะ เราไม่รู้จักของหวานชนิดนี้"
    
        print(Reply_text,flush=True)
        ReplyMessage(Reply_token,Reply_text,Channel_access_token)
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