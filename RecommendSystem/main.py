from fastapi import FastAPI
from pydantic import BaseModel
from pydantic import Field
import time
import asyncio

app = FastAPI()

# class DataInput(BaseModel):
#     name: str
    
# @app.get("/")
# def home():
#     return {"Hello" : "GET"}

# @app.post("/")
# def home_post(data_request: DataInput):
#     return {"Hello" : "POST", "msg" : data_request.name}

# async def와 def의 차이점
# 만약 await를 호출하도록 가이드하는 서드파티 라이브러리를 사용하는 경우 async def를 사용하고 그렇지 않은 경우에는 def와 같이 사용한다
# 잘 모르겠으면 일반적인 정의인 def를 사용하라고 공식 라이브러리에서는 권장하고 있다.
# 어떠한 경우에도 FastAPI는 비동기적으로 작동하고 매우 빠르다고 한다.
# 아래 예시를 살펴보자.

async def some_library(num: int, something: str):
    s = 0
    for i in range(num):
        print("something.. : ", something, i)
        # time.sleep(1)  # time.sleep은 비동기적으로 처리할 수 없기 때문에
        await asyncio.sleep(1)  # 이와 같이 처리해야 한다.
        s += int(something)
    return s

@app.post("/")
async def read_results(something: str):
    s1 = await some_library(5, something)
    return {"data" : "data", "s1" : s1}