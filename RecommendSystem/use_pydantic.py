"""
Pydantic?

Pydantic의 공식문서에는 다음과 같이 설명되어 있다.

Python type 주석을 이용한 데이터 유효성 검사 및 설정 관리 pydantic은 런타임에 유형 힌트를 적용하고, 
데이터가 유효하지 않은 경우 사용자에게 친숙한 오류를 제공한다. 
데이터가 순수한 표준 Python에 어떻게 포함되어야 하는지 정의하고, Pydantic으로 검증한다

즉, 데이터를 검증하는 라이브러리인 셈이다.
"""
from typing import List, Optional, Union
from datetime import datetime

from pydantic import BaseModel, Field

class Movie(BaseModel):  # BaseModel을 상속했는데, 이는 객체를 정의하는 방법을 제공한다.
    mid: int
    genre: str
    rate: Union[int, float]  # rate는 int혹은 float이고
    tag: Optional[str] = None  # tag는 str이고 기본값은 None이며
    date: Optional[datetime] = None  # date는 datetime 형식이어야 한다. 기본값은 None이며
    some_variable_list: List[int] = []  # 임의의 변수는 정수형 리스트 형태이어야 하며, 기본값은 빈 배열이다.
    
# 위에서 정해둔 데이터 형식 외 다른 형식으로 데이터를 구성하면 pydantic은 에러를 발생시킨다.
# 이처럼 pydantic은 데이터의 type이 올바르게 들어올 수 있도록 검증해준다.
tmp_data = {
    "mid" : "1",
    "genre" : "action",
    "rate" : 1.5,
    "tag" : None,
    "date" : "2024-02-19 13:44:11"
}

tmp_movie = Movie(**tmp_data)
print(tmp_movie.json())

# Field를 활용하면 데이터의 범위, 길이 제한 등을 설정할 수 있다.
class User(BaseModel):
    '''
    gt : 설정된 값보다 큰
    ge : 설정된 값보다 크거나 같은
    lt : 설정된 값보다 작은
    le : 설정된 값보다 작거나 같은
    '''
    user_id: int
    name: str = Field(min_length=2, max_length=7)  # 2글자 이상, 7글자 이하
    age: int = Field(gt=1, le=130)  # 1 초과 130 이하
    
tmp_user_data = {
    "user_id" : "100",
    "name" : "heonwoo",
    "age" : "12"
}

tmp_user = User(**tmp_user_data)
print(tmp_user.json())