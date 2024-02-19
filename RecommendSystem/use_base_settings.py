"""
BaseSetting

이전 파일에서는 BaseModel을 활용해서 객체를 정의했다. 
BaseSetting을 활용하면 Setting과 관련된 것들을 정의하고, setting에 사용되는 env와 같은 것들이 default로 읽어져온다.
"""
from pydantic import BaseSettings, Field, validator

class DBConfig(BaseSettings):
    host: str = Field(default="127.0.0.1", env="db_host")
    port: int = Field(default=3306, env="db_port")
    
    @validator("port")
    def check_port(cls, port_input):
        if port_input not in [3306, 8080]:  # 만약 port가 3306 이나 8080이 아니라면
            raise ValueError("port error")  # ValueError를 발생시킨다.
        return port_input
    
    class Config:
        env_file = ".env_ex"
        
print(DBConfig().dict())
