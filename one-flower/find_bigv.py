import os
import re

import dotenv
from langchain_community.llms.tongyi import Tongyi
from langchain_core.prompts import PromptTemplate

from agents.weibo_agent import lookup_v
from tools.general_tool import remove_non_chinese_fields
from tools.scraping_tool import get_data
from tools.text_gen_tool import generate_letter

dotenv.load_dotenv()


def find_bigv(flower: str):
    llm = Tongyi(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"), temporature=0, verboose=True)
    response_uid = lookup_v(flower_type=flower, llm=llm)
    print(f"response_uid:{response_uid}")

    if None != response_uid and None != response_uid['output'] and ('I-DONOT-KNOW' not in (response_uid['output'])):
        uid = re.findall(r'\d+', response_uid['output'])[0]
        print(f"uid:{uid}")

        if None == uid:
            raise Exception("未找到该花名对应的大V")

        # 根据UID爬取大V信息
        person_info = get_data(uid)
        print(person_info)
        # 移除无用的信息
        remove_non_chinese_fields(person_info)
        print(person_info)

        # 初始化链
        result = generate_letter(llm, person_info)
        print(f'result:{result}')

        return result
    else:
        raise Exception("未找到该花名对应的大V")

# if __name__ == '__main__':
#     llm = Tongyi(dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"), temporature=0, verboose=True)
#     response_uid = lookup_v(flower_type="牡丹", llm=llm)
#     print(f"response_uid:{response_uid}")
#
#     uid = re.findall(r'\d+', response_uid['output'])[0]
#     print(f"uid:{uid}")
#
#     # 根据UID爬取大V信息
#     person_info = get_data(uid)
#     print(person_info)
#     # 移除无用的信息
#     remove_non_chinese_fields(person_info)
#     print(person_info)
#
#     # 初始化链
#     result = generate_letter(llm, person_info)
#     print(result)
