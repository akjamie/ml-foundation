# 导入所需的库
import json
import os

import requests
import time
import dotenv

dotenv.load_dotenv()

# 定义爬取微博用户信息的函数
def scrape_weibo(url: str):
    '''爬取相关鲜花服务商的资料'''
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
        "Referer": "https://weibo.com"
    }
    cookies = {
        # "cookie": '''SINAGLOBAL=3762226753815.13.1696496172299; ALF=1699182321; SCF=AiOo8xtPwGonZcAbYyHXZbz9ixm97mWi0vHt_VvuOKB-u4-rcvlGtWCrE6MfMucpxiOy5bYpkIFNWTj7nYGcyp4.; _sc_token=v2%3A2qyeqD3cTZFNTl0sn3KAYe4fNqzMUEP-C7nxNsd_Q1r-vpYMlF2K3xc4vWNuLNBbp3RsohghkJdlSVN09cymVo5AKAm0V92004V8cSRe9O5v9B65jd4yiG_sATDeB06GnjiJulXUrEF_6XsHh1ozK6jvbTKEUIkF7v0_BlbX6IcWrPkwh6xL_WM_0YUV2v7CtNPwyxfbAjaWnG32TsxG_ftN3s5m7qfaRftU6iTOSnE%3D; XSRF-TOKEN=4o0E6jaUQ0BlN77az0sURTg3; PC_TOKEN=dcf0e7607f; login_sid_t=36ebf31f1b3694fb71e77e35d30f052f; cross_origin_proto=SSL; WBStorage=4d96c54e|undefined; _s_tentry=passport.weibo.com; UOR=www.google.com,weibo.com,login.sina.com.cn; Apache=7563213131783.361.1696667509205; ULV=1696667509207:2:2:2:7563213131783.361.1696667509205:1696496172302; wb_view_log=3440*14401; WBtopGlobal_register_version=2023100716; crossidccode=CODE-gz-1QP2Jh-13l47h-79FGqrAQgQbR8ccb7b504; SSOLoginState=1696667553; SUB=_2A25IJWfwDeThGeFJ6lsQ-SbNzjuIHXVr5gm4rDV8PUJbkNAbLUWtkW1NfJd_XHamKIzj5RlT_-RGMma6z3YQZUK3; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFDKvBlvg14YuHk_4c6MEH_5NHD95QNS024eK.ReK-NWs4DqcjZCJ8oIN.pSKzceBtt; WBPSESS=gyY2mn77F4p5VxWF2IB_yFR0phHVTNfaJAHAMprnW7MeUr-NHPZNyeeyKae3tHELlc_RbcI1XPSz-TjSJqWrIXs-yh1fwhxL4mSDrnpPZEogFt8ScF5NEwSqPGn7x2KMAgTHtWde-3MBm6orQ98PDA=='''
        "cookie": os.environ['WEIBO_COOKIE']
    }
    response = requests.get(url, headers=headers, cookies=cookies)
    time.sleep(5)   # 加上3s 的延时防止被反爬
    return response.text

# 根据UID构建URL爬取信息
def get_data(id):
    url = "https://weibo.com/ajax/profile/detail?uid={}".format(id)
    html = scrape_weibo(url)
    response = json.loads(html)

    return response
