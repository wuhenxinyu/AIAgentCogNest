import requests
import time
import zipfile

api_key = 'eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1MDEzNzQ5OCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc2NDczOTM1NSwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTgyMjE0MjMzNjciLCJvcGVuSWQiOm51bGwsInV1aWQiOiIyOTUzMjU0NS01N2RlLTQ3ZDItOWEzZi05NDBhZjlkOTUzNWQiLCJlbWFpbCI6IiIsImV4cCI6MTc2NTk0ODk1NX0.7pA7k4jeQhu-IV7Wx1L_L5sYGo8xhgoWEtbHihEV1IsBKU28JOsyqksL_glA0d-5ppoVWI4BWgphvCR92BQ3Gg'

def get_task_id(file_name):
    url='https://mineru.net/api/v4/extract/task'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}".format(api_key)
    }
    pdf_url = 'https://vl-image.oss-cn-shanghai.aliyuncs.com/pdf/' + file_name
    data = {
        'url':pdf_url,
        'is_ocr':True,
        'enable_formula': False,
    }

    res = requests.post(url,headers=header,json=data)
    print(res.status_code)
    print(res.json())
    print(res.json()["data"])
    task_id = res.json()["data"]['task_id']
    return task_id

def get_result(task_id):
    url = f'https://mineru.net/api/v4/extract/task/{task_id}'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}".format(api_key)
    }

    while True:
        res = requests.get(url, headers=header)
        result = res.json()["data"]
        print(result)
        state = result.get('state')
        err_msg = result.get('err_msg', '')
        # 如果任务还在进行中，等待后重试
        if state in ['pending', 'running']:
            print("任务未完成，等待5秒后重试...")
            time.sleep(5)
            continue
        # 如果有错误，输出错误信息
        if err_msg:
            print(f"任务出错: {err_msg}")
            return
        # 如果任务完成，下载文件
        if state == 'done':
            full_zip_url = result.get('full_zip_url')
            if full_zip_url:
                local_filename = f"{task_id}.zip"
                print(f"开始下载: {full_zip_url}")
                r = requests.get(full_zip_url, stream=True)
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"下载完成，已保存到: {local_filename}")
                # 下载完成后自动解压
                unzip_file(local_filename)
            else:
                print("未找到 full_zip_url，无法下载。")
            return
        # 其他未知状态
        print(f"未知状态: {state}")
        return

# 解压zip文件的函数
def unzip_file(zip_path, extract_dir=None):
    """
    解压指定的zip文件到目标文件夹。
    :param zip_path: zip文件路径
    :param extract_dir: 解压目标文件夹，默认为zip同名目录
    """
    if extract_dir is None:
        extract_dir = zip_path.rstrip('.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"已解压到: {extract_dir}")

if __name__ == "__main__":
    file_name = '【财报】中芯国际：中芯国际2024年年度报告.pdf'
    task_id = get_task_id(file_name)
    print('task_id:',task_id)
    get_result(task_id)
