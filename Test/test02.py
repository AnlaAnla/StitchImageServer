import os
import shutil
from typing import List

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

# 创建一个目标文件夹来存放上传的文件
UPLOAD_DIRECTORY = "./uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

app = FastAPI()


@app.post("/upload-folder/")
async def upload_folder(files: List[UploadFile] = File(...)):
    """
    接收通过 webkitdirectory 上传的整个文件夹
    """
    saved_files = []
    for file in files:
        # file.filename 会包含从选定目录开始的相对路径
        # 例如: "my_folder/data.csv" 或 "my_folder/images/pic.png"

        # 安全性检查：防止路径遍历攻击 (e.g., "my_folder/../../etc/passwd")
        if ".." in file.filename:
            raise HTTPException(status_code=400, detail=f"Invalid filename: {file.filename}. Contains '..'")

        # 在服务器上创建完整的目标路径
        # os.path.join 会正确处理不同操作系统的路径分隔符
        destination_path = os.path.join(UPLOAD_DIRECTORY, file.filename)

        # 获取目标文件的目录路径
        destination_dir = os.path.dirname(destination_path)

        # 如果目录不存在，则创建它
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        try:
            # 异步地将文件内容写入目标路径
            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append(file.filename)
        finally:
            # 确保关闭文件
            await file.close()

    return {"message": f"Successfully uploaded {len(saved_files)} files", "filenames": saved_files}


# 提供一个简单的 HTML 上传页面用于测试
@app.get("/")
async def main():
    content = """
    <body>
    <h2>上传整个文件夹</h2>
    <p>选择一个文件夹，其中的所有文件（包括子目录中的文件）都将被上传。</p>
    <form action="/upload-folder/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" webkitdirectory directory multiple>
    <input type="submit">
    </form>
    </body>
    """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)