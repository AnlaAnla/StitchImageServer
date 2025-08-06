import uvicorn

if __name__ == "__main__":
    # 使用 uvicorn 启动应用
    # reload=True 可以在开发时代码改变后自动重启服务
    print("http://127.0.0.1:7745/docs")
    uvicorn.run("app.main:app", host="0.0.0.0", port=7745, reload=False)
