from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.api.stitch import router as stitch_router

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 应用初始化 ---
app = FastAPI(
    title="拼图API",
    description="一个提供单张或批量图片拼接功能的服务。"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(stitch_router, prefix="/api")
