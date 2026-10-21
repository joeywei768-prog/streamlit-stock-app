import os
import re
import json

import pandas as pd
import streamlit as st
import tushare as ts
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# 0. 页面配置（必须是第一个 st 命令）
# ============================================================
st.set_page_config(page_title="股价波动预测 APP", page_icon="📈", layout="wide")


# ============================================================
# 1. 密钥读取：优先 st.secrets（部署用），其次 .env（本地用）
# ============================================================
load_dotenv(override=True)


def get_secret(name: str):
    """先查 st.secrets，再查环境变量/.env，都没有返回 None。"""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name)


TUSHARE_TOKEN = get_secret("TUSHARE_TOKEN")
DEEPSEEK_API_KEY = get_secret("DEEPSEEK_API_KEY")

missing = [n for n, v in [("TUSHARE_TOKEN", TUSHARE_TOKEN),
                          ("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)] if not v]
if missing:
    st.error(f"缺少密钥：{', '.join(missing)}。请在 `.streamlit/secrets.toml` 或 `.env` 中配置。")
    st.code(
        'TUSHARE_TOKEN = "你的tushare token"\n'
        'DEEPSEEK_API_KEY = "你的deepseek key"',
        language="toml",
    )
    st.stop()


# ============================================================
# 2. 资源初始化（cache_resource：只加载一次，所有会话共享）
# ============================================================
@st.cache_resource
def init_tushare(token: str):
    ts.set_token(token=token)
    return ts.pro_api()


@st.cache_resource
def get_llm_client(api_key: str):
    return OpenAI(base_url="https://api.deepseek.com", api_key=api_key)


pro = init_tushare(TUSHARE_TOKEN)
llm_client = get_llm_client(DEEPSEEK_API_KEY)


# ============================================================
# 3. 股票池
# ============================================================
STOCKS = {
    "000001.SZ": "平安银行",
    "600519.SH": "贵州茅台",
    "002594.SZ": "比亚迪",
    "300750.SZ": "宁德时代",
}


# ============================================================
# 4. 取数（cache_data：参数相同就直接返回缓存，不重复打 API）
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price(ts_code: str, days: int = 90) -> pd.DataFrame:
    """取近 days 个自然日的后复权日线，返回按日期升序的 DataFrame。"""
    end = pd.Timestamp.now().strftime("%Y%m%d")
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y%m%d")
    bar = ts.pro_bar(ts_code=ts_code, adj="hfq", start_date=start, end_date=end)
    if bar is None or bar.empty:
        return pd.DataFrame()
    bar = bar.sort_values("trade_date").reset_index(drop=True)
    bar["trade_date"] = pd.to_datetime(bar["trade_date"])
    return bar


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_daily_basic(ts_code: str, n_days: int = 5) -> pd.DataFrame:
    """倒推取最近 n_days 个交易日的每日指标；非交易日自动跳过。"""
    fields = "ts_code,trade_date,turnover_rate,volume_ratio,pe,pb"
    df = pd.DataFrame()
    today = pd.Timestamp.now()
    for i in range(15):  # 多倒推几天，容错周末/节假日
        trade_date = (today - pd.Timedelta(days=i)).strftime("%Y%m%d")
        try:
            daily = pro.daily_basic(ts_code=ts_code, trade_date=trade_date, fields=fields)
            if daily is not None and not daily.empty:
                df = pd.concat([df, daily], ignore_index=True)
        except Exception:
            continue
        if len(df) >= n_days:
            break
    return df.head(n_days).reset_index(drop=True)


# ============================================================
# 5. 大模型调用 + prompt 构造 + JSON 提取
# ============================================================
def llm_chat(prompt: str) -> str:
    completion = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是人工智能助手，擅长金融数据分析。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content


def build_prompt(firm_name: str, df: pd.DataFrame) -> str:
    question = (
        f"will future price of this stock {firm_name} move up or down "
        f"in the subsequent fifth trading day?"
    )
    resolution_criteria = "预测的判断标准：股票价格在随后的第五个交易日是否高于或低于当前价格。"
    materials = str(df.to_dict(orient="records"))

    return f"""
You are an expert in future event prediction.
- Question: {question}
- Resolution Criteria: {resolution_criteria}
- Materials: {materials}

Solve the problem step by step as required by the instructions.

Instructions:
    1. 重述并扩展问题，明确要预测的目标。
    2. 列出支持上涨的理由，并给出每条理由的强度（1-5）。
    3. 列出支持下跌的理由，并给出每条理由的强度（1-5）。
    4. 综合正反两方，聚合出一个初始概率判断。
    5. 反思：你的判断是否受到材料中某一项的过度影响？如有请修正。
    6. 基于反思结果，给出最终概率。
    7. Output your final probability prediction
       (the sum of up and down probabilities must equal to 1).
       Your final prediction MUST be in JSON format with keys: up and down
       (values are decimals between 0 and 1, e.g. {{"up": 0.62, "down": 0.38}}).
    8. 回答问题用中文，JSON 放在最后一行。
"""


def extract_json(text: str) -> dict:
    """从模型长回复中抠出 {"up": ..., "down": ...}，并归一化。"""
    if not text:
        return {"up": 0.0, "down": 0.0}

    # 优先匹配带 up/down 的 JSON 片段（顺序不限）
    patterns = [
        r'\{[^{}]*"up"[^{}]*"down"[^{}]*\}',
        r'\{[^{}]*"down"[^{}]*"up"[^{}]*\}',
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            try:
                obj = json.loads(m.group(0))
                up = float(obj.get("up", 0))
                down = float(obj.get("down", 0))
                total = up + down
                if total > 0:
                    return {"up": up / total, "down": down / total}
            except Exception:
                continue

    # 兜底：直接抓 up/down 后面的数字
    up_m = re.search(r'"up"\s*:\s*([0-9.]+)', text)
    down_m = re.search(r'"down"\s*:\s*([0-9.]+)', text)
    if up_m and down_m:
        up = float(up_m.group(1))
        down = float(down_m.group(1))
        total = up + down
        if total > 0:
            return {"up": up / total, "down": down / total}

    return {"up": 0.0, "down": 0.0}


# ============================================================
# 6. 页面布局
# ============================================================
st.title("📈 股价波动预测 APP")
st.caption("数据来源：tushare · 模型：DeepSeek · 教学演示，不构成投资建议")

with st.sidebar:
    st.header("控制面板")
    selected = st.selectbox("选择股票", [f"{c} {n}" for c, n in STOCKS.items()])
    lookback = st.slider("行情显示最近 N 个交易日", min_value=10, max_value=60, value=30)
    run_btn = st.button("🚀 开始预测", type="primary", use_container_width=True)
    st.divider()
    st.caption("预测耗时约 10–30 秒，请耐心等待。")

code, name = selected.split()

# 换股票时清掉旧的预测结果，避免张冠李戴
if st.session_state.get("pred_code") != code:
    st.session_state.pop("pred", None)
    st.session_state.pop("answer", None)
    st.session_state["pred_code"] = code


# ---------- 6.1 行情图 ----------
st.subheader(f"{name}（{code}）· 后复权收盘价")
with st.spinner("加载行情数据..."):
    bar = fetch_price(code)

if bar.empty:
    st.warning("未取到行情数据，请检查 token 权限或稍后重试。")
else:
    chart_df = bar.set_index("trade_date")["close"].tail(lookback)
    st.line_chart(chart_df)

    latest = bar["close"].iloc[-1]
    prev = bar["close"].iloc[-2] if len(bar) > 1 else latest
    c1, c2, c3 = st.columns(3)
    c1.metric("最新收盘价（后复权）", f"{latest:.2f}", delta=f"{latest - prev:+.2f}")
    c2.metric("最近交易日", bar["trade_date"].iloc[-1].strftime("%Y-%m-%d"))
    c3.metric("样本天数", f"{len(bar)} 天")


# ---------- 6.2 每日指标表 ----------
st.subheader("最近 5 个交易日 · 每日指标")
with st.spinner("加载每日指标..."):
    basic_df = fetch_daily_basic(code)

if basic_df.empty:
    st.warning("未取到每日指标数据（可能是 token 积分不足或近期非交易日）。")
else:
    st.dataframe(basic_df, use_container_width=True, hide_index=True)
    st.caption("turnover_rate 换手率 · volume_ratio 量比 · pe 市盈率 · pb 市净率")


# ---------- 6.3 按钮触发预测 ----------
if run_btn:
    if basic_df.empty:
        st.error("没有可用的指标数据，无法预测。")
    else:
        with st.spinner("AI 正在分析中，请稍候..."):
            try:
                prompt = build_prompt(name, basic_df)
                answer = llm_chat(prompt)
                pred = extract_json(answer)
                st.session_state["pred"] = pred
                st.session_state["answer"] = answer
            except Exception as e:
                st.error(f"预测失败：{e}")


# ---------- 6.4 展示预测结果（跨 rerun 保留） ----------
if "pred" in st.session_state and st.session_state.get("pred_code") == code:
    pred = st.session_state["pred"]
    answer = st.session_state.get("answer", "")

    st.subheader("🤖 AI 预测结果")
    p1, p2 = st.columns(2)
    p1.metric("上涨概率（第 5 个交易日）", f"{pred['up']:.1%}")
    p2.metric("下跌概率（第 5 个交易日）", f"{pred['down']:.1%}")

    if abs(pred["up"] + pred["down"] - 1) > 0.01:
        st.warning("⚠️ 概率和不为 1，JSON 提取可能异常，请查看原始回复。")

    with st.expander("查看 AI 完整分析过程"):
        st.write(answer)