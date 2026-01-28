# -*- coding: utf-8 -*-
"""
GitHub Actions-ready postmarket email agent (free runner)
Key points:
- Reads secrets from ENV (GITHUB Actions secrets)
- Guards to only run at 17:00 US/Eastern on weekdays
- Pulls S&P500 tickers from slickcharts
- Scans movers (pre/post market) using yfinance (limited to top N to reduce time)
- Generates Traditional Chinese report via Gemini
- Sends email via Gmail SMTP (App Password recommended)
"""

import os
import re
import time
import unicodedata
from io import StringIO
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import pytz
import requests
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import smtplib
from email.message import EmailMessage
from email import policy


# =======================
# ENV / CONFIG
# =======================
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Secrets from GitHub Actions
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "").strip()
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "").strip()  # Gmail App Password
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "").strip()

# Controls (non-secret)
# How many tickers to scan for movers (speed control)
MAX_SCAN = int(os.getenv("MAX_SCAN", "1000"))  # default: scan first 150 S&P500 tickers
# Thresholds
THRESHOLD_TOP = float(os.getenv("THRESHOLD_TOP", "1.0"))      # % for top set
THRESHOLD_OTHERS = float(os.getenv("THRESHOLD_OTHERS", "1.0")) # % for others
# Top set definition (first N in slickcharts table)
TOP_SET_N = int(os.getenv("TOP_SET_N", "100"))

ALLOW_EMOJI = os.getenv("ALLOW_EMOJI", "false").lower() == "true"

from zoneinfo import ZoneInfo
EST = ZoneInfo("America/New_York")

# =======================
# UTF-8 / safety
# =======================
def sanitize_text_for_email(text: str) -> str:
    """
    Remove chars that sometimes break email environments.
    - normalize NFC
    - strip null bytes
    - remove most control chars except \n \t
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\x00", "")

    text = "".join(
        ch for ch in text
        if (ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    )

    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    return text


def ascii_log(*args):
    """Avoid printing non-ascii that could break some runners/log collectors."""
    safe = []
    for a in args:
        s = str(a)
        s = s.encode("ascii", errors="backslashreplace").decode("ascii", errors="ignore")
        safe.append(s)
    print(" ".join(safe), flush=True)


# =======================
# Time helpers
# =======================
def now_est() -> datetime:
    return datetime.now(EST)


def should_run_now() -> bool:
    """
    Safety guard:
    - only run on weekdays
    - only run at 17:00 (ET)
    """
    n = now_est()
    if n.weekday() >= 5:  # Sat/Sun
        return False
    return (n.hour == 17)


# =======================
# Data: S&P500 tickers
# =======================
def fetch_sp500_tickers() -> List[str]:
    url = "https://www.slickcharts.com/sp500"
    for attempt in range(3):
        try:
            html = requests.get(url, headers=HEADERS, timeout=30).text
            df = pd.read_html(StringIO(html))[0]
            # slickcharts table column is usually "Symbol"
            tickers = df["Symbol"].astype(str).str.strip().tolist()
            return [t for t in tickers if t and t != "nan"]
        except Exception as e:
            ascii_log("SP500_FETCH_FAILED", f"attempt={attempt+1}", repr(e))
            time.sleep(2)
    raise RuntimeError("Failed to fetch S&P 500 tickers from slickcharts.")


# =======================
# Gemini model
# =======================
def get_available_model() -> str:
    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-pro-latest",
    ]

    available = []
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            available.append(m.name)

    for p in preferred:
        if p in available:
            return p
    if available:
        return available[0]
    raise RuntimeError("No generateContent-capable models available.")


# =======================
# Market data fetch
# =======================
def safe_prev_close(ticker: yf.Ticker) -> float:
    try:
        info = ticker.info or {}
        pc = info.get("previousClose", None)
        if pc is None:
            return 0.0
        return float(pc) if pc else 0.0
    except Exception:
        return 0.0


def extract_yf_news_url(n: dict) -> str:
    """
    yfinance tk.news item -> best-effort URL (Yahoo canonical/clickThrough).
    """
    try:
        c = n.get("content") or {}

        # Prefer canonicalUrl.url (usually the clean Yahoo Finance page)
        cu = (c.get("canonicalUrl") or {}).get("url") or ""
        if isinstance(cu, str) and cu.startswith("http"):
            return cu

        # Fallback: clickThroughUrl.url
        ct = (c.get("clickThroughUrl") or {}).get("url") or ""
        if isinstance(ct, str) and ct.startswith("http"):
            return ct

        # Fallback: previewUrl (sometimes present)
        pv = c.get("previewUrl") or ""
        if isinstance(pv, str) and pv.startswith("http"):
            return pv
    except Exception:
        pass
    return ""


def parse_tk_news(tk: yf.Ticker, max_items: int = 2) -> List[Dict[str, str]]:
    """
    Returns [{"title": "...", "url": "...", "publisher": "..."}]
    from tk.news (Yahoo Finance via yfinance). Same-source title+url.
    """
    out: List[Dict[str, str]] = []
    try:
        raw = tk.news if getattr(tk, "news", None) else []
        for n in (raw[:max_items] if raw else []):
            if not isinstance(n, dict):
                continue
            c = n.get("content") or {}
            title = (c.get("title") or n.get("title") or "").strip()
            url = extract_yf_news_url(n)
            provider = (c.get("provider") or {}).get("displayName") or ""

            if title and url:
                out.append({"title": title, "url": url, "publisher": provider})
            elif title:
                # still keep title if url missing, but your sample shows url usually exists
                out.append({"title": title, "url": "", "publisher": provider})
    except Exception:
        pass
    return out


def safe_last_price_1m(hist: pd.DataFrame) -> float:
    try:
        return float(hist["Close"].iloc[-1])
    except Exception:
        return 0.0


def compute_postmarket_move_from_hist(hist: pd.DataFrame, now_et: datetime) -> Tuple[float, float, float]:
    """
    Returns (close_4pm, post_last, post_move_pct)

    close_4pm: today's regular-session close (last bar in 09:30–16:00 ET)
    post_last: last traded price in post-market up to now_et (16:01–20:00 ET)
    post_move_pct: % change from close_4pm to post_last
    """
    if hist is None or hist.empty:
        raise ValueError("empty hist")

    h = hist.copy()

    # Ensure timezone-aware, then convert to ET
    if getattr(h.index, "tz", None) is None:
        h.index = h.index.tz_localize("UTC")
    h = h.tz_convert("America/New_York")

    # Regular session close (baseline)
    reg = h.between_time("09:30", "16:00")
    if reg.empty:
        raise ValueError("no regular session bars")
    close_4pm = float(reg["Close"].iloc[-1])

    # Post-market (avoid 16:00 close-auction noise)
    post = h.between_time("16:01", "20:00")
    post = post[post.index <= now_et]

    # Optional: filter out zero-volume prints (safer)
    if "Volume" in post.columns:
        post = post[post["Volume"] > 0]

    if post.empty:
        return close_4pm, close_4pm, 0.0

    post_last = float(post["Close"].iloc[-1])
    post_move_pct = ((post_last / close_4pm) - 1.0) * 100.0

    return close_4pm, post_last, post_move_pct


def fetch_movers(
    tickers: List[str],
    top_set: set,
    max_scan: int = 1000
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns winners, losers (each up to 3 items):
    item = {symbol, change, price, threshold, news}
    """
    winners: List[Dict] = []
    losers: List[Dict] = []
    start = time.time()


    scanned = 0
    for symbol in tickers:
        scanned += 1
        if scanned > max_scan:
            break

        # Stop early if we already have enough movers
        if len(winners) >= 3 and len(losers) >= 3:
            break

        ascii_log("Scanning", symbol, f"({scanned}/{max_scan})")
        try:
            tk = yf.Ticker(symbol)

            # 1m prepost; can be heavy, so keep it best-effort
            hist = tk.history(period="1d", interval="1m", prepost=True)
            if hist is None or hist.empty:
                continue

            # Real post-market move vs today's 4pm close
            close_4pm, post_last, change = compute_postmarket_move_from_hist(hist, now_est())

            threshold = THRESHOLD_TOP if symbol in top_set else THRESHOLD_OTHERS

            if abs(change) < threshold:
                continue

            # news (same-source title+url from tk.news)
            news_items = parse_tk_news(tk, max_items=2)

            # Optional debug
            if news_items and news_items[0].get("url"):
                ascii_log("NEWS_OK", symbol, news_items[0]["url"][:120])
            else:
                ascii_log("NEWS_EMPTY", symbol)

            item = {
                "symbol": symbol,
                "change": f"{change:.2f}%",
                "price": round(post_last, 2),     # post-market last price
                "close": round(close_4pm, 2),     # today's 4pm close (debug/trace)
                "threshold": f"{threshold:.0f}%",
                "news": news_items,  # title + url
            }

            if change > 0 and len(winners) < 3:
                winners.append(item)
            elif change < 0 and len(losers) < 3:
                losers.append(item)

        except Exception as e:
            ascii_log("MOVER_SCAN_FAILED", symbol, repr(e))

    ascii_log(
        "Mover scan done.",
        "scanned=", scanned,
        "winners=", len(winners),
        "losers=", len(losers),
        "seconds=", round(time.time() - start, 2)
    )
    return winners, losers


def fetch_earnings_best_effort(
    tickers: List[str],
    target_date,
    max_scan: int = 1000
) -> List[str]:
    """
    Earnings calendar via yfinance per ticker is inconsistent & slow.
    Best-effort. target_date is a datetime.date you want to match.
    """
    out: List[str] = []

    scanned = 0
    for symbol in tickers:
        scanned += 1

        if scanned % 50 == 0:
            ascii_log("EARNINGS_SCAN_PROGRESS", scanned)

        if scanned % 20 == 0:
            time.sleep(0.5)
        if scanned > max_scan:
            break

        try:
            tk = yf.Ticker(symbol)
            cal = tk.calendar
            if cal is None:
                continue

            target = None
            if isinstance(cal, dict):
                v = cal.get("Earnings Date") or cal.get("EarningsDate")
                if isinstance(v, (list, tuple)) and v:
                    target = v[0]
                else:
                    target = v
            else:
                try:
                    if hasattr(cal, "loc") and "Earnings Date" in getattr(cal, "index", []):
                        target = cal.loc["Earnings Date"].iloc[0]
                    elif hasattr(cal, "columns") and "Earnings Date" in getattr(cal, "columns", []):
                        target = cal["Earnings Date"].iloc[0]
                except Exception:
                    target = None

            if target is None:
                continue

            if hasattr(target, "date"):
                target = target.date()

            if target == target_date:
                out.append(symbol)

        except Exception:
            continue

    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)

    ascii_log("EARNINGS_SCAN_DONE", "scanned=", scanned, "found=", len(uniq))
    return uniq

def next_trading_day(d):
    # Mon=0 ... Fri=4
    return d + timedelta(days=3) if d.weekday() == 4 else d + timedelta(days=1)


# =======================
# AI report
# =======================
def get_ai_report(model, earnings: List[str], winners: List[Dict], losers: List[Dict], target_date) -> str:
    """
    Earnings timing (pre-market vs after-hours) is not reliably available from yfinance.
    Therefore, we list all companies reporting earnings today without assigning a time label.
    """
    earnings_today = earnings[:]

    prompt = f"""
你是專業美股分析師，負責撰寫給投資研究主管的每日美股盤後晚報。
現在時間為美東時間下午 5:00(收盤後)，請根據以下已提供的結構化資料，生成一份專業、精煉、格式完全一致的盤後晚報。

已提供資料：
明日預計公布財報公司：{earnings_today}
盤後大幅波動公司（盤後強勢股）：{winners}
盤後大幅波動公司（盤後疲弱股）：{losers}

引用規則：
盤後強勢股/疲弱股段落中，「參考資料連結」必須逐行列出該檔股票 news 欄位中的 url（若 url 為空或 news 為空，則不輸出）；不得自行編造或補上任何網址。

語言與風格要求：
使用繁體中文（台灣用語）
專業、美股研究員口吻
條列式、重點導向
僅輸出內容本身，不要任何前言或結語，輸出格式為 HTML（不含 <html> 或 <body> 標籤）
HTML 使用規範：
僅可使用以下 HTML 標籤：
<h1>、<h2>、<h3>、<b>、<br>
不得使用 style、class、font、span、div、css
不得在句子中插入 HTML 標籤

結構與格式要求（必須完全一致，不可調整順序或名稱）：
<h1>{now_est().strftime('%Y-%m-%d')} 美股盤後市場報告"</h1>
<h2>🗓️下交易日預計公布財報公司</h2>
公司英文全部名稱 + (Ticker)<br> #不要連續兩個ticker
公司英文全部名稱 + (Ticker)<br> #不要連續兩個ticker
公司英文全部名稱 + (Ticker)<br> #不要連續兩個ticker
（請遵守英文公司全名先，再空格鍵(Ticker)的格式。依輸入資料數量持續列出；若無資料直接寫「無」，若僅有1或2支則僅列出實際公司數量）
<br>

<h2>⚠️盤後大幅波動公司分析</h2>
<h3>📈盤後強勢股</h3>
<b>1. Ticker： +X%</b><br>
兩至三句話說明主要原因(不分段)<br>
（參考資料，不用括號跟寫參考資料連結等字樣，直接貼連結就好；不超過一篇）<br>
<br>
<b>2. Ticker： +X%</b><br>
兩至三句話說明主要原因(不分段)<br>
（參考資料，不用括號跟寫參考資料連結等字樣，直接貼連結就好；不超過一篇）<br>
<br>
<b>3. Ticker： +X%</b><br>
兩至三句話說明主要原因(不分段)<br>
（參考資料，不用括號跟寫參考資料連結等字樣，直接貼連結就好；不超過一篇）<br>
（若無資料直接寫「無」，若僅有1或2支則僅列出實際公司數量）
<br>

<h3>📉盤後疲弱股</h3>
<b>1. Ticker： -X%</b><br>
兩至三句話說明主要原因(不分段)<br>
（參考資料，不用括號跟寫參考資料連結等字樣，直接貼連結就好；不超過一篇）<br>
<br>

<b>2. Ticker： -X%</b><br>
兩至三句話說明主要原因(不分段)<br>
（參考資料，不用括號跟寫參考資料連結等字樣，直接貼連結就好；不超過一篇）<br>
<br>

<b>3. Ticker： -X%</b><br>
兩至三句話說明主要原因(不分段)<br>
（參考資料，不用括號跟寫參考資料連結等字樣，直接貼連結就好；不超過一篇）<br>
（若無資料直接寫「無」，若僅有1或2支則僅列出實際公司數量）
<br>


<h2>🔍市場概覽與關注焦點</h2>
<b>1. 新聞一標題(中文)</b><br>
2 至 3 句說明，盡量聚焦總體經濟、經濟數據、貨幣政策、利率或地緣政治，不可與其他新聞產生連結(不分段)<br>
（參考資料，不用寫參考資料連結等字樣；不超過一篇）<br>
<br>
<b>2. 新聞二標題(中文)</b><br>
2 至 3 句說明，內容需與新聞一完全無關(不分段)<br>
（參考資料，不用寫參考資料連結等字樣；不超過一篇）<br>
<br>
<b>3. 新聞三標題(中文)</b><br>
2 至 3 句說明，內容需與前兩則完全無關(不分段)<br>
（參考資料，不用寫參考資料連結等字樣；不超過一篇）<br>
<br>
<br>

補充規則：
市場概覽與關注焦點以總體為優先，若資料不足，才可補充與盤後波動相關的個股事件，盡量不要重複盤後強勢股/疲弱股的內容
不提供投資建議、不使用情緒性或判斷性語言
若某一欄位資料不足，仍需保留該欄位與編號，不可刪除段落
只輸出最終晚報內容，不要解釋、不自我評論。
中文標點符號使用全形，包括，、。：；
"""

    try:
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return sanitize_text_for_email(text)
    except Exception as e:
        ascii_log("AI_REPORT_FAILED", repr(e))
        fallback = (
            "明日預計公布財報公司\n"
            "無\n"
            "盤後大幅波動公司分析\n"
            "盤後強勢股\n"
            "1. 無：+ 0%\n"
            "2. 無：+ 0%\n"
            "3. 無：+ 0%\n"
            "盤後疲弱股\n"
            "1. 無：- 0%\n"
            "2. 無：- 0%\n"
            "3. 無：- 0%\n"
            "市場概覽與關注焦點\n"
            "無\n"
            "無\n"
            "無\n"
        )

        return sanitize_text_for_email(fallback)


# =======================
# Email
# =======================
def send_email(body: str) -> bool:
    try:
        subject_text = f"【美股盤後日報】{now_est().strftime('%Y-%m-%d')} 市場報告"

        msg = EmailMessage(policy=policy.SMTP)
        msg["Subject"] = subject_text
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER

        safe_body = sanitize_text_for_email(body)

        msg.set_content(
            "This email contains an HTML report. Please view in an HTML-capable email client.",
            charset="utf-8"
        )
        
        msg.add_alternative(safe_body, subtype="html", charset="utf-8")        

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg, mail_options=["SMTPUTF8"])

        return True
    except Exception as e:
        ascii_log("EMAIL_SEND_FAILED", repr(e))
        return False


# =======================
# Main
# =======================
def main():
    t0 = time.time()
    # Hard checks
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY env var.")
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        raise RuntimeError("Missing EMAIL_* env vars (sender/password/receiver).")

    # Guard (important for GitHub cron UTC vs ET & DST)
#    if not should_run_now():
#        ascii_log("SKIP: Not 17:00 ET weekday. Now(ET)=", now_est().isoformat())
#        return

    ascii_log("Job started (ET):", now_est().isoformat())

    # 1) tickers
    sp500 = fetch_sp500_tickers()
    top_set = set(sp500[:TOP_SET_N])

    # 2) movers (limited scan for speed)
    winners, losers = fetch_movers(sp500, top_set, max_scan=MAX_SCAN)

    # 3) earnings today (best effort, also limited)
    target_date = next_trading_day(now_est().date())
    earnings_next_day = fetch_earnings_best_effort(
        sp500,
        target_date=target_date,
        max_scan=min(1000, MAX_SCAN)
    )

    # 4) gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model_name = get_available_model()
    ascii_log("Using model:", model_name)
    model = genai.GenerativeModel(model_name)

    ascii_log("Generating report...")
    report = get_ai_report(model, earnings_next_day, winners, losers, target_date)

    # 5) send
    ok = send_email(report)
    if ok:
        ascii_log("EMAIL_SENT_OK")
    else:
        ascii_log("EMAIL_SENT_FAIL")

    ascii_log("TOTAL_SECONDS", round(time.time() - t0, 2))

if __name__ == "__main__":
    main()
