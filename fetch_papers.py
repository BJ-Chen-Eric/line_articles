import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import os
import json
from openai import OpenAI

# ==========================================
# 1. 配置區域 (Configuration)
# ==========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER_ID = os.getenv("LINE_USER_ID") # 這是你的個人 USER ID 或群組 ID


if OPENROUTER_API_KEY:
    # OpenRouter 的標準相容端點
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free" # 或者換成 "openai/gpt-oss-120b"
else:
    print("Warning: OPENROUTER_API_KEY not found.")
    client = None

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
HISTORY_FILE = "processed_papers.log"

# 定義研究領域：整合「應用」與「多效性」
FIELDS = {
    "Field 1: Biomarkers in Host-Virus ScRNA-seq": {
        "query": '(flavivirus OR "dengue virus" OR ZIKV) AND ("single-cell" OR scRNA-seq) AND (biomarker OR "therapeutic target" OR "clinical translation")',
        "keywords": ["flavivirus", "dengue", "ZIKV", "scRNA-seq", "biomarker", "therapeutic target", "clinical"]
    },
    "Field 2: Viral Evolution & Phylodynamics": {
        "query": '("avian influenza" OR AIV OR H5N1) AND (phylodynamic OR evolution OR spillover) AND (modeling OR computational OR "machine learning")',
        "keywords": ["avian influenza", "H5N1", "evolution", "phylodynamics", "spillover", "computational", "modeling"]
    },
    "Field 3: Complex Traits Pleiotropy & V2F": {
        "query": '("complex trait" OR "cardiometabolic") AND ("V2F" OR "Variant-to-Function" OR "fine-mapping") AND (pleiotropy OR "shared genetics")',
        "keywords": ["complex trait", "V2F", "pleiotropy", "cross-phenotype", "fine-mapping"]
    },
    "Field 4: Bayesian Methods in Genomics": {
        "query": '(Bayesian OR MCMC OR "Gaussian process") AND (genomics OR "single-cell" OR bioinformatics)',
        "keywords": ["Bayesian", "MCMC", "Gaussian process", "genomics", "bioinformatics"]
    }
}

MAX_RESULTS_PER_SOURCE = 3

# ==========================================
# 2. 抓取引擎 (Fetch Engines)
# ==========================================

def fetch_arxiv(query, max_results=3):
    base_url = 'http://export.arxiv.org/api/query?'
    encoded_query = urllib.parse.quote(query)
    url = f'{base_url}search_query=all:{encoded_query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}'
    try:
        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')
        root = ET.fromstring(content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        return [{'title': e.find('atom:title', ns).text.strip(), 'summary': e.find('atom:summary', ns).text.strip(), 'link': e.find('atom:id', ns).text, 'source': 'arXiv'} for e in root.findall('atom:entry', ns)]
    except: return []

def fetch_pubmed(query, max_results=3):
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded_query}&retmax={max_results}&sort=pub+date"
    try:
        with urllib.request.urlopen(search_url) as resp:
            ids = [id_elem.text for id_elem in ET.fromstring(resp.read()).findall(".//Id")]
        if not ids: return []
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(ids)}"
        with urllib.request.urlopen(fetch_url) as resp:
            summary_tree = ET.fromstring(resp.read())
        return [{'title': d.find("./Item[@Name='Title']").text, 'summary': "Details on PubMed.", 'link': f"https://pubmed.ncbi.nlm.nih.gov/{d.find('Id').text}/", 'source': 'PubMed'} for d in summary_tree.findall(".//DocSum")]
    except: return []

def fetch_biorxiv(keywords, max_results=3):
    start_date = (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d")
    url = f"https://api.biorxiv.org/details/biorxiv/{start_date}/{datetime.now().strftime('%Y-%m-%d')}/0/json"
    try:
        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        filtered = []
        for p in data.get('collection', []):
            if any(k.lower() in (p['title'] + p['abstract']).lower() for k in keywords):
                filtered.append({'title': p['title'], 'summary': p['abstract'], 'link': f"https://www.biorxiv.org/content/{p['doi']}", 'source': 'bioRxiv'})
                if len(filtered) >= max_results: break
        return filtered
    except: return []

# ==========================================
# 3. 核心執行邏輯 (Main Logic)
# ==========================================

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f: return set(line.strip() for line in f)
    return set()

def save_to_history(paper_link):
    with open(HISTORY_FILE, "a") as f: f.write(paper_link + "\n")


def get_ai_summary(field_name, papers):
    if not client or not papers: return "No data."
    
    paper_text = "".join([f"[{p['source']}] {p['title']}\n{p['summary'][:300]}...\n\n" for p in papers])
    
    prompt = f"""
    You are Eric's Senior Bioinformatics Advisor. Eric bridges wet and dry labs, focusing on precision health, biomarkers, and computational models for viral evolution.
    Analyze these papers for the field: {field_name}.
    Papers: {paper_text}

    Provide a structured, concise response:
    1. **Key Methodological Trend (1 sentence):** What is the core computational or wet-lab technique being utilized?
    2. **Translational Value (1 sentence):** What specific biomarker or downstream application is identified?
    3. **Critical Limitation (1 sentence):** Point out a potential flaw in the single-cell resolution, modeling assumption, or experimental design.
    """
    
    try:
        # 🚨 調用 OpenRouter 的 ChatCompletion 語法
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ],
            # OpenRouter 建議加入以下 Headers 讓平台統計（選填，不加也能跑）
            extra_headers={
                "HTTP-Referer": "https://localhost", 
                "X-Title": "Eric Bioinformatics Bot",
            },
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e: 
        return f"Summary generation failed. Error: {e}"
    

def send_to_slack(text):
    if not SLACK_WEBHOOK_URL: return
    payload = {"text": f"🧬 *Eric's Multi-Omics Update* ({datetime.now().strftime('%Y-%m-%d')})\n\n{text[:3000]}"}
    req = urllib.request.Request(SLACK_WEBHOOK_URL, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try: urllib.request.urlopen(req)
    except Exception as e: print(f"Slack Error: {e}")

def send_to_line(text):
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        print("Warning: LINE configuration missing.")
        return
        
    url = "https://api.line.me/v2/bot/message/push"
    
    # 🚨 LINE 訊息限制單則最大 5000 字，這裡取 3000 字非常安全
    # 注意：LINE 不支援 Slack 的 Markdown 語法（如 *粗體*），它會直接顯示符號
    payload = {
        "to": LINE_USER_ID,
        "messages": [
            {
                "type": "text",
                "text": f"🧬 Eric's Multi-Omics Update ({datetime.now().strftime('%Y-%m-%d')})\n\n{text[:3000]}"
            }
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            print("[✓] 已成功發送更新至 LINE！")
    except Exception as e:
        print(f"LINE Error: {e}")


def main():
    date_str = datetime.now().strftime('%Y-%m-%d')
    os.makedirs("briefings", exist_ok=True)
    filename = os.path.join("briefings", f"research_briefing_{date_str}.md")
    
    history = load_history()
    report_content = f"# Daily Research Briefing - {date_str}\n\n"
    slack_summary = ""
    any_new = False

    for field, config in FIELDS.items():
        print(f"==========================================")
        print(f"Fetching {field}...")
        
        # 🚨 拆開抓取，加上數量診斷
        arxiv_p = fetch_arxiv(config['query'])
        pubmed_p = fetch_pubmed(config['query'])
        biorxiv_p = fetch_biorxiv(config['keywords'])
        
        all_p = arxiv_p + pubmed_p + biorxiv_p
        print(f"  [➔] 原始抓取總數: {len(all_p)} 篇 (arXiv: {len(arxiv_p)}, PubMed: {len(pubmed_p)}, bioRxiv: {len(biorxiv_p)})")
        
        new_p = [p for p in all_p if p['link'] not in history]
        print(f"  [➔] 過濾歷史紀錄後的新論文: {len(new_p)} 篇")
        
        if new_p:
            any_new = True
            report_content += f"## {field}\n"
            to_analyze = new_p[:3]
            ai_text = get_ai_summary(field, to_analyze)
            report_content += f"### 💡 AI Analysis\n{ai_text}\n\n"
            slack_summary += f"📍 *{field}*\n{ai_text}\n\n"
            for p in to_analyze:
                report_content += f"- **[{p['source']}]** [{p['title']}]({p['link']})\n"
                save_to_history(p['link'])
            report_content += "\n---\n"

    print(f"==========================================")
    with open(filename, 'w', encoding='utf-8') as f: f.write(report_content)
    if any_new: 
        # 🚨 記得把原本的 send_to_slack 改成 send_to_line
        send_to_line(slack_summary)

    # with open(filename, 'w', encoding='utf-8') as f: f.write(report_content)
    # if any_new: 
    #     send_to_slack(slack_summary)
    #     print("[✓] 已發送更新至 Slack。")
    else:
        print("[!] 本次無新論文，未發送 Slack。")

if __name__ == "__main__":
    main()