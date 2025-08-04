import pandas as pd
from pathlib import Path
from thefuzz import process, fuzz
from tqdm import tqdm
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import random
import os
from dotenv import load_dotenv
import openai

# 🔑 OpenAI API 키 (선택적으로 입력)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

VALID_NAMES = ['김민수', '박지훈', '이지우', '정해인', '최유리']

def ask_chatgpt(name: str) -> str:
    if not hasattr(openai, 'api_key') or not openai.api_key:
        return name
    try:
        prompt = (
            f"'{name}'이라는 이름에 오탈자가 있을 수 있습니다. "
            f"다음 목록 {VALID_NAMES} 중에서 가장 유사한 이름을 찾아주세요. "
            f"유사한 이름이 없다면 '{name}'을 그대로 반환해주세요. "
            f"답변은 이름 하나만 반환해야 합니다."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return name

def correct_operator(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return name
    name = name.strip()
    match, score = process.extractOne(name, VALID_NAMES, scorer=fuzz.ratio)
    if score >= 60:
        return match
    elif score >= 40:
        return ask_chatgpt(name)
    else:
        return name

def refine_status(row):
    remark = str(row.get("remark", ""))
    if any(x in remark for x in ["점검이 필요", "전기 계통의 부하가 높아지는 추세임", "정밀 점검이 필요함"]):
        return "추후 정밀 점검 필요"
    elif any(x in remark for x in ["공정 품질에 영향", "즉각적인 유지보수가 요구됨"]):
        return "이상(교체필요)"
    else:
        return "이상없음"

def assign_inspector_shift(inspector):
    inspector_shift_map = {
        "서지우": "A", "이준서": "B", "정재훈": "C", "홍지민": "A"
    }
    if pd.isna(inspector):
        return random.choice(["A", "B", "C"])
    return inspector_shift_map.get(inspector, random.choice(["A", "B", "C"]))

def safe_merge(df1, df2, on, how="left", name=""):
    keys = [k for k in on if k in df1.columns and k in df2.columns]
    if not keys:
        print(f"⚠️ {name} 병합 건너뜀: 공통 키 없음 → {on}")
        return df1
    try:
        return df1.merge(df2, on=keys, how=how)
    except Exception as e:
        print(f"❌ 병합 실패 ({name}):", e)
        return df1

def extract_keywords_tfidf(texts, top_k=4):
    from kiwipiepy import Kiwi
    from sklearn.feature_extraction.text import TfidfVectorizer

    kiwi = Kiwi()
    stopwords = {
        "의", "이", "가", "은", "는", "들", "좀", "잘", "걍", "과", "도", "를", "으로", "에", "하고", "뿐", "등",
        "있으며", "되어", "수", "있다", "있음", "및", "대한", "때문에", "것", "있고", "있어"
    }

    def extract_nouns(text):
        return [
            word for word, tag, _, _ in kiwi.analyze(text)[0][0]
            if tag.startswith("NN") and word not in stopwords
        ]

    print(f"🔎 형태소 분석 중... 총 {len(texts)}개 문장")

    cache = {}
    noun_texts = []
    for i, text in enumerate(texts):
        if i % 500 == 0:
            print(f"  ✔️ {i}개 완료")
        text = text.strip()
        if text in cache:
            nouns = cache[text]
        else:
            try:
                nouns = extract_nouns(text)
            except:
                nouns = []
            cache[text] = nouns
        noun_texts.append(" ".join(nouns))

    vectorizer = TfidfVectorizer(max_features=300)
    X = vectorizer.fit_transform(noun_texts)

    print("🧠 키워드 추출 시작...")
    keywords_list = []
    for i, row in enumerate(X):
        indices = row.nonzero()[1]
        data = row.data
        if len(indices) == 0:
            keywords_list.append("")
            continue
        sorted_indices = sorted(zip(indices, data), key=lambda x: x[1], reverse=True)[:top_k]
        keywords = [vectorizer.get_feature_names_out()[i] for i, _ in sorted_indices]
        keywords_list.append(", ".join(keywords))
        if i % 5000 == 0:
            print(f"🔹 키워드 추출 {i}개 완료")

    return keywords_list


def run_preprocessing(base_path: Path, openai_key: str = None) -> pd.DataFrame:
    tqdm.pandas()
    if openai_key:
        openai.api_key = openai_key

    # CSV 불러오기
    production      = pd.read_csv(base_path / "production_log.csv")
    product_master  = pd.read_csv(base_path / "product_master.csv")
    shift_schedule  = pd.read_csv(base_path / "shift_schedule.csv")
    energy_usage    = pd.read_csv(base_path / "energy_usage.csv")
    inspection      = pd.read_csv(base_path / "inspection_result.csv")
    equipment_check = pd.read_csv(base_path / "equipment_check.csv")

    # 날짜 컬럼명 통일
    production      = production.rename(columns={"production_date": "date"})
    shift_schedule  = shift_schedule.rename(columns={"work_date": "date"})
    inspection      = inspection.rename(columns={"inspection_date": "date"})
    equipment_check = equipment_check.rename(columns={"check_date": "date"})

    for df in [production, shift_schedule, inspection, equipment_check, energy_usage]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date"], inplace=True)

    # 타입 변환
    production["produced_qty"] = pd.to_numeric(production["produced_qty"].replace("없음", pd.NA), errors="coerce")
    production["defect_qty"] = pd.to_numeric(production["defect_qty"], errors="coerce")

    # 평균 생산량 보간
    avg_map = (
        production[production['defect_qty'].notna()]
        .groupby(['factory_id', 'line_id', 'product_code'])['produced_qty']
        .mean().to_dict()
    )
    def fill_qty(row):
        if pd.isna(row['produced_qty']) and pd.notna(row['defect_qty']):
            key = (row['factory_id'], row['line_id'], row['product_code'])
            return avg_map.get(key, row['produced_qty'])
        return row['produced_qty']
    production["produced_qty"] = production.apply(fill_qty, axis=1)

    # 오탈자 교정
    production["operator"] = production["operator"].progress_apply(correct_operator)
    shift_schedule["operator"] = shift_schedule["operator"].progress_apply(correct_operator)

    # equipment_check 상태 재분류
    equipment_check["status"] = equipment_check.apply(refine_status, axis=1)

    # 검사원 교대조
    inspection["inspector_shift"] = inspection["inspector"].apply(assign_inspector_shift)

    # 병합
    merged = production.copy()
    merged = safe_merge(merged, product_master,  on=["product_code"], name="product_master")
    merged = safe_merge(merged, shift_schedule,  on=["factory_id", "line_id", "date", "operator"], name="shift_schedule")
    merged = safe_merge(merged, energy_usage,    on=["factory_id", "line_id", "date"], name="energy_usage")
    merged = safe_merge(merged, inspection,      on=["product_code", "date"], name="inspection")
    merged = safe_merge(merged, equipment_check, on=["factory_id", "line_id", "date"], name="equipment_check")

    # 컬럼 정리
    merged["shift"] = merged.groupby(["date", "operator"])["shift"].transform(lambda x: x.ffill().bfill())
    merged["produced_qty"] = merged["produced_qty"].fillna(0).astype(int)
    merged["defect_qty"] = merged["defect_qty"].fillna(0).astype(int)

    # 컬럼 추가
    for col in ["equipment_id", "product_name", "category", "spec_weight", "electricity_kwh", "gas_nm3",
                "inspector", "inspector_shift", "result", "status", "remark"]:
        if col not in merged.columns:
            merged[col] = pd.NA

    # 클러스터링
    tfidf = TfidfVectorizer(max_features=300)
    X = tfidf.fit_transform(merged["remark"].fillna("").astype(str))

    if X.shape[0] < 9:
        raise ValueError("클러스터링 샘플 수 부족")

    kmeans = KMeans(n_clusters=9, random_state=42)
    merged["remark_cluster"] = kmeans.fit_predict(X)

    # remark_keywords 생성
    if "remark" in merged.columns:
        print("🧠 remark_keywords 추출 중...")
        merged["remark"] = merged["remark"].fillna("")
        merged["remark_keywords"] = extract_keywords_tfidf(merged["remark"])

    # 저장
    BASE = base_path
    merged.to_csv(BASE / "result.csv", index=False, encoding="utf-8-sig")

    return merged
