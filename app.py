from flask import Flask, render_template, request, redirect, url_for, flash
from pathlib import Path
import pandas as pd
import os
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
from io import BytesIO
import statsmodels.api as sm
import statsmodels.formula.api as smf
import threading


# Flask 설정
app = Flask(__name__)
app.secret_key = 'your-secret-key'

BASE = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = "uploads"
RESULT_PATH = os.path.join(UPLOAD_FOLDER, "result.csv")
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

processing_done = False
result_df = None

# 전처리 함수 임포트
from data_processing import run_preprocessing
from collections import Counter

# KPI 계산 함수
def calculate_kpis(df):
    df['produced_qty'] = pd.to_numeric(df['produced_qty'], errors='coerce')
    df['defect_qty'] = pd.to_numeric(df['defect_qty'], errors='coerce')
    df['electricity_kwh'] = pd.to_numeric(df['electricity_kwh'], errors='coerce')
    df['gas_nm3'] = pd.to_numeric(df['gas_nm3'], errors='coerce')

    total_production = df['produced_qty'].sum()
    total_defect = df['defect_qty'].sum()
    total_energy = df['electricity_kwh'].sum() + df['gas_nm3'].sum()
    defect_rate = total_defect / (total_production + 1e-5)

    return {
        'defect_rate': f"{defect_rate * 100:.1f}%",
        'production_qty': int(total_production),
        'energy_usage': f"{total_energy:.0f} kWh"
    }

# 그래프 함수 (생략 없이 그대로 유지)
def get_production_trend(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["produced_qty"] = pd.to_numeric(df["produced_qty"], errors="coerce").fillna(0)
    daily_avg = df.groupby(["date", "line_id"])["produced_qty"].mean().reset_index()
    fig = px.line(daily_avg, x="date", y="produced_qty", color="line_id",
                  title="Production Trend (Daily Avg)")
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

def get_defect_rate_distribution(df):
    df["produced_qty"] = pd.to_numeric(df["produced_qty"], errors="coerce").fillna(0)
    df["defect_qty"] = pd.to_numeric(df["defect_qty"], errors="coerce").fillna(0)
    df = df[df["produced_qty"] > 0].copy()
    df["defect_rate"] = df["defect_qty"] / df["produced_qty"]
    df = df[df["defect_rate"] > 0]
    fig = px.box(
        df,
        x="line_id",
        y="defect_rate",
        title="Defect Rate Distribution (Non-zero Only)",
        category_orders={"line_id": sorted(df["line_id"].dropna().unique())},
        points="outliers"
    )
    fig.update_layout(yaxis_range=[0, 0.03])
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def get_energy_usage_chart(df):
    df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"], errors="coerce").fillna(0)
    df["gas_nm3"] = pd.to_numeric(df["gas_nm3"], errors="coerce").fillna(0)
    energy_avg = df.groupby("line_id")[["electricity_kwh", "gas_nm3"]].mean().reset_index()
    fig = px.bar(energy_avg, x="line_id", y=["electricity_kwh", "gas_nm3"],
                 barmode="group", title="Energy Usage by Line")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

def spc_chart(series, name):
    mean = series.mean()
    std = series.std()
    ucl = mean + 3 * std
    lcl = mean - 3 * std
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series.index, series.values, label=name)
    ax.axhline(mean, color='green', linestyle='--', label='Mean')
    ax.axhline(ucl, color='red', linestyle='--', label='UCL')
    ax.axhline(lcl, color='red', linestyle='--', label='LCL')
    ax.set_title(f'SPC Chart - {name}')
    ax.legend()
    ax.grid(True)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def spc_chart_by_line(df, y_col='electricity_kwh'):
    df_summary = df.groupby(['date', 'line_id'])[y_col].mean().reset_index()
    stats = df_summary.groupby('line_id')[y_col].agg(['mean', 'std']).reset_index()
    stats['ucl'] = stats['mean'] + 3 * stats['std']
    stats['lcl'] = stats['mean'] - 3 * stats['std']

    fig = go.Figure()
    for line_name in df_summary['line_id'].unique():
        df_line = df_summary[df_summary['line_id'] == line_name]
        stat_line = stats[stats['line_id'] == line_name].iloc[0]
        fig.add_trace(go.Scatter(x=df_line['date'], y=df_line[y_col],
                                 mode='lines+markers', name=f'{line_name}'))
        fig.add_trace(go.Scatter(x=df_line['date'], y=[stat_line['mean']] * len(df_line),
                                 mode='lines', name=f'{line_name} Mean',
                                 line=dict(dash='dash', color='green'), showlegend=False))
        fig.add_trace(go.Scatter(x=df_line['date'], y=[stat_line['ucl']] * len(df_line),
                                 mode='lines', name=f'{line_name} UCL',
                                 line=dict(dash='dot', color='red'), showlegend=False))
        fig.add_trace(go.Scatter(x=df_line['date'], y=[stat_line['lcl']] * len(df_line),
                                 mode='lines', name=f'{line_name} LCL',
                                 line=dict(dash='dot', color='red'), showlegend=False))
    fig.update_layout(
        title=f"SPC Chart - {y_col} (by line)",
        xaxis_title="Date",
        yaxis_title=y_col,
        height=500
    )
    return fig.to_html(full_html=False)

def factory_energy_anova(df):
    results = {}

    anova_df = df[['factory_id', 'electricity_kwh', 'gas_nm3']].dropna()

    # 전력
    model_elec = smf.ols('electricity_kwh ~ C(factory_id)', data=anova_df).fit()
    anova_elec = sm.stats.anova_lm(model_elec, typ=2)
    pval_elec = anova_elec.loc['C(factory_id)', 'PR(>F)']
    results["elec"] = {
        "anova_table": anova_elec.to_html(classes="table table-bordered"),
        "p_value": pval_elec,
        "is_significant": pval_elec < 0.05
    }

    # 가스
    model_gas = smf.ols('gas_nm3 ~ C(factory_id)', data=anova_df).fit()
    anova_gas = sm.stats.anova_lm(model_gas, typ=2)
    pval_gas = anova_gas.loc['C(factory_id)', 'PR(>F)']
    results["gas"] = {
        "anova_table": anova_gas.to_html(classes="table table-bordered"),
        "p_value": pval_gas,
        "is_significant": pval_gas < 0.05
    }
    return results

def get_top_remark_issues(df, top_n=5):
    keywords = ['감지', '누락', '의심', '시급', '권장', '필요', '초과', '요구']
    df = df.copy()
    df['remark'] = df['remark'].fillna('')
    df['remark_reason'] = df['remark'].apply(lambda x: [kw for kw in x if kw in x])
    df['has_issue'] = df['remark_reason'].apply(lambda x: len(x) > 0)

    # ✅ 날짜 + remark 조합이 유일한 데이터로부터
    # 날짜 순서대로 한 날짜당 대표 1건씩 뽑고, 상위 5개 추출
    top_rows = (
        df[df['has_issue']]
        .drop_duplicates(subset=['date', 'remark'])   # 중복 제거
        .sort_values('date')                          # 오래된 날짜부터
        .groupby('date')
        .head(1)                                      # 각 날짜당 1건
        .sort_values('date', ascending=False)         # 최신순으로 정렬
        .head(top_n)[['date', 'remark']]              # 상위 n개만
        .to_dict(orient='records')
    )
    return top_rows

def generate_energy_trend_split_charts(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    daily_avg = df.groupby('date')[['electricity_kwh', 'gas_nm3']].mean()

    fig_elec = px.line(
        daily_avg,
        x=daily_avg.index,
        y='electricity_kwh',
        title='Daily Electricity Consumption Trend',
        labels={'electricity_kwh': 'Electricity (kWh)', 'date': 'Date'}
    )
    fig_elec.update_traces(line=dict(color='blue'))

    fig_gas = px.line(
        daily_avg,
        x=daily_avg.index,
        y='gas_nm3',
        title='Daily Gas Consumption Trend',
        labels={'gas_nm3': 'Gas (Nm³)', 'date': 'Date'}
    )
    fig_gas.update_traces(line=dict(color='orange'))

    energy_trend_elec = plot(fig_elec, output_type='div', include_plotlyjs=False)
    energy_trend_gas = plot(fig_gas, output_type='div', include_plotlyjs=False)

    return energy_trend_elec, energy_trend_gas

def generate_factory_line_bar_charts_plotly(df):
    summary = df.groupby(['factory_id', 'line_id'])[['produced_qty', 'defect_qty']].mean().reset_index()

    fig_produced = px.bar(
        summary,
        x="factory_id",
        y="produced_qty",
        color="line_id",
        barmode="group",
        title="Produced Quantity per Factory & Line",
        labels={"factory_id": "Factory", "produced_qty": "Produced Qty", "line_id": "Line ID"}
    )
    produced_chart_html = fig_produced.to_html(full_html=False)

    defect_fig = px.bar(
        summary,
        x="factory_id",
        y="defect_qty",
        color="line_id",
        barmode="group",
        title="Defect Quantity per Factory & Line",
        labels={"factory_id": "Factory", "defect_qty": "Defect Qty", "line_id": "Line ID"}
    )
    defect_chart_html = defect_fig.to_html(full_html=False)

    return produced_chart_html, defect_chart_html



def run_analysis(folder_path):
    global processing_done, result_df
    try:
        df = run_preprocessing(Path(folder_path))  # 폴더 내 모든 csv 전처리
        df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
        result_df = df
        processing_done = True
        print("✅ 분석 완료")
    except Exception as e:
        print("❌ 분석 중 오류:", e)
        processing_done = False





@app.route("/", methods=["GET", "POST"])
def index():
    global processing_done, result_df

    if request.method == "POST":
        files = request.files.getlist("files")  # ✅ 다중 파일 받아오기

        # 업로드된 파일 저장
        for file in files:
            if file and file.filename.endswith(".csv"):
                save_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(save_path)
                print(f"✔ 저장됨: {file.filename}")

        # ✅ 동기 실행으로 변경
        processing_done = False
        result_df = None
        run_analysis(UPLOAD_FOLDER)

    # ✅ 결과가 완료되었는지 확인
    show_result = processing_done and os.path.exists(RESULT_PATH)
    result_preview = result_df.head(10).to_html(classes="table") if show_result else None

    # 📊 기존 result.csv 로드 및 분석
    df = None
    kpis = {"defect_rate": "-", "production_qty": "-", "energy_usage": "-"}
    production_html = defect_html = energy_html = None
    spc_elec_img = spc_gas_img = None
    spc_by_line_html = spc_by_line_gas_html = None
    anova_results = None

    remark_top5 = []
    energy_trend_elec = ""
    energy_trend_gas = ""
    produced_chart_html = ""
    defect_chart_html = ""


    if os.path.exists(RESULT_PATH):
        try:
            print("📂 result.csv 존재 확인됨")
            df = pd.read_csv(RESULT_PATH)
            print("✅ CSV 로딩 성공")
            kpis = calculate_kpis(df)

            production_html = get_production_trend(df)
            defect_html = get_defect_rate_distribution(df)
            energy_html = get_energy_usage_chart(df)

            df['date'] = pd.to_datetime(df['date'], errors="coerce")
            elec_series = df.set_index("date")["electricity_kwh"].dropna()
            gas_series = df.set_index("date")["gas_nm3"].dropna()
            spc_elec_img = spc_chart(elec_series, "electricity_kwh")
            spc_gas_img = spc_chart(gas_series, "gas_nm3")
            spc_by_line_html = spc_chart_by_line(df, y_col='electricity_kwh')
            spc_by_line_gas_html = spc_chart_by_line(df, y_col='gas_nm3')
            anova_results = factory_energy_anova(df)
            remark_top5 = get_top_remark_issues(df)
            energy_trend_elec, energy_trend_gas = generate_energy_trend_split_charts(df)
            produced_chart_html, defect_chart_html = generate_factory_line_bar_charts_plotly(df)


        except Exception as e:
            print("❌ CSV 로드 또는 그래프 생성 오류:", e)
            flash("❌ CSV 불러오기 또는 그래프 생성 중 오류 발생")

    return render_template(
        "index.html",
        show_result=show_result,
        result_preview=result_preview,
        table=df.head().to_html(index=False) if df is not None else None,
        data=df.head(5).to_csv(index=False) if df is not None else None,
        kpis=kpis,
        production_chart=production_html,
        defect_chart=defect_html,
        energy_chart=energy_html,
        spc_elec=spc_elec_img,
        spc_gas=spc_gas_img,
        spc_by_line=spc_by_line_html,
        spc_by_line_gas=spc_by_line_gas_html,
        anova_results=anova_results,
        remark_top5=remark_top5,
        energy_trend_elec=energy_trend_elec,
        energy_trend_gas=energy_trend_gas,
        produced_chart=produced_chart_html,
        defect_bar_chart=defect_chart_html,

    )


# @app.route("/upload", methods=["POST"])
# def upload():
#     files = request.files.getlist("files")
#     if not files or files[0].filename == "":
#         flash("❌ 파일을 업로드해주세요.")
#         return redirect(url_for("index"))
#     for f in files:
#         if f and f.filename.endswith(".csv"):
#             save_path = os.path.join(UPLOAD_FOLDER, f.filename)
#             f.save(save_path)
#             print(f"✔️ 저장됨: {f.filename}")
#     try:
#         df = run_preprocessing(Path(UPLOAD_FOLDER))
#         df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
#         print("✅ result.csv 저장 완료")
#     except Exception as e:
#         print(f"❌ 전처리 오류: {e}")
#         flash(f"❌ 전처리 중 오류 발생: {str(e)}")
#         return redirect(url_for("index"))
#     return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render 환경변수 사용
    # app.run(host="0.0.0.0", port=port) 










