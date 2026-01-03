import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter, argrelextrema
import datetime
import time
from concurrent.futures import ThreadPoolExecutor

# --- 1. 深度 UI 与布局配置 (彻底解决白边与截断) ---
st.set_page_config(page_title="RS 极速实时分析系统 v27.0", layout="wide")
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1.5rem;
        padding-right: 2.5rem; /* 强制右侧留白，防止图例截断 */
        max-width: 100%;
    }
    [data-testid="stSidebar"] { width: 300px !important; }
    </style>
    """, unsafe_allow_html=True)


def set_global_style():
    # 多级字体回退机制，确保中文不乱码
    fonts = ['Microsoft YaHei', 'SimHei', 'STHeiti', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False


set_global_style()


# --- 2. 极速数据引擎 (并行下载 & 持久化) ---

@st.cache_data(ttl=86400, persist="disk")
def fetch_sw_index(symbol):
    try:
        df = ak.index_hist_sw(symbol=symbol, period="day")
        df = df.rename(columns={'日期': 'date', '收盘': 'idx_c'})
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'idx_c']]
    except:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_mkt_index(m_code):
    try:
        df = ak.stock_zh_index_daily(symbol=m_code)
        df = df.rename(columns={'date': 'date', 'close': 'idx_c'})
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'idx_c']]
    except:
        return pd.DataFrame()


def fetch_stock_smart(code, start, end):
    try:
        df = ak.stock_zh_a_hist(symbol=code, start_date=start, end_date=end, adjust="hfq")
        df = df.rename(columns={'日期': 'date', '收盘': 'C', '成交量': 'V'})
        df['date'] = pd.to_datetime(df['date'])
        # 实时补全逻辑
        today_str = datetime.datetime.now().strftime("%Y%m%d")
        if end >= today_str:
            try:
                spot = ak.stock_individual_info_em(symbol=code)
                curr_c = float(spot[spot['item'] == '最新价']['value'].values[0])
                curr_v = float(spot[spot['item'] == '成交量']['value'].values[0])
                if df['date'].max().strftime("%Y%m%d") < today_str:
                    df = pd.concat([df, pd.DataFrame([{'date': pd.to_datetime(today_str), 'C': curr_c, 'V': curr_v}])],
                                   ignore_index=True)
            except:
                pass
        return df
    except:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_real_name(code):
    """三级获取保障"""
    try:
        df = ak.stock_individual_info_em(symbol=code)
        name = df[df['item'] == '股票名称']['value'].values[0]
        if name and name != "Is null": return name
    except:
        pass
    try:
        spot = ak.stock_zh_a_spot_em()
        return spot[spot['代码'] == code]['名称'].values[0]
    except:
        return f"个股({code})"


# --- 3. 核心计算 ---
def perform_calculation(df_s, df_i, df_m):
    data = df_s.merge(df_i, on='date').merge(df_m, on='date')
    data = data.rename(columns={'idx_c_x': 'I', 'idx_c_y': 'M'}).sort_values('date').reset_index(drop=True)
    if data.empty: return None, None, None, None, None, None

    p0 = data.iloc[0]
    data['RS_I'] = (data['C'] / p0['C']) - (data['I'] / p0['I'])
    data['RS_M'] = (data['C'] / p0['C']) - (data['M'] / p0['M'])
    data['MA5'] = data['C'].rolling(5).mean()

    win = 15 if len(data) > 30 else (5 if len(data) > 10 else 3)
    data['S_RI'] = savgol_filter(data['RS_I'].values, win, 3)
    data['S_RM'] = savgol_filter(data['RS_M'].values, win, 3)
    data['S_MA'] = savgol_filter(data['MA5'].ffill().bfill().values, win, 3)
    data['S_P'] = savgol_filter(data['C'].values, win, 3)

    # 拐点提取逻辑
    def get_ex(v):
        idx = np.sort(np.unique(np.concatenate([argrelextrema(v, np.greater)[0], argrelextrema(v, np.less)[0]])))
        return idx

    ext_i, ext_m, ext_ma = get_ex(data['S_RI'].values), get_ex(data['S_RM'].values), get_ex(data['S_MA'].values)
    b_mask = ((data['C'] > data['MA5']) & (data['C'].shift(1) <= data['MA5'].shift(1))).fillna(False).values
    v_mask = ((data['V'] > data['V'].rolling(5).mean() * 1.8) & (data['C'] > data['C'].shift(1))).fillna(False).values
    return data, ext_i, ext_m, ext_ma, b_mask, v_mask


# --- 4. 侧边栏与流程控制 ---
st.sidebar.title("🚀 RS 极速决策引擎")
s_code = st.sidebar.text_input("1. 个股代码", value="002530")
i_code = st.sidebar.text_input("2. 行业代码(申万)", value="801074")
today_d = datetime.date.today()
st_d = st.sidebar.date_input("3. 开始日期", today_d - datetime.timedelta(days=150))
ed_d = st.sidebar.date_input("4. 结束日期", today_d)
run_btn = st.sidebar.button("开始实时诊断")

if run_btn:
    p_bar = st.progress(0)
    status = st.empty()
    try:
        status.text("正在建立并发网络连接...")
        p_bar.progress(20)
        s_str, e_str = st_d.strftime("%Y%m%d"), ed_d.strftime("%Y%m%d")
        m_code = "sh000001" if s_code.startswith(('60', '68')) else "sz399001"

        with ThreadPoolExecutor(max_workers=3) as pool:
            f_s = pool.submit(fetch_stock_smart, s_code, s_str, e_str)
            f_i = pool.submit(fetch_sw_index, i_code)
            f_m = pool.submit(fetch_mkt_index, m_code)
            df_stock, df_ind, df_mkt = f_s.result(), f_i.result(), f_m.result()

        stock_name = get_real_name(s_code)
        p_bar.progress(60)

        status.text(f"正在分析 {stock_name} 的共振信号...")
        data, ext_i, ext_m, ext_ma, b_mask, v_mask = perform_calculation(df_stock, df_ind, df_mkt)
        if data is None: raise ValueError("无法合并数据，请检查日期或代码")

        p_bar.progress(85)
        status.text("正在构建可视化终端...")

        # 建议逻辑
        last = data.iloc[-1]
        slope_i = last['S_RI'] - data.iloc[-2]['S_RI']
        is_above = last['C'] > last['MA5']
        if slope_i > 0 and is_above:
            adv, bg_c = "【强势看多】共振反转", "#cf1322"
        elif slope_i > 0 or is_above:
            adv, bg_c = "【谨慎看多】形态修复", "#f39c12"
        else:
            adv, bg_c = "【避险观望】趋势走弱", "#27ae60"

        # --- 5. 旗舰绘图布局 ---
        fig = plt.figure(figsize=(22, 11), facecolor='white')
        # 增加主图比例，缩窄侧边，解决留白问题
        gs = gridspec.GridSpec(2, 2, height_ratios=[3.3, 1], width_ratios=[5.6, 1.2], hspace=0.18, wspace=0.06)
        ax1, ax3 = plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0], sharex=plt.subplot(gs[0, 0]))
        ax2 = ax1.twinx()
        x = np.arange(len(data))

        ax1.plot(x, data['S_RI'], color='#1890ff', lw=3.3, label='行业强度')
        ax1.plot(x, data['S_RM'], color='#fa8c16', lw=2.2, ls='--', alpha=0.75, label='大盘强度')
        ax1.axhline(0, color='#ff4d4f', lw=1.2, ls='--', alpha=0.5)
        ax2.plot(x, data['S_MA'], color='#722ed1', lw=2.8, alpha=0.85)
        ax2.plot(x, data['S_P'], color='#52c41a', lw=1.5, alpha=0.12)

        # 拐点与突破星号 (修复双线拐点不全)
        if len(ext_i) > 0: ax1.scatter(ext_i, data['S_RI'].iloc[ext_i], color='#262626', s=60, zorder=10)
        if len(ext_m) > 0: ax1.scatter(ext_m, data['S_RM'].iloc[ext_m], color='#262626', s=35, alpha=0.6, zorder=9)
        if len(ext_ma) > 0: ax2.scatter(ext_ma, data['S_MA'].iloc[ext_ma], color='#722ed1', marker='d', s=100,
                                        facecolor='none', lw=1.5)

        b_idx = np.where(b_mask)[0]
        if len(b_idx) > 0: ax2.scatter(b_idx, data['C'].iloc[b_idx], color='#fadb14', marker='*', s=400,
                                       edgecolors='#333', lw=1, zorder=11)

        # 成交量与异动标记 (1.8倍原则)
        v_colors = ['#f5222d' if data['C'].iloc[i] >= (data['C'].iloc[i - 1] if i > 0 else 0) else '#52c41a' for i in
                    range(len(data))]
        ax3.bar(x, data['V'], color=v_colors, alpha=0.65, width=0.85)
        v_sig_idx = np.where(v_mask)[0]
        if len(v_sig_idx) > 0:
            ax3.scatter(v_sig_idx, data['V'].iloc[v_sig_idx] * 1.15, color='#cf1322', marker='v', s=150, zorder=10)

        # --- 6. 右侧信息卡片 (解决截断与重叠) ---
        ax_info = plt.subplot(gs[:, 1])
        ax_info.axis('off')
        ax_info.text(0.05, 0.96, f"最新建议 ({last['date'].strftime('%m-%d')}):", fontsize=11.5, fontweight='bold',
                     transform=ax_info.transAxes)
        # 高亮决策框
        ax_info.add_patch(Rectangle((0.05, 0.89), 0.92, 0.065, color=bg_c, alpha=0.9, transform=ax_info.transAxes))
        ax_info.text(0.51, 0.922, adv, color='white', fontsize=11.5, fontweight='bold', ha='center',
                     transform=ax_info.transAxes)

        y_ptr = 0.84
        ax_info.text(0.05, y_ptr, "--- [ 图表记法定义 ] ---", fontsize=10.5, color='#8c8c8c',
                     transform=ax_info.transAxes)

        items = [
            ('line', '#1890ff', '对行业强度', '-'), ('line', '#fa8c16', '对大盘强度', '--'),
            ('line', '#722ed1', '5日拟合均线', '-'), ('box', '#52c41a', '股价背景趋势', '-'),
            ('dot', '#262626', '趋势拐点(斜率0)', 'o'), ('diamond', '#722ed1', '均线拐点(转折)', 'd'),
            ('star', '#fadb14', '确认买入信号', '*'), ('tri', '#cf1322', '异动放量(>1.8)', 'v')
        ]

        y_step = 0.068
        for i, (itype, col, label, style) in enumerate(items):
            yy = y_ptr - 0.075 - i * y_step
            if itype == 'line':
                ax_info.plot([0.1, 0.28], [yy, yy], color=col, lw=2.5, ls=style, transform=ax_info.transAxes)
            elif itype == 'box':
                ax_info.add_patch(
                    Rectangle((0.1, yy - 0.005), 0.18, 0.012, facecolor=col, alpha=0.35, transform=ax_info.transAxes))
            elif itype == 'dot':
                ax_info.scatter(0.19, yy, color=col, s=70, transform=ax_info.transAxes)
            elif itype == 'diamond':
                ax_info.scatter(0.19, yy, color=col, marker='d', s=80, facecolor='none', lw=1.5,
                                transform=ax_info.transAxes)
            elif itype == 'star':
                ax_info.scatter(0.19, yy, color=col, marker='*', s=160, edgecolors='#333', transform=ax_info.transAxes)
            elif itype == 'tri':
                ax_info.scatter(0.19, yy, color=col, marker='v', s=110, transform=ax_info.transAxes)
            ax_info.text(0.4, yy, label, fontsize=10, va='center', transform=ax_info.transAxes)

        # 战法口诀
        ax_info.add_patch(
            Rectangle((0.05, 0.05), 0.92, 0.13, facecolor='#f8f9fa', edgecolor='#dee2e6', transform=ax_info.transAxes))
        ax_info.text(0.1, 0.12, "💡 盈利模型：\n寻找蓝线拐点点向上，\n且伴随金色星号共振。", fontsize=10, color='#495057',
                     transform=ax_info.transAxes)

        # --- 7. 标题动态化与坐标精修 ---
        plt.suptitle(f"{stock_name} ({s_code}) 交易决策分析系统", fontsize=22, fontweight='bold', y=0.985)

        tick_pos = np.linspace(0, len(data) - 1, 10, dtype=int)
        ax3.set_xticks(tick_pos)
        ax3.set_xticklabels(data['date'].dt.strftime('%m-%d').iloc[tick_pos], rotation=0, fontsize=11)
        ax1.grid(True, alpha=0.15, linestyle=':')

        # 强制解决截断问题的核心
        plt.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.08)

        p_bar.progress(100)
        status.text(f"诊断完成：{stock_name} (最新价: {last['C']:.2f})")
        st.pyplot(fig, use_container_width=True)

    except Exception as e:
        st.error(f"分析失败: {e}")
        p_bar.empty()
        status.empty()

else:
    # 引导界面
    st.info("👋 **欢迎进入 v27.0 极速决策系统**")
    st.write("请在左侧配置代码，点击 [开始实时诊断] 即可获取全屏级专业决策图表。")
