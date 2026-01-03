import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter, argrelextrema
import datetime
from concurrent.futures import ThreadPoolExecutor

# --- 1. 页面配置：宽屏模式 & 减少边距 ---
st.set_page_config(page_title="RS 交易决策分析系统", layout="wide")
st.markdown("""
    <style>
    .main .block-container {padding-top: 1rem; padding-bottom: 1rem; padding-left: 2rem; padding-right: 2rem;}
    </style>
    """, unsafe_allow_html=True)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 2. 并发数据引擎 ---
@st.cache_data(ttl=3600)
def fetch_index_data(symbol, is_sw=True):
    try:
        if is_sw:
            df = ak.index_hist_sw(symbol=symbol, period="day")
            df = df.rename(columns={'日期': 'date', '收盘': 'idx_c'})
        else:
            df = ak.stock_zh_index_daily(symbol=symbol)
            df = df.rename(columns={'close': 'idx_c'})
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'idx_c']]
    except:
        return pd.DataFrame()


def fetch_stock_hist(code, start, end):
    try:
        df = ak.stock_zh_a_hist(symbol=code, start_date=start, end_date=end, adjust="hfq")
        df = df.rename(columns={'日期': 'date', '收盘': 'C', '成交量': 'V'})
        df['date'] = pd.to_datetime(df['date'])
        return df[['date', 'C', 'V']]
    except:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_real_stock_name(code):
    """多级保障获取股票真实名称"""
    try:
        # 第一路径：单股基础信息
        df = ak.stock_individual_info_em(symbol=code)
        return df[df['item'] == '股票名称']['value'].values[0]
    except:
        try:
            # 第二路径：全A股快照过滤（仅在第一路径失败时调用）
            spot = ak.stock_zh_a_spot_em()
            name = spot[spot['代码'] == code]['名称'].values[0]
            return name
        except:
            return "个股"


def get_market_code(s):
    if s.startswith(('60', '68')):
        return "sh000001"
    elif s.startswith(('00', '30')):
        return "sz399001"
    return "sz899050" if s.startswith(('8', '4')) else "sh000001"


# --- 3. 侧边栏交互 ---
st.sidebar.title("🚀 极速决策引擎")
stock_code = st.sidebar.text_input("1. 个股代码", value="002530")
ind_code = st.sidebar.text_input("2. 行业代码(申万)", value="801074")
today = datetime.date.today()
start_date = st.sidebar.date_input("3. 开始日期", today - datetime.timedelta(days=150))
end_date = st.sidebar.date_input("4. 结束日期", today)

run_button = st.sidebar.button("开始实时诊断")

# --- 4. 主逻辑控制 ---
if run_button:
    try:
        with st.spinner('⚡ 并发引擎正在加速下载数据...'):
            s_str, e_str = start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
            m_code = get_market_code(stock_code)

            with ThreadPoolExecutor(max_workers=3) as executor:
                f1 = executor.submit(fetch_stock_hist, stock_code, s_str, e_str)
                f2 = executor.submit(fetch_index_data, ind_code, True)
                f3 = executor.submit(fetch_index_data, m_code, False)
                df_stock, df_ind, df_mkt = f1.result(), f2.result(), f3.result()

            stock_name = get_real_stock_name(stock_code)
            if df_stock.empty or df_ind.empty:
                st.error("数据调取失败，请检查参数。")
                st.stop()

            data = df_stock.merge(df_ind, on='date').merge(df_mkt, on='date')
            data = data.rename(columns={'idx_c_x': 'I', 'idx_c_y': 'M'}).sort_values('date').reset_index(drop=True)

            # --- 指标计算 ---
            p0 = data.iloc[0]
            data['RS_I'] = (data['C'] / p0['C']) - (data['I'] / p0['I'])
            data['RS_M'] = (data['C'] / p0['C']) - (data['M'] / p0['M'])
            data['MA5'] = data['C'].rolling(5).mean()
            win = 15 if len(data) > 15 else 5
            data['S_RI'] = savgol_filter(data['RS_I'].values, win, 3)
            data['S_RM'] = savgol_filter(data['RS_M'].values, win, 3)
            data['S_MA'] = savgol_filter(data['MA5'].ffill().bfill().values, win, 3)
            data['S_P'] = savgol_filter(data['C'].values, win, 3)


            # 拐点提取
            def get_ex(v):
                return np.sort(
                    np.unique(np.concatenate([argrelextrema(v, np.greater)[0], argrelextrema(v, np.less)[0]])))


            ext_i, ext_m, ext_ma = get_ex(data['S_RI'].values), get_ex(data['S_RM'].values), get_ex(data['S_MA'].values)

            break_mask = ((data['C'] > data['MA5']) & (data['C'].shift(1) <= data['MA5'].shift(1))).fillna(False).values
            vol_mask = ((data['V'] > data['V'].rolling(5).mean() * 1.8) & (data['C'] > data['C'].shift(1))).fillna(
                False).values

            # 建议逻辑
            last, prev = data.iloc[-1], data.iloc[-2]
            slope_i = last['S_RI'] - prev['S_RI']
            is_above = last['C'] > last['MA5']
            if slope_i > 0 and is_above:
                advice, bg_c = "【强势看多】共振反转", "#cf1322"
            elif slope_i > 0 or is_above:
                advice, bg_c = "【谨慎看多】形态修复", "#f39c12"
            else:
                advice, bg_c = "【避险观望】趋势走弱", "#27ae60"

            # --- 5. 绘图：增大画布比例 & 补全拐点 ---
            fig = plt.figure(figsize=(18, 10), facecolor='white')
            # 增大 width_ratios 使得图表区域更广
            gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[4.8, 1.2], hspace=0.15, wspace=0.08)
            ax1, ax3 = plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0], sharex=plt.subplot(gs[0, 0]))
            ax2 = ax1.twinx()
            x = np.arange(len(data))

            ax1.plot(x, data['S_RI'], color='#1890ff', lw=3, label='行业强度')
            ax1.plot(x, data['S_RM'], color='#fa8c16', lw=2, ls='--', alpha=0.7)  # 大盘线
            ax1.axhline(0, color='#ff4d4f', lw=1.2, ls='--', alpha=0.5)
            ax2.plot(x, data['S_MA'], color='#722ed1', lw=2.5, alpha=0.8)
            ax2.plot(x, data['S_P'], color='#52c41a', lw=1.5, alpha=0.1)

            # 标注双线斜率为0的拐点
            ax1.scatter(ext_i, data['S_RI'].iloc[ext_i], color='#262626', s=50, zorder=10)  # 行业线黑点
            ax1.scatter(ext_m, data['S_RM'].iloc[ext_m], color='#262626', s=35, alpha=0.5, zorder=9)  # 大盘线黑点
            ax2.scatter(ext_ma, data['S_MA'].iloc[ext_ma], color='#722ed1', marker='d', s=100, facecolor='none', lw=1.5)

            b_idx = np.where(break_mask)[0]
            ax2.scatter(b_idx, data['C'].iloc[b_idx], color='#fadb14', marker='*', s=350, edgecolors='#333', zorder=11)

            # 成交量
            v_cols = ['#f5222d' if data['C'].iloc[i] >= (data['C'].iloc[i - 1] if i > 0 else 0) else '#52c41a' for i in
                      range(len(data))]
            ax3.bar(x, data['V'], color=v_cols, alpha=0.6)
            v_sig_idx = np.where(vol_mask)[0]
            if len(v_sig_idx) > 0:
                ax3.scatter(v_sig_idx, data['V'].iloc[v_sig_idx] * 1.15, color='#cf1322', marker='v', s=120)

            # --- 6. 右侧精修图例字典 ---
            ax_info = plt.subplot(gs[:, 1])
            ax_info.axis('off')
            ax_info.text(0.05, 0.96, f"最新决策 ({last['date'].strftime('%m-%d')}):", fontsize=11, fontweight='bold',
                         transform=ax_info.transAxes)
            ax_info.add_patch(Rectangle((0.05, 0.89), 0.9, 0.06, color=bg_c, alpha=0.8, transform=ax_info.transAxes))
            ax_info.text(0.5, 0.92, advice, color='white', fontsize=11, fontweight='bold', ha='center',
                         transform=ax_info.transAxes)

            y_ptr = 0.84
            ax_info.text(0.05, y_ptr, "--- [ 标记定义字典 ] ---", fontsize=10, color='#8c8c8c',
                         transform=ax_info.transAxes)
            items = [
                ('line', '#1890ff', '对行业强度 (RS-Ind)', '-'), ('line', '#fa8c16', '对大盘强度 (RS-Mkt)', '--'),
                ('line', '#722ed1', '5日拟合均线 (S-MA5)', '-'), ('box', '#52c41a', '股价平滑趋势 (S-Price)', '-'),
                ('dot', '#262626', '趋势拐点 (斜率为0)', 'o'), ('diamond', '#722ed1', '均线拐点 (方向转折)', 'd'),
                ('star', '#fadb14', '突破信号 (站上MA5)', '*'), ('tri', '#cf1322', '异常放量 (机构异动)', 'v'),
                ('line', '#ff4d4f', '零轴基准线', '--')
            ]
            y_step = 0.058
            for i, (itype, col, label, style) in enumerate(items):
                yy = y_ptr - 0.06 - i * y_step
                if itype == 'line':
                    ax_info.plot([0.08, 0.25], [yy, yy], color=col, lw=2, ls=style, transform=ax_info.transAxes)
                elif itype == 'box':
                    ax_info.add_patch(Rectangle((0.08, yy - 0.005), 0.17, 0.01, facecolor=col, alpha=0.3,
                                                transform=ax_info.transAxes))
                elif itype == 'dot':
                    ax_info.scatter(0.165, yy, color=col, s=60, transform=ax_info.transAxes)
                elif itype == 'diamond':
                    ax_info.scatter(0.165, yy, color=col, marker='d', s=60, facecolor='none', lw=1.2,
                                    transform=ax_info.transAxes)
                elif itype == 'star':
                    ax_info.scatter(0.165, yy, color=col, marker='*', s=120, edgecolors='#333',
                                    transform=ax_info.transAxes)
                elif itype == 'tri':
                    ax_info.scatter(0.165, yy, color=col, marker='v', s=80, transform=ax_info.transAxes)
                ax_info.text(0.32, yy, label, fontsize=9.5, va='center', transform=ax_info.transAxes)

            ax_info.add_patch(Rectangle((0.05, 0.05), 0.9, 0.13, facecolor='#f9f9f9', edgecolor='#d9d9d9',
                                        transform=ax_info.transAxes))
            ax_info.text(0.1, 0.12, "💡 实战逻辑提示：\n关注蓝/橙双线同时向上\n且伴随【金色星号】突破。", fontsize=9.5,
                         color='#555555', transform=ax_info.transAxes)

            plt.suptitle(f"{stock_name} ({stock_code}) 交易决策分析系统", fontsize=18, fontweight='bold', y=0.98)

            # X轴日期精修
            tick_pos = np.linspace(0, len(data) - 1, 10, dtype=int)
            ax3.set_xticks(tick_pos)
            ax3.set_xticklabels(data['date'].dt.strftime('%m-%d').iloc[tick_pos], rotation=0)
            ax1.grid(True, alpha=0.15, linestyle=':')

            # 极大化展示
            st.pyplot(fig, use_container_width=True)
            st.success(f"诊断完成：{stock_name} 数据已同步更新。")

    except Exception as e:
        st.error(f"分析失败: {e}")

else:
    # --- 初始状态显示 ---
    st.markdown("---")
    st.info("👋 **欢迎进入 RS 极速交易决策系统**")
    st.write("请在左侧边栏配置股票代码、行业代码及日期，然后点击 **[开始实时诊断]** 获取分析图表。")
    st.markdown("---")