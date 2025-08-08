import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============== Config ==================
st.set_page_config(page_title="Asset Allocation com Fronteira Eficiente", layout="wide")
st.title("📁 Asset Allocation com Fronteira Eficiente")
st.caption("Build: v7 — Otimização + Pesos Manuais + Benchmark + Sugestão de Perfil (não bloqueia)")

CAMINHO_PLANILHA_ATIVOS = r"C:/Users/AmorimPedro/Downloads/asset allocation/ativos.xlsx"
CAMINHO_CLASSIFICACAO = r"C:/Users/AmorimPedro/Downloads/asset allocation/classificacao_ativos.xlsx"
CAMINHO_BENCHMARK = r"C:/Users/AmorimPedro/Downloads/asset allocation/benchmark.xlsx"

# =============== Funções ==================
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

def class_constraints(weights, tickers, perfil, restricoes_por_perfil, classe_ativos):
    restricoes = restricoes_por_perfil[perfil]
    classes = pd.Series([classe_ativos.get(t, "Outros") for t in tickers])
    cons = []
    for classe, (min_v, max_v) in restricoes.items():
        idxs = [i for i, c in enumerate(classes) if c == classe]
        if idxs:
            cons.append({'type': 'ineq', 'fun': lambda x, idxs=idxs, mv=min_v: np.sum(x[idxs]) - mv})
            cons.append({'type': 'ineq', 'fun': lambda x, idxs=idxs, mv=max_v: mv - np.sum(x[idxs])})
    return cons

def minimize_volatility_with_constraints(mean_returns, cov_matrix, target_return, tickers, perfil, restricoes_por_perfil, classe_ativos):
    num_assets = len(tickers)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return},
    ]
    constraints += class_constraints(np.ones(num_assets), tickers, perfil, restricoes_por_perfil, classe_ativos)
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = [1. / num_assets] * num_assets
    result = minimize(lambda x: portfolio_performance(x, mean_returns, cov_matrix)[1],
                      init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError("Otimização falhou para o retorno alvo informado.")
    return result.x

def gerar_fronteira_eficiente(mean_returns, cov_matrix, tickers, perfil, restricoes_por_perfil, classe_ativos):
    target_returns = np.linspace(0.01, 0.15, 50)
    vols, rets = [], []
    for r in target_returns:
        try:
            w = minimize_volatility_with_constraints(mean_returns, cov_matrix, r, tickers, perfil, restricoes_por_perfil, classe_ativos)
            ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
            rets.append(ret); vols.append(vol)
        except Exception:
            continue
    return np.array(rets), np.array(vols)

def indicadores_totais(portfolio_returns):
    ann_return = (1 + portfolio_returns.mean())**252 - 1
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    cum = (1 + portfolio_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    return ann_return, ann_vol, sharpe, max_dd

def indicadores_por_ano(portfolio_returns):
    df = portfolio_returns.to_frame("ret")
    df["ano"] = df.index.year.astype(str)
    out = []
    for ano, g in df.groupby("ano"):
        r = (1 + g["ret"]).prod() - 1
        vol = g["ret"].std() * np.sqrt(252)
        sharpe = r / vol if vol != 0 else np.nan
        cum = (1 + g["ret"]).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min()
        out.append([ano, f"{r*100:.2f}%", f"{vol*100:.2f}%", f"{sharpe:.2f}", f"{max_dd*100:.2f}%"])
    return pd.DataFrame(out, columns=["Ano", "Retorno", "Volatilidade", "Sharpe", "Máx. Drawdown"])

def render_dashboard(pesos, titulo, returns, mean_returns, cov_matrix, df, classe_ativos,
                     benchmark_retornos=None, benchmark_nome=None, mostrar_fronteira=False,
                     frontier_vols=None, frontier_returns=None):
    # tabela
    result_df = pd.DataFrame({
        "Ativo": df.columns,
        "Peso (%)": np.round(pesos * 100, 2),
        "Classe": [classe_ativos.get(t, "Outros") for t in df.columns]
    })
    st.subheader(titulo)
    st.dataframe(result_df[result_df["Peso (%)"] > 0], use_container_width=True)

    portfolio_returns = (returns * pesos).sum(axis=1)

    col1, col2, col3 = st.columns(3)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(3.2, 3.2))
        dados = result_df[result_df["Peso (%)"] > 0]
        if not dados.empty:
            ax1.pie(dados["Peso (%)"], labels=dados["Ativo"], autopct='%1.1f%%')
        ax1.set_title("Gráfico de Pizza")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 3.2))
        (1 + portfolio_returns).cumprod().plot(ax=ax2, color="blue", linewidth=2, label="Carteira")
        if (benchmark_retornos is not None) and (not benchmark_retornos.empty):
            port_alinh, bench_alinh = portfolio_returns.align(benchmark_retornos, join="inner")
            (1 + bench_alinh).cumprod().plot(ax=ax2, color="gray", linestyle="--", linewidth=1.5,
                                             label=(benchmark_nome or "Benchmark"))
        ax2.set_title("Evolução do Portfólio")
        ax2.legend()
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(5, 3.2))
        if mostrar_fronteira and (frontier_vols is not None) and (frontier_returns is not None):
            ax3.plot(frontier_vols, frontier_returns, 'b--', linewidth=1.2)
            ret_otimo, vol_otimo = portfolio_performance(pesos, mean_returns, cov_matrix)
            ax3.scatter(vol_otimo, ret_otimo, color='red', label='Ponto atual')
            ax3.set_title("Fronteira Eficiente")
            ax3.set_xlabel("Volatilidade")
            ax3.set_ylabel("Retorno")
            ax3.legend()
        else:
            ret, vol = portfolio_performance(pesos, mean_returns, cov_matrix)
            ax3.axis("off")
            ax3.text(0.0, 0.8, "Resumo da Carteira", fontsize=12, weight="bold")
            ax3.text(0.0, 0.6, f"Retorno Esperado: {ret*100:.2f}%", fontsize=10)
            ax3.text(0.0, 0.5, f"Vol Anualizada: {vol*100:.2f}%", fontsize=10)
        st.pyplot(fig3)

    st.subheader("📊 Indicadores de Performance Totais")
    ann_return, ann_vol, sharpe, max_dd = indicadores_totais(portfolio_returns)
    st.table({
        "Retorno Anualizado": f"{ann_return*100:.2f}%",
        "Volatilidade Anualizada": f"{ann_vol*100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Máximo Drawdown": f"{max_dd*100:.2f}%",
    })

    st.subheader("📆 Indicadores por Ano")
    st.dataframe(indicadores_por_ano(portfolio_returns), use_container_width=True)

def share_por_classe(pesos, tickers, classe_ativos):
    classes = pd.Series([classe_ativos.get(t, "Outros") for t in tickers])
    df_aux = pd.DataFrame({"classe": classes, "peso": pesos})
    return df_aux.groupby("classe")["peso"].sum().to_dict()

def sugerir_perfil(pesos, tickers, classe_ativos, limites):
    shares = share_por_classe(pesos, tickers, classe_ativos)
    # função auxiliar para verificar se atende ao perfil
    def atende(perfil):
        for classe, (mn, mx) in limites[perfil].items():
            s = shares.get(classe, 0.0)
            if s < mn - 1e-9 or s > mx + 1e-9:
                return False
        return True
    # retorna o primeiro perfil que atende (Conservador -> Moderado -> Agressivo)
    for p in ["Conservador", "Moderado", "Agressivo"]:
        if atende(p):
            return p
    return "Agressivo"

# =============== Sidebar ==================
st.sidebar.header("Configurações")
criterio = st.sidebar.radio("⚙️ Critério de alocação", ["Retorno alvo", "Máx. Drawdown"])
perfil = st.sidebar.selectbox("Perfil do investidor:", ["Conservador", "Moderado", "Agressivo"])
if criterio == "Retorno alvo":
    retorno_alvo = st.sidebar.slider("🎯 Retorno alvo anual (%)", 2.0, 12.0, 6.0, 0.1) / 100
    max_dd_user = None
else:
    max_dd_user = st.sidebar.slider("📉 Máx. Drawdown permitido (%)", 1.0, 50.0, 20.0, 0.5) / 100
    retorno_alvo = None

limites_demo = {
    "Conservador": {"Ações": (0, 0.2), "Commodities": (0, 0.1), "Renda Fixa": (0.7, 1.0)},
    "Moderado": {"Ações": (0.2, 0.5), "Commodities": (0, 0.2), "Renda Fixa": (0.4, 0.8)},
    "Agressivo": {"Ações": (0.4, 1.0), "Commodities": (0, 0.3), "Renda Fixa": (0, 0.5)},
}

st.sidebar.markdown("### Limites por Classe")
for classe, (mn, mx) in limites_demo[perfil].items():
    st.sidebar.write(f"**{classe}**: {mn*100:.0f}% – {mx*100:.0f}%")

# =============== Pipeline principal ==================
try:
    # dados
    df = pd.read_excel(CAMINHO_PLANILHA_ATIVOS)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index().ffill()
    returns = df.pct_change().dropna()

    # classes
    df_classes = pd.read_excel(CAMINHO_CLASSIFICACAO)
    classe_ativos = dict(zip(df_classes["Ativo"], df_classes["Classe"]))
    restricoes_por_perfil = limites_demo

    # stats
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # otimização
    if criterio == "Retorno alvo":
        pesos_otimizados = minimize_volatility_with_constraints(mean_returns, cov_matrix, retorno_alvo, df.columns, perfil, restricoes_por_perfil, classe_ativos)
    else:
        def objetivo(x):
            port_ret = (returns * x).sum(axis=1)
            _, _, _, max_dd = indicadores_totais(port_ret)
            if max_dd < -max_dd_user:
                return -np.dot(x, mean_returns)
            return 1e6

        num_assets = len(df.columns)
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = [1. / num_assets] * num_assets
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        constraints += class_constraints(np.ones(num_assets), df.columns, perfil, restricoes_por_perfil, classe_ativos)

        result = minimize(objetivo, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError("Não foi possível encontrar uma carteira com o drawdown desejado.")
        pesos_otimizados = result.x

    # returns carteira otimizada
    portfolio_returns_otm = (returns * pesos_otimizados).sum(axis=1)

    # benchmark
    benchmark_escolhido = None
    benchmark_retornos = None
    if os.path.exists(CAMINHO_BENCHMARK):
        df_bench = pd.read_excel(CAMINHO_BENCHMARK)
        df_bench.columns = df_bench.columns.astype(str).str.strip()
        if "Date" in df_bench.columns:
            df_bench["Date"] = pd.to_datetime(df_bench["Date"], dayfirst=True, errors="coerce")
            df_bench = df_bench.dropna(subset=["Date"]).sort_values("Date").set_index("Date").ffill()
            series_opts = [c for c in df_bench.columns if pd.api.types.is_numeric_dtype(df_bench[c])]
            if series_opts:
                benchmark_escolhido = st.selectbox("Benchmark de comparação", series_opts, index=0)
                bench_series = df_bench[benchmark_escolhido].pct_change().dropna()
                # alinha com a carteira otimizada para o gráfico padrão da aba Otimização
                _, benchmark_retornos = portfolio_returns_otm.align(bench_series, join="inner")
        # se não houver Date, não mostra nada

    # fronteira para aba otimização
    ret_otm, vol_otm = portfolio_performance(pesos_otimizados, mean_returns, cov_matrix)
    frontier_returns, frontier_vols = gerar_fronteira_eficiente(mean_returns, cov_matrix, df.columns, perfil, restricoes_por_perfil, classe_ativos)

    # ====== ABAS ======
    tab_otm, tab_manual = st.tabs(["Otimização", "Pesos manuais"])

    with tab_otm:
        if retorno_alvo is not None:
            titulo = f"📊 Alocação Ótima - Perfil {perfil} ({retorno_alvo*100:.1f}%)"
        else:
            titulo = f"📊 Alocação Ótima - Perfil {perfil} (Modo: Máx. Drawdown)"
        render_dashboard(
            pesos=pesos_otimizados,
            titulo=titulo,
            returns=returns,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            df=df,
            classe_ativos=classe_ativos,
            benchmark_retornos=benchmark_retornos,
            benchmark_nome=benchmark_escolhido,
            mostrar_fronteira=True,
            frontier_vols=frontier_vols,
            frontier_returns=frontier_returns
        )

        # Comparação enxuta (Carteira x Benchmark)
        if (benchmark_retornos is not None) and (not benchmark_retornos.empty):
            st.subheader("📊 Comparação com Benchmark")
            # alinhar séries para métricas
            port_alinh, bench_alinh = portfolio_returns_otm.align(benchmark_retornos, join="inner")
            def metrics(s):
                r = (1 + s.mean())**252 - 1
                v = s.std() * np.sqrt(252)
                sh = r / v if v != 0 else np.nan
                cum = (1 + s).cumprod(); peak = cum.cummax(); dd = (cum - peak)/peak; mdd = dd.min()
                return r, v, sh, mdd
            r_p, v_p, sh_p, dd_p = metrics(port_alinh)
            r_b, v_b, sh_b, dd_b = metrics(bench_alinh)
            comp_df = pd.DataFrame({
                "": ["Carteira", "Benchmark"],
                "Retorno (a.a.)": [f"{r_p*100:.2f}%", f"{r_b*100:.2f}%"],
                "Vol. (a.a.)": [f"{v_p*100:.2f}%", f"{v_b*100:.2f}%"],
                "Sharpe (rf=0)": [f"{sh_p:.2f}", f"{sh_b:.2f}"],
                "Máx DD": [f"{dd_p*100:.2f}%", f"{dd_b*100:.2f}%"],
            })
            st.dataframe(comp_df, use_container_width=True)

    with tab_manual:
        st.subheader("✍️ Pesos manuais (em %)")
        df_pesos = pd.DataFrame({"Ativo": df.columns, "Peso (%)": np.zeros(len(df.columns))})
        if st.checkbox("Iniciar com pesos iguais (opcional)"):
            df_pesos["Peso (%)"] = np.round(100 / len(df.columns), 2)

        edited = st.data_editor(
            df_pesos,
            num_rows="fixed",
            use_container_width=True,
            column_config={"Peso (%)": st.column_config.NumberColumn(step=0.1, min_value=0.0, max_value=100.0)}
        )

        soma = edited["Peso (%)"].sum()
        colA, colB = st.columns([1,1])
        with colA:
            st.write(f"**Soma atual:** {soma:.2f}%")
        with colB:
            normalizar = st.checkbox("Normalizar automaticamente para 100%", value=True)

        if st.button("Aplicar pesos manuais"):
            pesos_man = edited["Peso (%)"].to_numpy(dtype=float) / 100.0
            if normalizar and pesos_man.sum() > 0:
                pesos_man = pesos_man / pesos_man.sum()
            elif not np.isclose(pesos_man.sum(), 1.0):
                st.error("A soma dos pesos precisa ser 100% (ou marque Normalizar).")
                st.stop()

            # Sugestão de perfil (não bloqueia)
            perfil_sugerido = sugerir_perfil(pesos_man, df.columns, classe_ativos, limites_demo)
            if perfil_sugerido != perfil:
                st.info(f"🔎 Pela distribuição por classe, esta carteira se encaixa melhor no perfil **{perfil_sugerido}** (perfil atual selecionado: **{perfil}**).")

            # Benchmark para comparação com a carteira manual (realinha com a série manual)
            bench_para_manual = None
            bench_nome = benchmark_escolhido
            if (benchmark_retornos is not None) and (not benchmark_retornos.empty):
                # recalcula os retornos da carteira manual e alinha com benchmark
                port_man = (returns * pesos_man).sum(axis=1)
                _, bench_para_manual = port_man.align(benchmark_retornos, join="inner")

            render_dashboard(
                pesos=pesos_man,
                titulo="📊 Carteira com Pesos Manuais",
                returns=returns,
                mean_returns=mean_returns,
                cov_matrix=cov_matrix,
                df=df,
                classe_ativos=classe_ativos,
                benchmark_retornos=bench_para_manual,
                benchmark_nome=bench_nome,
                mostrar_fronteira=False
            )

except Exception as e:
    st.error(f"❌ Erro ao processar: {e}")
