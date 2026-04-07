import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
import numpy as np

# Dependências exigidas pela desserialização do modelo:
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pmdarima as pm

# ==========================================
# RE-DECLARAÇÃO DAS CLASSES DO MODELO
# ==========================================
# Funções de suporte caso existam
def _winsorize_series(y_arr, k):
    # Dummy implementation caso falte
    from scipy.stats import iqr
    q75, q25 = np.percentile(y_arr, [75 ,25])
    iqr_val = q75 - q25
    lower, upper = q25 - k*iqr_val, q75 + k*iqr_val
    return np.clip(y_arr, lower, upper), lower, upper

def _select_exog_by_spearman(X_df, y, max_features, min_corr, keep_top_k):
    # Dummy implementation caso falte
    from scipy.stats import spearmanr
    corrs = []
    for col in X_df.columns:
        cr, _ = spearmanr(X_df[col], y)
        corrs.append(abs(cr))
    sel_idx = [i for i, c in enumerate(corrs) if c >= min_corr][:max_features]
    return sel_idx, None

class SARIMAXWrapper(BaseEstimator, RegressorMixin):

    def __init__(self, max_p=3, max_q=3, max_d=2,
                 max_P=2, max_Q=2, max_D=1, m=12,
                 max_exog_features=6, min_corr=0.15,
                 winsorize=True, winsorize_k=2.5,
                 information_criterion='aic', use_exog=True):
        self.max_p = max_p; self.max_q = max_q; self.max_d = max_d
        self.max_P = max_P; self.max_Q = max_Q; self.max_D = max_D
        self.m = m
        self.max_exog_features = max_exog_features
        self.min_corr = min_corr
        self.winsorize = winsorize; self.winsorize_k = winsorize_k
        self.information_criterion = information_criterion
        self.use_exog = use_exog

    def _prepare_exog(self, X, y=None, fit=False):
        if not self.use_exog:
            return None
        if X is None or (hasattr(X, 'shape') and X.shape[1] == 0):
            return None
        X_arr = np.asarray(X, dtype=np.float64)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            sel_idx, _ = _select_exog_by_spearman(
                pd.DataFrame(X_arr), y,
                max_features=self.max_exog_features,
                min_corr=self.min_corr, keep_top_k=1
            )
            if sel_idx is None or len(sel_idx) == 0:
                self.selected_cols_ = None
                return None
            self.selected_cols_ = sel_idx
            X_sel = X_arr[:, sel_idx]
            self.exog_scaler_ = StandardScaler()
            return self.exog_scaler_.fit_transform(X_sel)
        if self.selected_cols_ is None:
            return None
        X_sel = X_arr[:, self.selected_cols_]
        return self.exog_scaler_.transform(X_sel)

    def fit(self, X, y):
        self.selected_cols_ = None
        self.exog_scaler_ = None
        y_arr = np.asarray(y, dtype=np.float64)
        if self.winsorize:
            y_fit, self.lower_, self.upper_ = _winsorize_series(y_arr, k=self.winsorize_k)
        else:
            y_fit = y_arr
            self.lower_, self.upper_ = y_arr.min(), y_arr.max()

        X_exog = self._prepare_exog(X, y=y_fit, fit=True)
        self.model_ = pm.auto_arima(
            y_fit, X=X_exog, seasonal=True, m=self.m,
            max_p=self.max_p, max_q=self.max_q, max_d=self.max_d,
            max_P=self.max_P, max_Q=self.max_Q, max_D=self.max_D,
            start_p=1, start_q=1, start_P=0, start_Q=0,
            d=None, D=None,
            information_criterion=self.information_criterion,
            suppress_warnings=True, error_action='ignore',
            stepwise=True, trace=False, maxiter=200, with_intercept=True,
        )
        self.order_ = self.model_.order
        self.seasonal_order_ = self.model_.seasonal_order
        self.y_train_mean_ = np.mean(y_arr)
        self.y_train_std_ = np.std(y_arr)
        return self

    def predict(self, X):
        X_exog = self._prepare_exog(X, fit=False)
        preds = self.model_.predict(n_periods=len(X), X=X_exog)
        preds = np.asarray(preds, dtype=np.float64)
        clip_low  = self.y_train_mean_ - 4 * self.y_train_std_
        clip_high = self.y_train_mean_ + 4 * self.y_train_std_
        return np.clip(preds, clip_low, clip_high)

    def summary(self):
        if hasattr(self, 'model_'):
            exog_str = f"+{len(self.selected_cols_)}exog" if self.selected_cols_ else "puro"
            return f"SARIMAX{self.order_}x{self.seasonal_order_}({exog_str})"
        return "Not fitted"

class DirectFitResult:
    def __init__(self, estimator):
        self.best_estimator_ = estimator
        self.best_params_ = estimator.get_params()
    def predict(self, X):
        return self.best_estimator_.predict(X)

# ==========================================
# CONFIGURAÇÃO DA PÁGINA E CSS CUSTOMIZADO
# ==========================================
st.set_page_config(
    page_title="Dashboard UFPB - Previsão de Empenho",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para criar os cartões HTML estilo "Clean UI" com fórmulas
def criar_cartao_estatistica(titulo, valor, simbolo, formula, cor_borda):
    return f"""
    <div style="background-color: white; padding: 15px; border-radius: 8px; border-left: 5px solid {cor_borda}; border-top: 1px solid #f1f2f6; border-right: 1px solid #f1f2f6; border-bottom: 1px solid #f1f2f6; box-shadow: 2px 2px 10px rgba(0,0,0,0.03); margin-bottom: 15px; height: 115px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <p style="margin: 0; font-size: 14px; font-weight: 600; color: #000000; text-transform: uppercase; letter-spacing: 0.5px;">{titulo}</p>
            <div style="background-color: #f8f9fa; padding: 2px 8px; border-radius: 4px; color: {cor_borda}; font-family: serif; font-style: italic; font-weight: bold; font-size: 16px;">{simbolo}</div>
        </div>
        <h3 style="margin: 10px 0 5px 0; color: #2C3E50; font-size: 22px;">{valor}</h3>
        <p style="margin: 0; font-size: 11px; color: #000000; font-family: monospace;">{formula}</p>
    </div>
    """

def formata_br(valor):
    if pd.isna(valor): return "-"
    return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# ==========================================
# DIRETÓRIOS E CARREGAMENTO DE DADOS
# ==========================================
BASE_DIR = Path(r"C:\Users\josed\Downloads\dash-tcc")
ARQUIVO_HISTORICO = BASE_DIR / "dados_completos_3.csv"
ARQUIVO_PREVISAO = BASE_DIR / "previsoes_dashboard_2025_v.csv"
ARQUIVO_MODELO = BASE_DIR / "melhor_modelo.joblib"
ARQUIVO_AGRUPADO = BASE_DIR / "dados_empenhados_agrupados_1.csv"

@st.cache_data
def carregar_dados():
    try:
        # Histórico Completo Bruto (Para a Aba Estatística)
        df_raw = pd.read_csv(ARQUIVO_HISTORICO)
        df_raw['ano_mes'] = pd.to_datetime(df_raw['ano_mes'])
        df_hist = df_raw[df_raw['ano_mes'].dt.year < 2025].copy()
        
        # Previsão
        df_prev = pd.read_csv(ARQUIVO_PREVISAO)
        df_prev['ano_mes'] = pd.to_datetime(df_prev['ano_mes'])
        
        # Agrupado por Elemento de Despesa
        df_agrupado = pd.read_csv(ARQUIVO_AGRUPADO)
        df_agrupado['ano_mes_lancamento'] = pd.to_datetime(df_agrupado['ano_mes_lancamento'])
        
        # Juntar tudo para a linha do tempo principal
        df_completo = pd.concat([
            df_hist[['ano_mes', 'valor_empenhado']].assign(Tipo='Realizado'),
            df_prev[['ano_mes', 'realizado']].dropna().rename(columns={'realizado': 'valor_empenhado'}).assign(Tipo='Realizado'),
            df_prev[['ano_mes', 'pred_sarimax']].rename(columns={'pred_sarimax': 'valor_empenhado'}).assign(Tipo='Previsão (SARIMA)')
        ])
        
        # Carregar Artefatos do Modelo
        artefatos = joblib.load(ARQUIVO_MODELO)
        
        return df_raw, df_hist, df_prev, df_completo, df_agrupado, artefatos
    except Exception as e:
        st.error(f"Erro ao carregar os dados. Verifique os caminhos. Erro: {e}")
        st.stop()

df_raw, df_hist, df_prev, df_completo, df_agrupado, artefatos = carregar_dados()

# ==========================================
# SIDEBAR / FILTROS GERAIS
# ==========================================
st.sidebar.image("cdn ufpb logo.png", width=80)
st.sidebar.title("Filtros Temporais")

anos_disponiveis = sorted(df_hist['ano_mes'].dt.year.unique().tolist() + [2025])
ano_selecionado = st.sidebar.selectbox("Selecione o Ano para Análise:", ["Todos"] + anos_disponiveis, index=0)

if ano_selecionado != "Todos":
    meses_disponiveis = df_completo[df_completo['ano_mes'].dt.year == ano_selecionado]['ano_mes'].dt.strftime('%m - %b').unique()
    mes_selecionado = st.sidebar.selectbox("Filtre o Mês:", ["Todos"] + list(meses_disponiveis))
else:
    mes_selecionado = "Todos"

st.sidebar.markdown("---")
st.sidebar.markdown("**Projeto de TCC**\n\n*Previsão Mensal de Valor Empenhado UFPB*")

# ==========================================
# CABEÇALHO DO DASHBOARD
# ==========================================
st.title("Dashboard Analítico e Preditivo Orçamentário - UFPB")

# ==========================================
# ABAS DA APLICAÇÃO
# ==========================================
aba1, aba4, aba5 = st.tabs([
    "Visão Geral", 
    "Explorador de Variáveis",
    "Elementos de Despesa"
])

# ------------------------------------------
# ABA 1: VISÃO GERAL
# ------------------------------------------
with aba1:
    st.subheader("Série Temporal de Despesas Empenhadas Discricionárias (2015 - 2025)")
    
    df_plot = df_completo.copy()
    if ano_selecionado != "Todos":
        df_plot = df_plot[df_plot['ano_mes'].dt.year == ano_selecionado]
            
    fig = px.line(df_plot, x='ano_mes', y='valor_empenhado', color='Tipo',
                  color_discrete_map={
                      'Realizado': '#7289da', # Azul que você pediu (tom de lavanda/azul mais claro)
                      'Previsão (SARIMA)': '#e31a1c' # Vermelho
                  },
                  labels={'ano_mes': 'Data', 'valor_empenhado': 'Valor Empenhado (R$)'},
                  markers=False) # Remover os pontos (marcadores)

    # Deixar a linha da previsão tracejada
    fig.update_traces(patch={"line": {"dash": "dash"}}, selector={"name": "Previsão (SARIMA)"})

    fig.update_layout(hovermode="x unified", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Valores Anuais: Previsto x Realizado")
    
    # Filtra df_prev para o ano selecionado, ou usa tudo se "Todos" (lembrando que df_prev tem dados de 2025 foward/test)
    if ano_selecionado == "Todos":
        df_prev_filtro = df_prev.copy()
        label_ano = "Total"
    else:
        df_prev_filtro = df_prev[df_prev['ano_mes'].dt.year == ano_selecionado].copy()
        label_ano = f"{ano_selecionado}"
        
    if not df_prev_filtro.empty and 'realizado' in df_prev_filtro.columns and 'pred_sarimax' in df_prev_filtro.columns:
        tot_prev = df_prev_filtro['pred_sarimax'].sum()
        tot_real = df_prev_filtro['realizado'].sum()
        
        # Erro numérico (pode haver casos de divisão por zero)
        if tot_real != 0:
            erro_pct = (abs(tot_prev - tot_real) / tot_real) * 100
        else:
            erro_pct = 0.0
            
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        with col_comp1:
            st.markdown(criar_cartao_estatistica(f"Previsto ({label_ano})", f"R$ {formata_br(tot_prev)}", "P", "Valor total projetado", "#8E44AD"), unsafe_allow_html=True)
        with col_comp2:
            st.markdown(criar_cartao_estatistica(f"Realizado ({label_ano})", f"R$ {formata_br(tot_real)}", "R", "Valor total real", "#27AE60"), unsafe_allow_html=True)
        with col_comp3:
            st.markdown(criar_cartao_estatistica(f"Erro Percentual", f"{formata_br(erro_pct)}%", "%", "Diferença % geral", "#F39C12"), unsafe_allow_html=True)
    else:
        st.info(f"Dados consolidados de previsão x realizado não disponíveis para o ano selecionado.")

    st.markdown("---")
    st.markdown("### Métricas Gerais e de Previsão")
    col1, col2, col3 = st.columns(3)
    if ano_selecionado == "Todos":
        gasto_total = df_hist['valor_empenhado'].sum()
        media_mensal = df_hist['valor_empenhado'].mean()
        max_mensal = df_hist['valor_empenhado'].max()
        
        col1.metric("Total Histórico (2015-2024)", f"R$ {formata_br(gasto_total)}")
        col2.metric("Média de Empenho Mensal", f"R$ {formata_br(media_mensal)}")
        col3.metric("Maior Empenho Registrado", f"R$ {formata_br(max_mensal)}")
    
    # Lê as métricas do modelo treinado (joblib) e exibe como cards abaixo
    m = artefatos['metrics']
    mae = m.get('MAE', m.get('mae', 0))
    mape = m.get('sMAPE', m.get('mape', 0))  
    rmse = m.get('RMSE', m.get('rmse', 0))

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.markdown(criar_cartao_estatistica("MAE (Erro Absoluto Médio)", f"R$ {formata_br(mae)}", "M", "Média dos erros", "#3498DB"), unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(criar_cartao_estatistica("MAPE (Erro Percentual)", f"{formata_br(mape)}%", "%", "Status: Aceitável", "#27AE60"), unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(criar_cartao_estatistica("RMSE (Raiz do Erro Quadrático)", f"R$ {formata_br(rmse)}", "R", "Penaliza grandes erros", "#E74C3C"), unsafe_allow_html=True)


# ------------------------------------------
# ABA 4: EXPLORADOR DE VARIÁVEIS
# ------------------------------------------
with aba4:
    st.subheader("Análise Estatística Macro")
    st.markdown("Filtre variáveis socioeconômicas e institucionais para entender os fatos históricos do orçamento.")
    
    col_var, col_start, col_end = st.columns([2, 1, 1])
    min_date_raw = df_raw['ano_mes'].min().to_pydatetime()
    max_date_raw = df_raw['ano_mes'].max().to_pydatetime()
    
    with col_start:
        data_inicio_raw = st.date_input("Início (Macro)", min_value=min_date_raw, max_value=max_date_raw, value=min_date_raw)
    with col_end:
        data_fim_raw = st.date_input("Término (Macro)", min_value=min_date_raw, max_value=max_date_raw, value=max_date_raw)
        
    df_raw_filtrado = df_raw[(df_raw['ano_mes'] >= pd.to_datetime(data_inicio_raw)) & (df_raw['ano_mes'] <= pd.to_datetime(data_fim_raw))].copy()
    
    # Criar coluna formatada
    df_raw_filtrado['mes_ano_str'] = df_raw_filtrado['ano_mes'].dt.strftime('%m/%Y')
    
    colunas_ignoradas = ['ano_mes', 'mes_ano_str']
    colunas_disponiveis = [c for c in df_raw.columns if c not in colunas_ignoradas]
    
    with col_var:
        variaveis_selecionadas = st.multiselect(
            "Selecione as variáveis para análise:",
            options=colunas_disponiveis,
            default=["ipca", "selic_meta_mensal"]
        )
    
    if variaveis_selecionadas:
        st.markdown("---")
        
        fig_vars = px.bar(
            df_raw_filtrado, 
            x='mes_ano_str', 
            y=variaveis_selecionadas, 
            barmode='group',
            labels={'value': '', 'variable': 'Variável', 'mes_ano_str': 'Mês/Ano'}
        )
        
        # Padrão Brasileiro de formatação e rótulo no topo (sem R$ porque IPCA e Selic são %)
        fig_vars.update_traces(textposition='outside', texttemplate='%{y:,.0f}')
        fig_vars.update_yaxes(showticklabels=False, title='')
        
        # Aplica padrão de pontuação brasileiro (",." = decimais com vírgula, milhares com ponto)
        fig_vars.update_layout(
            separators=",.", 
            hovermode="x unified", 
            height=450, 
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
            margin=dict(t=50) 
        )
        
        # Ordem cronológica das categorias
        fig_vars.update_xaxes(type='category', categoryorder='array', categoryarray=df_raw_filtrado['mes_ano_str'])
        
        st.plotly_chart(fig_vars, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        for var in variaveis_selecionadas:
            stats = df_raw_filtrado[var].describe()
            st.markdown(f"<h4 style='color: #2C3E50; border-bottom: 2px solid #f1f2f6; padding-bottom: 5px;'>📌 {var.replace('_', ' ').title()}</h4>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(criar_cartao_estatistica("Média", formata_br(stats['mean']), "x̄", "Σx / n", "#3498DB"), unsafe_allow_html=True)
            with c2:
                st.markdown(criar_cartao_estatistica("Mediana", formata_br(stats['50%']), "Md", "Valor central (50%)", "#27AE60"), unsafe_allow_html=True)
            with c3:
                st.markdown(criar_cartao_estatistica("Desvio Padrão", formata_br(stats['std']), "σ", "√[Σ(x-x̄)²/(n-1)]", "#E74C3C"), unsafe_allow_html=True)
            with c4:
                val_min_max = f"<span style='font-size:16px;'>↓ {formata_br(stats['min'])} <br> ↑ {formata_br(stats['max'])}</span>"
                st.markdown(criar_cartao_estatistica("Mín / Máx", val_min_max, "↕", "Amplitude do período", "#F39C12"), unsafe_allow_html=True)
            st.write("")
        
        with st.expander("Visualizar Microdados Brutos (Macro)"):
            st.dataframe(df_raw_filtrado[['ano_mes'] + variaveis_selecionadas].style.format(precision=2), use_container_width=True)

# ------------------------------------------
# ABA 5: EXPLORADOR DE ELEMENTOS DE DESPESA
# ------------------------------------------
with aba5:
    st.subheader("Microdados: Estatística por Elemento de Despesa")
    st.markdown("Analise como o orçamento mensal foi distribuído entre diferentes naturezas de gasto (ex: Diárias, Obras, Equipamentos).")

    col_var5, col_start5, col_end5 = st.columns([2, 1, 1])
    min_date_agr = df_agrupado['ano_mes_lancamento'].min().to_pydatetime()
    max_date_agr = df_agrupado['ano_mes_lancamento'].max().to_pydatetime()
    
    with col_start5:
        data_inicio_agr = st.date_input("Início (Despesas)", min_value=min_date_agr, max_value=max_date_agr, value=min_date_agr)
    with col_end5:
        data_fim_agr = st.date_input("Término (Despesas)", min_value=min_date_agr, max_value=max_date_agr, value=max_date_agr)
        
    df_agr_filt = df_agrupado[(df_agrupado['ano_mes_lancamento'] >= pd.to_datetime(data_inicio_agr)) & (df_agrupado['ano_mes_lancamento'] <= pd.to_datetime(data_fim_agr))].copy()
    
    # Agrupando ranking de gastos no período
    df_ranking = df_agr_filt.groupby('Nome Elemento de Despesa')['valor_empenhado'].sum().reset_index()
    df_ranking = df_ranking.sort_values('valor_empenhado', ascending=False)
    todos_elementos = df_ranking['Nome Elemento de Despesa'].tolist()

    with col_var5:
        elementos_selecionados = st.multiselect(
            "Filtre os Elementos de Despesa específicos:",
            options=todos_elementos,
            default=todos_elementos[:3] if len(todos_elementos) >= 3 else todos_elementos
        )

    if elementos_selecionados:
        st.markdown("---")
        
        # Filtrar apenas os selecionados e preparar os dados
        df_agr_foco = df_agr_filt[df_agr_filt['Nome Elemento de Despesa'].isin(elementos_selecionados)]
        df_linha_agr = df_agr_foco.groupby(['ano_mes_lancamento', 'Nome Elemento de Despesa'])['valor_empenhado'].sum().reset_index()
        df_linha_agr = df_linha_agr.sort_values('ano_mes_lancamento')
        df_linha_agr['mes_ano_str'] = df_linha_agr['ano_mes_lancamento'].dt.strftime('%m/%Y')
        
        # Gráfico de Barras com Rótulos "R$ XX.XXX.XXX"
        fig_agr = px.bar(
            df_linha_agr, 
            x='mes_ano_str', 
            y='valor_empenhado', 
            color='Nome Elemento de Despesa', 
            barmode='group',
            labels={'valor_empenhado': '', 'mes_ano_str': 'Mês/Ano'}
        )
        
        # Adiciona o R$, coloca o valor arredondado acima da barra e remove a escala Y
        fig_agr.update_traces(textposition='outside', texttemplate='R$ %{y:,.0f}')
        fig_agr.update_yaxes(showticklabels=False, title='')
        
        meses_ordenados = df_linha_agr['mes_ano_str'].drop_duplicates().tolist()
        fig_agr.update_xaxes(type='category', categoryorder='array', categoryarray=meses_ordenados)
        
        fig_agr.update_layout(
            separators=",.", # Transforma o formato D3 (,0f) no padrão brasileiro (pontos nos milhares)
            hovermode="x unified", 
            height=450, 
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
            margin=dict(t=50) 
        )
        st.plotly_chart(fig_agr, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        for elemento in elementos_selecionados:
            df_elemento = df_linha_agr[df_linha_agr['Nome Elemento de Despesa'] == elemento]['valor_empenhado']
            
            if len(df_elemento) > 0:
                stats = df_elemento.describe()
                st.markdown(f"<h4 style='color: #2C3E50; border-bottom: 2px solid #f1f2f6; padding-bottom: 5px;'>📌 {elemento.title()}</h4>", unsafe_allow_html=True)
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(criar_cartao_estatistica("Média Mensal", f"R$ {formata_br(stats['mean'])}", "x̄", "Custo médio mensal", "#3498DB"), unsafe_allow_html=True)
                with c2:
                    st.markdown(criar_cartao_estatistica("Total no Período", f"R$ {formata_br(df_elemento.sum())}", "Σ", "Soma absoluta", "#27AE60"), unsafe_allow_html=True)
                with c3:
                    st.markdown(criar_cartao_estatistica("Desvio Padrão", f"R$ {formata_br(stats['std'])}", "σ", "Volatilidade do gasto", "#E74C3C"), unsafe_allow_html=True)
                with c4:
                    val_min_max = f"<span style='font-size:16px;'>↓ R$ {formata_br(stats['min'])} <br> ↑ R$ {formata_br(stats['max'])}</span>"
                    st.markdown(criar_cartao_estatistica("Mín / Máx", val_min_max, "↕", "Extremos de empenho", "#F39C12"), unsafe_allow_html=True)
                st.write("")
        
        with st.expander("Visualizar Microdados Brutos (Elementos de Despesa)"):
            df_pivot = df_linha_agr.pivot(index='ano_mes_lancamento', columns='Nome Elemento de Despesa', values='valor_empenhado').fillna(0)
            df_pivot.index = df_pivot.index.strftime('%Y-%m')
            st.dataframe(df_pivot.style.format("R$ {:,.2f}"), use_container_width=True)
    else:
        st.info(" Selecione pelo menos um elemento de despesa para gerar o dashboard analítico.")

    st.markdown("---")
    st.markdown("### Top 10 Maiores Despesas no Período Selecionado")
    
    df_top10 = df_ranking.head(10).copy()
    fig_bar = px.bar(df_top10, y='Nome Elemento de Despesa', x='valor_empenhado', orientation='h', 
                     color='valor_empenhado', color_continuous_scale='Blues')
    
    # Texto na barra lateral e expansão do eixo X para não cortar o R$
    fig_bar.update_traces(textposition='outside', texttemplate='R$ %{x:,.0f}', textfont=dict(color='black'))
    fig_bar.update_xaxes(showticklabels=False, title='', range=[0, df_top10['valor_empenhado'].max() * 1.3])
    fig_bar.update_yaxes(tickfont=dict(color='black', size=12))
    
    fig_bar.update_layout(
        separators=",.", # Padrão brasileiro
        yaxis={'categoryorder':'total ascending', 'title': ''}, 
        hovermode="y unified", 
        height=450, 
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)