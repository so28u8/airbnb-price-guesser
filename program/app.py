import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px
import plotly.graph_objects as go

from data_loader import load_data

#  Római szám 
_ROMAN = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,
          'X':10,'XI':11,'XII':12,'XIII':13,'XIV':14,'XV':15,'XVI':16,
          'XVII':17,'XVIII':18,'XIX':19,'XX':20,'XXI':21,'XXII':22,'XXIII':23}

def roman_to_int(name):
    m = re.match(r'^([IVXL]+)\.', str(name))
    return _ROMAN.get(m.group(1), 0) if m else 0

#  Konfiguráció 

st.set_page_config(page_title="Budapest Airbnb Elemző", page_icon="", layout="wide",
                   initial_sidebar_state="collapsed")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(145deg, #0a0a1a 0%, #111133 40%, #0d0d2b 100%) !important;
    color: #e0e0f0 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 2rem !important; max-width: 1300px !important; }

h1 {
    background: linear-gradient(135deg, #7b5dff 0%, #36d1dc 50%, #5b86e5 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 900 !important; font-size: 2.6rem !important;
    letter-spacing: -0.5px; margin-bottom: 0 !important;
}
h2, h3 { color: #c5c0ff !important; font-weight: 700 !important; }
p, span, label, .stMarkdown { color: #b0b0cc !important; }
hr { border-color: rgba(123, 93, 255, 0.15) !important; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04); backdrop-filter: blur(16px);
    border: 1px solid rgba(123,93,255,0.18); border-radius: 16px;
    padding: 20px 18px; transition: all 0.3s ease;
    box-shadow: 0 4px 30px rgba(0,0,0,0.25);
}
[data-testid="stMetric"]:hover {
    background: rgba(255,255,255,0.08); border-color: rgba(123,93,255,0.35);
    transform: translateY(-3px); box-shadow: 0 8px 40px rgba(123,93,255,0.15);
}
[data-testid="stMetricLabel"] {
    color: #8888bb !important; font-size: 0.82rem !important;
    font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.7px;
}
[data-testid="stMetricValue"] { color: #fff !important; font-weight: 800 !important; font-size: 1.7rem !important; }
[data-testid="stMetricDelta"] { font-weight: 600 !important; }

.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(255,255,255,0.03); border-radius: 14px;
    padding: 6px; border: 1px solid rgba(123,93,255,0.1);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 12px 28px; font-weight: 600;
    font-size: 0.95rem; color: #8888bb !important; transition: all 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #c5c0ff !important; background: rgba(123,93,255,0.08); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(123,93,255,0.25), rgba(54,209,220,0.15)) !important;
    color: #fff !important; border: 1px solid rgba(123,93,255,0.35);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #7b5dff 0%, #36d1dc 100%) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    padding: 14px 28px !important; font-weight: 700 !important; font-size: 1rem !important;
    transition: all 0.3s ease !important; box-shadow: 0 4px 20px rgba(123,93,255,0.25) !important;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 6px 30px rgba(123,93,255,0.45) !important; transform: translateY(-2px) !important;
}

[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(123,93,255,0.15) !important;
    border-radius: 10px !important; color: #e0e0f0 !important;
}
.stRadio > div {
    background: rgba(255,255,255,0.03); border-radius: 12px;
    padding: 8px 12px; border: 1px solid rgba(123,93,255,0.1);
}
[data-testid="stAlert"] {
    background: rgba(123,93,255,0.08) !important;
    border: 1px solid rgba(123,93,255,0.2) !important;
    border-radius: 14px !important; color: #c5c0ff !important;
}
.hero-subtitle { font-size: 1.1rem; color: #8888bb; margin-top: -6px; margin-bottom: 6px; }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: fadeInUp 0.7s ease-out forwards; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    df = load_data("../listings.csv")
    q1, q3 = df['price'].quantile(0.25), df['price'].quantile(0.75)
    iqr = q3 - q1
    low, high = max(q1 - 1.5 * iqr, 1), q3 + 1.5 * iqr
    df = df[(df['price'] >= low) & (df['price'] <= high)]
    df['district_num']  = df['neighbourhood_cleansed'].apply(roman_to_int).astype(float)
    df['review_weight'] = np.log1p(df['number_of_reviews']) * df['review_scores_rating']
    return df

@st.cache_resource
def get_model():
    return joblib.load("model/price_model.pkl")

@st.cache_data
def get_log_target():
    try: return joblib.load("model/log_target.pkl")
    except FileNotFoundError: return False

df, model, log_target = get_data(), get_model(), get_log_target()

district_stats = (
    df.groupby('neighbourhood_cleansed')
    .agg(avg_price=('price','mean'), median_price=('price','median'),
         count=('price','count'), avg_lat=('latitude','mean'), avg_lon=('longitude','mean'))
    .reset_index()
)


PLOT_BG = "rgba(0,0,0,0)"
COLORSCALE = [[0,"#1a1a4e"],[.15,"#2d2d8e"],[.3,"#5b3dbb"],[.5,"#7b5dff"],
              [.65,"#36d1dc"],[.8,"#5bffa5"],[.9,"#ffe066"],[1,"#ff6b6b"]]
COLORSCALE_WARM = [[0,"#1a1a4e"],[.25,"#5b3dbb"],[.5,"#7b5dff"],[.75,"#ff8c42"],[1,"#ff4757"]]
GRID = "rgba(123,93,255,0.08)"

def dark_layout(fig, h=550):
    fig.update_layout(template="plotly_dark", paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      height=h, font=dict(family="Inter", color="#b0b0cc"), title_text="",
                      margin=dict(r=15,t=15,l=15,b=15),
                      xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
                      yaxis=dict(gridcolor=GRID, zerolinecolor=GRID))
    return fig

COLORBAR = dict(tickfont=dict(color="#8888bb"), bgcolor="rgba(0,0,0,0)", borderwidth=0, len=0.6)

#  Fejléc 

st.markdown('<div class="animate-in">', unsafe_allow_html=True)
st.title("Budapest Airbnb Piaci Elemző")
st.markdown('<p class="hero-subtitle">Valós piaci adatokon alapuló interaktív dashboard és ML-alapú árbecslő</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Aktív hirdetések", f"{len(df):,}")
col2.metric("Medián ár / éj",  f"{df['price'].median():,.0f} Ft")
col3.metric("Átlag értékelés",  f"{df['review_scores_rating'].mean():.2f} / 5")
col4.metric("Kerületek száma",  df['neighbourhood_cleansed'].nunique())
st.divider()

# TAB 1: Piaci térkép


tab1, tab2 = st.tabs(["Piaci térkép", "Árbecslő"])

with tab1:
    st.subheader("Átlagárak kerületek szerint")
    map_mode = st.radio("Megjelenítés:", ["Hőtérkép (sűrűség)", "Kerületi buborékok"], horizontal=True)

    map_center = {"lat": 47.497, "lon": 19.040}

    if map_mode == "Hőtérkép (sűrűség)":
        fig_map = px.density_mapbox(df, lat='latitude', lon='longitude', z='price', radius=14,
                                    center=map_center, zoom=11.2, mapbox_style="carto-darkmatter",
                                    color_continuous_scale=COLORSCALE_WARM, labels={'z':'Ár (Ft/éj)'}, height=620)
        fig_map.update_layout(title_text="", margin=dict(r=0,t=0,l=0,b=0), paper_bgcolor=PLOT_BG,
                              coloraxis_colorbar={**COLORBAR, "title": dict(text="Ár (Ft/éj)", font=dict(color="#c5c0ff"))})
    else:
        fig_map = px.scatter_mapbox(district_stats, lat='avg_lat', lon='avg_lon', size='count',
                                    color='avg_price', hover_name='neighbourhood_cleansed',
                                    hover_data={'avg_price':':,.0f','median_price':':,.0f','count':True,'avg_lat':False,'avg_lon':False},
                                    color_continuous_scale=COLORSCALE, size_max=45, zoom=11.2,
                                    center=map_center, mapbox_style="carto-darkmatter",
                                    labels={'avg_price':'Átlagár (Ft/éj)','median_price':'Medián ár (Ft/éj)','count':'Hirdetések száma'}, height=620)
        fig_map.update_layout(title_text="", margin=dict(r=0,t=0,l=0,b=0), paper_bgcolor=PLOT_BG,
                              coloraxis_colorbar={**COLORBAR, "title": dict(text="Átlagár", font=dict(color="#c5c0ff"))})
        fig_map.update_traces(marker=dict(opacity=0.85, sizemin=8))

    st.plotly_chart(fig_map, use_container_width=True)

    # Top 10 kerület
    st.markdown("---")
    st.subheader("Top 10 legdrágább kerület")
    top10 = district_stats.sort_values('avg_price', ascending=False).head(10)

    fig_bar = go.Figure(go.Bar(
        x=top10['neighbourhood_cleansed'], y=top10['avg_price'],
        marker=dict(color=top10['avg_price'], colorscale=COLORSCALE, cornerradius=6, line_width=0),
        opacity=0.9, hovertemplate='<b>%{x}</b><br>Átlagár: %{y:,.0f} Ft/éj<extra></extra>',
    ))
    dark_layout(fig_bar, 420)
    fig_bar.update_layout(xaxis_tickangle=-30, showlegend=False, xaxis_title="", yaxis_title="Átlagár (Ft/éj)")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Átlagár szobatípus szerint
    st.markdown("---")
    st.subheader("Átlagár szobatípus szerint")

    rt_colors = {"Entire home/apt":"#7b5dff", "Private room":"#36d1dc", "Shared room":"#ff6b6b", "Hotel room":"#ffe066"}
    rt = df.groupby('room_type').agg(avg=('price','mean'), median=('price','median')).reset_index().sort_values('avg', ascending=False)

    fig_rt = go.Figure()
    for name, col_y, opacity in [('Átlagár','avg',0.9), ('Medián ár','median',0.5)]:
        fig_rt.add_trace(go.Bar(
            x=rt['room_type'], y=rt[col_y], name=name, opacity=opacity,
            marker_color=[rt_colors.get(r,'#7b5dff') for r in rt['room_type']], marker_cornerradius=6,
            hovertemplate=f'<b>%{{x}}</b><br>{name}: %{{y:,.0f}} Ft/éj<extra></extra>',
        ))
    dark_layout(fig_rt, 420)
    fig_rt.update_layout(barmode='group', xaxis_title="", yaxis_title="Ár (Ft/éj)", legend=dict(font=dict(color='#b0b0cc')))
    st.plotly_chart(fig_rt, use_container_width=True)

    # Ár-eloszlás
    st.markdown("---")
    st.subheader("Ár-eloszlás")
    fig_hist = px.histogram(df, x='price', nbins=80, labels={'price':'Ár (Ft/éj)'}, color_discrete_sequence=["#7b5dff"])
    dark_layout(fig_hist, 350)
    fig_hist.update_traces(marker_line_width=0, opacity=0.8)
    fig_hist.update_layout(yaxis_title="Hirdetések száma", bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)


# TAB 2 Árbecslő

with tab2:
    st.markdown("""
    <div style="text-align:center; padding:18px 0 10px;">
        <span style="font-size:2.8rem; display:block; margin-bottom:4px;"></span>
        <h2 style="margin:0; font-size:1.6rem; background:linear-gradient(135deg,#7b5dff,#36d1dc);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:800;">
            Éjszakánkénti irányár becslése</h2>
        <p style="margin:8px 0 0; color:#8888bb; font-size:0.95rem;">
            Add meg a lakásod paramétereit, és a modell megbecsüli a piaci szempontból optimális napi árat.</p>
    </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("##### Elhelyezkedés")
        neighbourhood = st.selectbox("Kerület / negyed", options=sorted(df['neighbourhood_cleansed'].unique()))
        room_type = st.selectbox("Szobatípus", options=df['room_type'].unique(),
                                 help="Entire home/apt = egész lakás, Private room = privát szoba")

        st.markdown("##### Lakás mérete")
        bedrooms     = st.slider("Hálószobák száma", 0, 10, 1)
        bathrooms    = st.slider("Fürdőszobák száma", 0, 6, 1)

    with col_right:
        st.markdown("##### Értékelések")
        has_rating = st.toggle("Van már értékelés?", value=True)

        if has_rating:
            review_scores_rating      = st.slider("Átlagos értékelés (1–5)", 1.0, 5.0, 4.5, step=0.1)
            review_scores_cleanliness = st.slider("Tisztaság értékelés (1–5)", 1.0, 5.0, 4.7, step=0.1)
            review_scores_location    = st.slider("Lokáció értékelés (1–5)", 1.0, 5.0, 4.8, step=0.1)
            number_of_reviews         = st.slider("Értékelések száma", 1, 500, 20)
        else:
            review_scores_rating      = df['review_scores_rating'].median()
            review_scores_cleanliness = df['review_scores_cleanliness'].median()
            review_scores_location    = df['review_scores_location'].median()
            number_of_reviews         = 0

        st.markdown("##### Egyéb")
        minimum_nights    = st.slider("Minimum éjszakák száma", 1, 30, 2)
        host_is_superhost = st.toggle("Superhost?", value=False)
        instant_bookable  = st.toggle("Azonnali foglalás?", value=True)

    # Származtatott értékek
    ker = district_stats[district_stats['neighbourhood_cleansed'] == neighbourhood].iloc[0]
    district_num = float(roman_to_int(neighbourhood))

    if st.button("Ár becslése", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            'calculated_host_listings_count': 1,
            'bathrooms': bathrooms,
            'bedrooms': bedrooms,
            'longitude': ker['avg_lon'], 'latitude': ker['avg_lat'],
            'review_scores_rating': review_scores_rating,
            'minimum_nights': minimum_nights,
            'review_scores_cleanliness': review_scores_cleanliness,
            'review_scores_location': review_scores_location,
            'availability_30': int(df['availability_30'].median()),
            'district_num': district_num,
            'host_response_rate': float(df['host_response_rate'].median()),
            'review_weight': np.log1p(number_of_reviews) * review_scores_rating,
            'number_of_reviews': number_of_reviews,
            'room_type': room_type,
            'neighbourhood_cleansed': neighbourhood,
            'property_type_grouped': df['property_type_grouped'].mode().iloc[0],
        }])

        pred = model.predict(input_df)[0]
        predicted_price = max(np.expm1(pred) if log_target else pred, 0)
        diff_pct = (predicted_price - ker['avg_price']) / ker['avg_price'] * 100
        arrow, color = ("↑","#5bffa5") if diff_pct > 0 else ("↓","#ff6b6b")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(123,93,255,0.12),rgba(54,209,220,0.08));
            border:1px solid rgba(123,93,255,0.25); border-radius:20px; padding:36px 28px;
            text-align:center; margin:8px 0 20px;">
            <div style="color:#8888bb; font-size:0.85rem; text-transform:uppercase; letter-spacing:1.2px; font-weight:600;">
                Becsült irányár</div>
            <div style="font-size:3rem; font-weight:900; background:linear-gradient(135deg,#7b5dff,#36d1dc);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:8px 0; line-height:1.1;">
                {predicted_price:,.0f} Ft/éj</div>
            <div style="display:inline-block; background:rgba(123,93,255,0.1); border:1px solid rgba(123,93,255,0.2);
                border-radius:10px; padding:8px 18px; margin-top:10px;">
                <span style="color:#8888bb; font-size:0.85rem;">Kerületi átlag:</span>
                <span style="color:#e0e0f0; font-weight:700; margin-left:6px;">{ker['avg_price']:,.0f} Ft/éj</span>
                <span style="color:{color}; font-weight:700; margin-left:10px;">{arrow} {abs(diff_pct):.1f}%</span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.info(f"A modell szerint a megadott paraméterekkel **{predicted_price:,.0f} Ft/éj** a piaci irányár. "
                f"Ez a(z) *{neighbourhood}* kerületi átlagnál {'magasabb' if diff_pct > 0 else 'alacsonyabb'} "
                f"({abs(diff_pct):.1f}%-kal).")

#labléc
st.divider()
st.markdown("""<div style="text-align:center; padding:10px 0 20px;">
    <span style="color:#444466; font-size:0.8rem;">
        Adatforrás: <a href="https://insideairbnb.com" style="color:#7b5dff; text-decoration:none;">Inside Airbnb</a>
        · Budapest · 2024</span></div>""", unsafe_allow_html=True)