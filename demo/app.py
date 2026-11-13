import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

HERE = Path(__file__).parent


@st.cache_data
def load():
    df = pd.read_parquet(HERE / "predictions.parquet")
    df["loc"] = df["zone"].astype(str)
    df["day"] = df["ts_target"].dt.date
    df["time"] = df["ts_target"].dt.strftime("%H:%M")
    geo = json.loads((HERE / "zones.geojson").read_text(encoding="utf-8"))
    return df, geo


st.set_page_config(page_title="DemandCast", layout="wide")
st.title("DemandCast: где будет спрос через 30 минут")
st.caption("Воспроизведение тестового периода (март 2024). Прогноз сделан за 30 минут до показанного времени; "
           "данные TLC публикуются с задержкой, поэтому режима реального времени нет.")

df, geo = load()
left, right = st.columns(2)
day = left.selectbox("День", sorted(df["day"].unique()))
slots = sorted(df.loc[df["day"] == day, "time"].unique())
time = right.select_slider("Время", options=slots, value=slots[len(slots) // 2])
now = df[(df["day"] == day) & (df["time"] == time)].sort_values("pred", ascending=False)

fig = px.choropleth_map(now, geojson=geo, locations="loc", color="pred", hover_name="zone_name",
                        hover_data={"loc": False, "pred": ":.0f", "lo": ":.0f", "hi": ":.0f", "y": True},
                        labels={"pred": "прогноз", "lo": "от", "hi": "до", "y": "факт"},
                        color_continuous_scale="YlOrRd", map_style="carto-positron",
                        center={"lat": 40.75, "lon": -73.95}, zoom=10, opacity=0.75, height=520)
st.plotly_chart(fig)

st.subheader("Топ-10 районов по прогнозу")
top = now.head(10)[["zone_name", "pred", "lo", "hi", "y"]].round(0)
top.columns = ["Район", "Прогноз", "От (10%)", "До (90%)", "Факт"]
st.dataframe(top, hide_index=True)
